"""
Processing manager for video captioning with multi-GPU parallel processing
"""

import asyncio
import time
import torch
import threading
from pathlib import Path
from typing import Callable, Optional, List, Any, Dict
from dataclasses import dataclass, field

from backend.schemas import (
    Settings, ProgressUpdate, ProcessingStage, ProcessingSubstage,
    VideoInfo, WorkerProgress, MediaType
)


@dataclass
class WorkerState:
    """State for a single GPU worker"""
    worker_id: int
    device: str
    current_video: Optional[str] = None
    substage: ProcessingSubstage = ProcessingSubstage.IDLE
    substage_progress: float = 0.0
    model_info: Optional[Dict[str, Any]] = None
    is_busy: bool = False
    error: Optional[str] = None

    def to_worker_progress(self) -> WorkerProgress:
        return WorkerProgress(
            worker_id=self.worker_id,
            device=self.device,
            current_video=self.current_video,
            substage=self.substage,
            substage_progress=self.substage_progress,
        )


@dataclass
class ProcessingState:
    """Mutable state for tracking processing progress"""
    stage: ProcessingStage = ProcessingStage.IDLE
    current_video: Optional[str] = None
    video_index: int = 0
    total_videos: int = 0
    completed_videos: int = 0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    model_loaded: bool = False
    vram_used_gb: float = 0.0
    substage: ProcessingSubstage = ProcessingSubstage.IDLE
    substage_progress: float = 0.0
    error_message: Optional[str] = None
    start_time: float = 0.0
    # Multi-GPU fields
    batch_size: int = 1
    workers: List[WorkerState] = field(default_factory=list)
    # Transient completion event (cleared after each emit)
    _just_completed_video: Optional[str] = None
    _just_completed_caption_preview: Optional[str] = None

    def to_progress_update(self) -> ProgressUpdate:
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0.0
        return ProgressUpdate(
            stage=self.stage,
            current_video=self.current_video,
            video_index=self.video_index,
            total_videos=self.total_videos,
            completed_videos=self.completed_videos,
            tokens_generated=self.tokens_generated,
            tokens_per_sec=self.tokens_per_sec,
            model_loaded=self.model_loaded,
            vram_used_gb=self.vram_used_gb,
            substage=self.substage,
            substage_progress=self.substage_progress,
            error_message=self.error_message,
            elapsed_time=elapsed,
            batch_size=self.batch_size,
            workers=[w.to_worker_progress() for w in self.workers],
            just_completed_video=self._just_completed_video,
            just_completed_caption_preview=self._just_completed_caption_preview,
        )


class ProcessingManager:
    """
    Manages video processing with real-time progress updates.
    Supports multi-GPU parallel processing.
    """

    def __init__(self, progress_callback: Optional[Callable[[ProgressUpdate], Any]] = None):
        self.progress_callback = progress_callback
        self.model_info: Optional[Dict[str, Any]] = None  # For single GPU
        self.model_infos: Dict[str, Dict[str, Any]] = {}  # For multi-GPU: device -> model_info
        self.should_stop = False
        self.is_processing = False
        self.state = ProcessingState()
        self._lock = asyncio.Lock()
        self._tokens_lock = threading.Lock()
        print("[ProcessingManager] Initialized")

    async def emit_progress(self):
        """Send current progress to callback"""
        if self.progress_callback:
            update = self.state.to_progress_update()
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(update)
            else:
                self.progress_callback(update)
        # Clear transient completion event after sending
        self.state._just_completed_video = None
        self.state._just_completed_caption_preview = None

    def _get_display_name(self, video_path: Path) -> str:
        """Get display name matching frontend VideoInfo.name format (relative path with forward slashes)"""
        from backend import config as _config
        working_dir = _config.get_working_directory()
        try:
            return str(video_path.relative_to(working_dir)).replace('\\', '/')
        except ValueError:
            return video_path.name

    def _update_vram(self):
        """Update VRAM usage (sum across all GPUs)"""
        if torch.cuda.is_available():
            total_vram = 0
            for i in range(torch.cuda.device_count()):
                total_vram += torch.cuda.memory_allocated(i)
            self.state.vram_used_gb = total_vram / (1024 ** 3)

    async def load_model(self, settings: Settings) -> bool:
        """
        Load the model with specified settings.
        For single GPU (batch_size=1) or first GPU in multi-GPU setup.
        Returns True on success, False on failure.
        """
        from backend.model_loader import load_model, clear_cache

        print(f"[ProcessingManager] load_model called with model_id={settings.model_id}")

        async with self._lock:
            try:
                print("[ProcessingManager] Acquired lock, starting model load")
                self.state.stage = ProcessingStage.LOADING_MODEL
                self.state.substage = ProcessingSubstage.IDLE
                self.state.substage_progress = 0.0
                self.state.start_time = time.time()
                await self.emit_progress()

                # Clear existing model if any
                if self.model_info is not None:
                    clear_cache()
                    self.model_info = None
                    self.model_infos.clear()

                self.state.substage_progress = 0.1
                await self.emit_progress()

                # Load model in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                device = settings.device.value

                self.model_info = await loop.run_in_executor(
                    None,
                    lambda: load_model(
                        model_id=settings.model_id,
                        device=device,
                        dtype=settings.dtype.value,
                        use_sage_attention=settings.use_sage_attention,
                        use_torch_compile=settings.use_torch_compile,
                    )
                )

                # Store in model_infos dict for consistency
                self.model_infos[device] = self.model_info

                self.state.model_loaded = True
                self.state.substage_progress = 1.0
                self._update_vram()
                self.state.stage = ProcessingStage.IDLE
                await self.emit_progress()

                print(f"[ProcessingManager] Model loaded successfully, VRAM: {self.state.vram_used_gb:.2f} GB")
                return True

            except Exception as e:
                print(f"[ProcessingManager] Model load FAILED: {e}")
                import traceback
                traceback.print_exc()
                self.state.stage = ProcessingStage.ERROR
                self.state.error_message = str(e)
                self.state.model_loaded = False
                await self.emit_progress()
                return False

    async def load_models_parallel(self, settings: Settings, devices: List[str]) -> bool:
        """Load model copies on multiple GPUs in parallel"""
        from backend.model_loader import load_model, clear_cache

        print(f"[ProcessingManager] Loading models on {len(devices)} devices: {devices}")

        self.state.stage = ProcessingStage.LOADING_MODEL
        self.state.substage = ProcessingSubstage.IDLE
        self.state.substage_progress = 0.0
        self.state.start_time = time.time()
        await self.emit_progress()

        # Clear existing models
        clear_cache()
        self.model_info = None
        self.model_infos.clear()

        loop = asyncio.get_event_loop()
        loaded_count = 0
        progress_per_device = 1.0 / len(devices)

        # Load models sequentially to avoid memory issues during parallel loading
        for device in devices:
            if self.should_stop:
                break

            try:
                print(f"[ProcessingManager] Loading model on {device}...")
                model_info = await loop.run_in_executor(
                    None,
                    lambda d=device: load_model(
                        model_id=settings.model_id,
                        device=d,
                        dtype=settings.dtype.value,
                        use_sage_attention=settings.use_sage_attention,
                        use_torch_compile=settings.use_torch_compile,
                    )
                )
                self.model_infos[device] = model_info
                loaded_count += 1
                self.state.substage_progress += progress_per_device
                self._update_vram()
                await self.emit_progress()
                print(f"[ProcessingManager] Model loaded on {device}, VRAM: {self.state.vram_used_gb:.2f} GB")
            except Exception as e:
                print(f"[ProcessingManager] Failed to load model on {device}: {e}")
                # Continue with remaining GPUs

        if loaded_count == 0:
            self.state.stage = ProcessingStage.ERROR
            self.state.error_message = "Failed to load model on any GPU"
            await self.emit_progress()
            return False

        # Set primary model_info for backward compatibility
        if self.model_infos:
            self.model_info = list(self.model_infos.values())[0]

        self.state.model_loaded = True
        self.state.substage_progress = 1.0
        self._update_vram()
        self.state.stage = ProcessingStage.IDLE
        await self.emit_progress()

        print(f"[ProcessingManager] Models loaded on {loaded_count}/{len(devices)} devices")
        return True

    async def process_videos(
        self,
        videos: List[Path],
        settings: Settings,
    ) -> List[Dict[str, Any]]:
        """
        Process videos - dispatches to parallel or sequential based on batch_size.
        """
        if settings.batch_size > 1:
            return await self._process_videos_parallel(videos, settings)
        else:
            return await self._process_videos_sequential(videos, settings)

    async def _process_videos_parallel(
        self,
        videos: List[Path],
        settings: Settings,
    ) -> List[Dict[str, Any]]:
        """
        Process media (videos and images) in parallel across multiple GPUs.
        """
        from backend.model_loader import generate_caption
        from backend.video_processor import process_video, process_image
        from backend import config

        batch_size = min(settings.batch_size, len(videos))
        devices = [f"cuda:{i}" for i in range(batch_size)]

        print(f"[ProcessingManager] Processing {len(videos)} videos with {batch_size} workers on {devices}")

        # Load models if needed
        if not self.model_infos or len(self.model_infos) < batch_size:
            success = await self.load_models_parallel(settings, devices)
            if not success:
                return []

        async with self._lock:
            self.is_processing = True
            self.should_stop = False
            self.state.stage = ProcessingStage.PROCESSING
            self.state.total_videos = len(videos)
            self.state.video_index = 0
            self.state.completed_videos = 0
            self.state.batch_size = batch_size
            self.state.start_time = time.time()

            # Initialize worker states
            self.state.workers = [
                WorkerState(worker_id=i, device=devices[i])
                for i in range(batch_size)
            ]

            await self.emit_progress()

            results = []
            video_queue = list(videos)
            active_tasks: Dict[int, asyncio.Task] = {}

            loop = asyncio.get_event_loop()

            async def process_single_video(worker_id: int, video_path: Path):
                """Process a single video on a specific worker/GPU"""
                device = devices[worker_id]
                model_info = self.model_infos.get(device)

                if not model_info:
                    return {
                        "video": video_path.name,
                        "success": False,
                        "error": f"No model loaded on {device}",
                        "caption": None,
                        "worker_id": worker_id,
                    }

                worker = self.state.workers[worker_id]

                worker.is_busy = True
                worker.current_video = self._get_display_name(video_path)
                worker.substage = ProcessingSubstage.EXTRACTING_FRAMES
                worker.substage_progress = 0.0
                await self.emit_progress()

                result = {
                    "video": video_path.name,
                    "success": False,
                    "error": None,
                    "caption": None,
                    "worker_id": worker_id,
                }

                try:
                    # Extract frames (detect image vs video by extension)
                    worker.substage_progress = 0.2
                    await self.emit_progress()

                    is_image = video_path.suffix.lower() in config.IMAGE_EXTENSIONS
                    if is_image:
                        frames, video_meta = await loop.run_in_executor(
                            None,
                            lambda: process_image(
                                video_path,
                                frame_size=settings.frame_size,
                            )
                        )
                    else:
                        frames, video_meta = await loop.run_in_executor(
                            None,
                            lambda: process_video(
                                video_path,
                                max_frames=settings.max_frames,
                                frame_size=settings.frame_size,
                            )
                        )

                    worker.substage = ProcessingSubstage.ENCODING
                    worker.substage_progress = 0.4
                    await self.emit_progress()

                    # Generate caption
                    worker.substage = ProcessingSubstage.GENERATING
                    worker.substage_progress = 0.5
                    await self.emit_progress()

                    caption, gen_meta = await loop.run_in_executor(
                        None,
                        lambda: generate_caption(
                            model_info=model_info,
                            images=frames,
                            prompt=settings.prompt,
                            max_tokens=settings.max_tokens,
                            temperature=settings.temperature,
                        )
                    )

                    # Thread-safe token counter update
                    with self._tokens_lock:
                        self.state.tokens_generated += gen_meta["output_tokens"]
                        self.state.tokens_per_sec = gen_meta["tokens_per_sec"]

                    self._update_vram()

                    # Save caption
                    worker.substage_progress = 0.9
                    await self.emit_progress()

                    output_path = video_path.parent / (video_path.stem + config.OUTPUT_EXTENSION)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                        if settings.include_metadata:
                            f.write("\n\n" + "=" * 60 + "\n")
                            f.write("METADATA\n")
                            f.write("=" * 60 + "\n")
                            f.write(f"Video: {video_path.name}\n")
                            f.write(f"Worker: {worker_id} ({device})\n")
                            f.write(f"Frames processed: {gen_meta['num_frames']}\n")
                            f.write(f"Output tokens: {gen_meta['output_tokens']}\n")
                            f.write(f"Tokens/sec: {gen_meta['tokens_per_sec']:.1f}\n")

                    result["success"] = True
                    result["caption"] = caption[:200] + "..." if len(caption) > 200 else caption
                    result["output_path"] = str(output_path)

                except Exception as e:
                    result["error"] = str(e)
                    worker.error = str(e)
                    print(f"[ProcessingManager] Worker {worker_id} error processing {video_path.name}: {e}")

                finally:
                    worker.substage_progress = 1.0
                    worker.is_busy = False
                    worker.current_video = None
                    worker.substage = ProcessingSubstage.IDLE
                    self.state.completed_videos += 1
                    self.state._just_completed_video = self._get_display_name(video_path)
                    if result.get("success") and result.get("caption"):
                        caption_text = result["caption"]
                        self.state._just_completed_caption_preview = caption_text[:150] + "..." if len(caption_text) > 150 else caption_text
                    await self.emit_progress()

                return result

            # Main processing loop - distribute work across workers
            while video_queue or active_tasks:
                if self.should_stop:
                    # Cancel all active tasks
                    for task in active_tasks.values():
                        task.cancel()
                    break

                # Start new tasks on available workers
                for worker_id in range(batch_size):
                    if worker_id not in active_tasks and video_queue:
                        video_path = video_queue.pop(0)
                        self.state.video_index = len(videos) - len(video_queue) - len(active_tasks)
                        # Update current_video to first active video for backward compat
                        self.state.current_video = self._get_display_name(video_path)
                        task = asyncio.create_task(process_single_video(worker_id, video_path))
                        active_tasks[worker_id] = task

                if not active_tasks:
                    break

                # Wait for any task to complete
                done, _ = await asyncio.wait(
                    active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Collect results and free workers
                for task in done:
                    try:
                        result = task.result()
                        results.append(result)
                        worker_id = result["worker_id"]
                        del active_tasks[worker_id]
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        print(f"[ProcessingManager] Task error: {e}")

            # Complete
            self.state.stage = ProcessingStage.COMPLETE
            self.state.substage = ProcessingSubstage.IDLE
            self.state.current_video = None
            self.is_processing = False
            await self.emit_progress()

        return results

    async def _process_videos_sequential(
        self,
        videos: List[Path],
        settings: Settings,
    ) -> List[Dict[str, Any]]:
        """
        Process a list of media files (videos and images) sequentially (original single-GPU behavior).
        Returns list of results for each file.
        """
        from backend.model_loader import generate_caption
        from backend.video_processor import process_video, process_image
        from backend import config

        print(f"[ProcessingManager] process_videos called with {len(videos)} videos")
        for v in videos:
            print(f"  - {v}")

        results = []

        print(f"[ProcessingManager] Checking model state. model_loaded={self.state.model_loaded}, model_info={self.model_info is not None}")
        if not self.state.model_loaded or self.model_info is None:
            # Load model first (load_model acquires its own lock)
            print("[ProcessingManager] Model not loaded, loading now...")
            success = await self.load_model(settings)
            if not success:
                print("[ProcessingManager] Model load failed, returning early")
                return results
            print("[ProcessingManager] Model load succeeded")

        async with self._lock:
            print("[ProcessingManager] Acquired lock, starting video processing loop")
            self.is_processing = True
            self.should_stop = False
            self.state.stage = ProcessingStage.PROCESSING
            self.state.total_videos = len(videos)
            self.state.video_index = 0
            self.state.completed_videos = 0
            self.state.batch_size = 1
            self.state.workers = []  # No workers for sequential
            self.state.start_time = time.time()
            await self.emit_progress()

            loop = asyncio.get_event_loop()

            for i, video_path in enumerate(videos):
                if self.should_stop:
                    break

                self.state.video_index = i
                self.state.current_video = self._get_display_name(video_path)
                self.state.substage = ProcessingSubstage.EXTRACTING_FRAMES
                self.state.substage_progress = 0.0
                await self.emit_progress()

                result = {
                    "video": video_path.name,
                    "success": False,
                    "error": None,
                    "caption": None,
                }

                try:
                    # Extract frames (detect image vs video by extension)
                    self.state.substage_progress = 0.2
                    await self.emit_progress()

                    is_image = video_path.suffix.lower() in config.IMAGE_EXTENSIONS
                    if is_image:
                        frames, video_meta = await loop.run_in_executor(
                            None,
                            lambda: process_image(
                                video_path,
                                frame_size=settings.frame_size,
                            )
                        )
                    else:
                        frames, video_meta = await loop.run_in_executor(
                            None,
                            lambda: process_video(
                                video_path,
                                max_frames=settings.max_frames,
                                frame_size=settings.frame_size,
                            )
                        )

                    self.state.substage = ProcessingSubstage.ENCODING
                    self.state.substage_progress = 0.4
                    await self.emit_progress()

                    # Generate caption
                    self.state.substage = ProcessingSubstage.GENERATING
                    self.state.substage_progress = 0.5
                    await self.emit_progress()

                    caption, gen_meta = await loop.run_in_executor(
                        None,
                        lambda: generate_caption(
                            model_info=self.model_info,
                            images=frames,
                            prompt=settings.prompt,
                            max_tokens=settings.max_tokens,
                            temperature=settings.temperature,
                        )
                    )

                    self.state.tokens_generated += gen_meta["output_tokens"]
                    self.state.tokens_per_sec = gen_meta["tokens_per_sec"]
                    self._update_vram()

                    # Save caption to same directory as video
                    self.state.substage_progress = 0.9
                    await self.emit_progress()

                    output_path = video_path.parent / (video_path.stem + config.OUTPUT_EXTENSION)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                        if settings.include_metadata:
                            f.write("\n\n" + "=" * 60 + "\n")
                            f.write("METADATA\n")
                            f.write("=" * 60 + "\n")
                            f.write(f"Video: {video_path.name}\n")
                            f.write(f"Frames processed: {gen_meta['num_frames']}\n")
                            f.write(f"Output tokens: {gen_meta['output_tokens']}\n")
                            f.write(f"Tokens/sec: {gen_meta['tokens_per_sec']:.1f}\n")

                    result["success"] = True
                    result["caption"] = caption[:200] + "..." if len(caption) > 200 else caption
                    result["output_path"] = str(output_path)

                    self.state.substage_progress = 1.0
                    self.state.completed_videos += 1
                    self.state._just_completed_video = self._get_display_name(video_path)
                    preview = caption[:150] + "..." if len(caption) > 150 else caption
                    self.state._just_completed_caption_preview = preview
                    await self.emit_progress()

                except Exception as e:
                    result["error"] = str(e)
                    self.state.error_message = f"Error processing {video_path.name}: {e}"
                    await self.emit_progress()

                results.append(result)

            # Complete
            self.state.stage = ProcessingStage.COMPLETE
            self.state.substage = ProcessingSubstage.IDLE
            self.state.current_video = None
            self.is_processing = False
            await self.emit_progress()

        return results

    def stop(self):
        """Request processing to stop"""
        self.should_stop = True

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        self._update_vram()

        # For multi-GPU, report status of all loaded models
        devices_loaded = list(self.model_infos.keys()) if self.model_infos else []

        return {
            "loaded": self.state.model_loaded,
            "model_id": self.model_info.get("model_id") if self.model_info else None,
            "device": str(self.model_info.get("device")) if self.model_info else None,
            "devices_loaded": devices_loaded,
            "vram_used_gb": self.state.vram_used_gb,
            "sage_attention_active": self.model_info.get("sage_attention", False) if self.model_info else False,
            "torch_compiled": self.model_info.get("torch_compiled", False) if self.model_info else False,
        }

    async def unload_model(self):
        """Unload model and free GPU memory"""
        from backend.model_loader import clear_cache

        async with self._lock:
            # Clear all model references first
            self.model_info = None
            self.model_infos.clear()

            # Clear worker states
            for worker in self.state.workers:
                worker.model_info = None

            self.state.workers.clear()
            self.state.model_loaded = False

            # Now clear cache and GPU memory
            clear_cache()

            self._update_vram()
            await self.emit_progress()

            print("[ProcessingManager] Model unloaded and GPU memory freed")

    def reset(self):
        """Reset processing state"""
        self.state = ProcessingState()
        self.should_stop = False
        self.is_processing = False
