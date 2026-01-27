"""
FastAPI backend for Video Caption Suite
Provides REST API and WebSocket for real-time progress
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Set
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import hashlib
import subprocess
import io
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from backend.schemas import (
    Settings, SettingsUpdate, ProgressUpdate, VideoInfo, VideoListResponse,
    CaptionInfo, CaptionListResponse, ProcessingRequest, ProcessingResponse,
    ModelStatus, ErrorResponse, ProcessingStage, GPUInfoResponse,
    SavedPrompt, PromptLibrary, CreatePromptRequest, UpdatePromptRequest,
    DirectoryRequest, DirectoryResponse, DirectoryBrowseResponse, MediaType,
    # Analytics schemas
    StopwordPreset, WordFrequencyRequest, WordFrequencyResponse, WordFrequencyItem,
    NgramRequest, NgramResponse, NgramItem,
    CorrelationRequest, CorrelationResponse, CorrelationItem,
    AnalyticsSummary,
)
from backend.gpu_utils import get_system_info
from backend.processing import ProcessingManager
from video_processor import find_videos, find_images, find_all_media, get_video_info


# Settings file path
SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"
PROMPTS_FILE = Path(__file__).parent.parent / "prompt_library.json"

# Global state
_settings: Settings = None
_prompt_library: PromptLibrary = None
_processing_manager: ProcessingManager = None
_connected_websockets: Set[WebSocket] = set()
_processing_task: asyncio.Task = None


def load_settings() -> Settings:
    """Load settings from JSON file, or return defaults if not found"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Settings(**data)
        except Exception as e:
            print(f"[API] Warning: Failed to load settings from {SETTINGS_FILE}: {e}")
    return Settings()


def save_settings(settings: Settings) -> None:
    """Save settings to JSON file"""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings.model_dump(), f, indent=2)
        print(f"[API] Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        print(f"[API] Warning: Failed to save settings to {SETTINGS_FILE}: {e}")


def load_prompt_library() -> PromptLibrary:
    """Load prompt library from JSON file, or return empty library if not found"""
    if PROMPTS_FILE.exists():
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PromptLibrary(**data)
        except Exception as e:
            print(f"[API] Warning: Failed to load prompt library from {PROMPTS_FILE}: {e}")
    return PromptLibrary()


def save_prompt_library(library: PromptLibrary) -> None:
    """Save prompt library to JSON file"""
    try:
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(library.model_dump(), f, indent=2)
        print(f"[API] Prompt library saved to {PROMPTS_FILE}")
    except Exception as e:
        print(f"[API] Warning: Failed to save prompt library to {PROMPTS_FILE}: {e}")


async def broadcast_progress(update: ProgressUpdate):
    """Broadcast progress update to all connected WebSocket clients"""
    if not _connected_websockets:
        return

    message = update.model_dump_json()
    disconnected = set()

    for ws in _connected_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)

    # Clean up disconnected clients
    _connected_websockets.difference_update(disconnected)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global _processing_manager, _settings, _prompt_library

    # Startup
    _settings = load_settings()
    _prompt_library = load_prompt_library()
    _processing_manager = ProcessingManager(progress_callback=broadcast_progress)
    print("[API] Video Caption Suite backend started")
    print(f"[API] Default working directory: {config.get_working_directory()}")
    print(f"[API] Settings loaded from {SETTINGS_FILE}" if SETTINGS_FILE.exists() else "[API] Using default settings")
    print(f"[API] Prompt library loaded with {len(_prompt_library.prompts)} prompts")

    # Detect GPUs and validate batch_size
    gpu_info = get_system_info()
    print(f"[API] Detected {gpu_info['gpu_count']} GPU(s)")
    for gpu in gpu_info['gpus']:
        print(f"[API]   - {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)")

    # Validate batch_size against available GPUs
    max_batch = gpu_info['max_batch_size']
    if _settings.batch_size > max_batch:
        print(f"[API] Adjusting batch_size from {_settings.batch_size} to {max_batch} (max available)")
        _settings.batch_size = max_batch
        save_settings(_settings)

    yield

    # Shutdown
    if _processing_manager:
        _processing_manager.stop()
    print("[API] Backend shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Video Caption Suite API",
    description="Backend API for video captioning with Qwen3-VL",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Settings Endpoints
# ============================================================================

@app.get("/api/settings", response_model=Settings)
async def get_settings():
    """Get current settings"""
    return _settings


@app.post("/api/settings", response_model=Settings)
async def update_settings(update: SettingsUpdate):
    """Update settings (partial update supported)"""
    global _settings

    update_data = update.model_dump(exclude_unset=True)
    current_data = _settings.model_dump()
    current_data.update(update_data)
    _settings = Settings(**current_data)
    save_settings(_settings)

    return _settings


@app.post("/api/settings/reset", response_model=Settings)
async def reset_settings():
    """Reset settings to defaults"""
    global _settings
    _settings = Settings()
    save_settings(_settings)
    return _settings


# ============================================================================
# System Endpoints
# ============================================================================

@app.get("/api/system/gpu", response_model=GPUInfoResponse)
async def get_gpu_info():
    """Get GPU information for frontend"""
    info = get_system_info()
    return GPUInfoResponse(**info)


# ============================================================================
# Directory Endpoints
# ============================================================================

@app.get("/api/directory", response_model=DirectoryResponse)
async def get_directory():
    """Get current working directory and media settings"""
    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()
    include_videos = config.get_include_videos()
    include_images = config.get_include_images()

    videos, images = find_all_media(working_dir, traverse, include_videos, include_images)

    return DirectoryResponse(
        directory=str(working_dir),
        video_count=len(videos),
        image_count=len(images),
        traverse_subfolders=traverse,
        include_videos=include_videos,
        include_images=include_images
    )


@app.post("/api/directory", response_model=DirectoryResponse)
async def set_directory(request: DirectoryRequest):
    """Set working directory and media settings - validates path exists"""
    # Security: reject path traversal
    if ".." in request.directory:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")

    path = Path(request.directory)

    if not path.exists():
        raise HTTPException(status_code=400, detail="Directory does not exist")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    # Set the working directory and options
    config.set_working_directory(path)
    config.set_traverse_subfolders(request.traverse_subfolders)
    config.set_include_videos(request.include_videos)
    config.set_include_images(request.include_images)
    print(f"[API] Working directory set to: {path} (traverse={request.traverse_subfolders}, videos={request.include_videos}, images={request.include_images})")

    # Count media in new directory (single-pass scan)
    videos, images = find_all_media(
        path, request.traverse_subfolders, request.include_videos, request.include_images
    )

    return DirectoryResponse(
        directory=str(path),
        video_count=len(videos),
        image_count=len(images),
        traverse_subfolders=request.traverse_subfolders,
        include_videos=request.include_videos,
        include_images=request.include_images
    )


@app.get("/api/directory/browse", response_model=DirectoryBrowseResponse)
async def browse_directory(path: str = None):
    """List subdirectories for folder picker UI"""
    # Security: reject path traversal
    if path and ".." in path:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")

    if path:
        base = Path(path)
    else:
        # Default to home directory
        base = Path.home()

    if not base.exists() or not base.is_dir():
        base = Path.home()

    directories = []
    try:
        for item in base.iterdir():
            try:
                # Skip hidden directories and check if it's a directory
                if item.is_dir() and not item.name.startswith('.'):
                    directories.append({
                        "name": item.name,
                        "path": str(item)
                    })
            except PermissionError:
                # Skip directories we can't access
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    # Sort alphabetically
    directories.sort(key=lambda x: x["name"].lower())

    # Calculate parent path
    parent = str(base.parent) if base.parent != base else None

    return DirectoryBrowseResponse(
        current=str(base),
        parent=parent,
        directories=directories
    )


# ============================================================================
# Prompt Library Endpoints
# ============================================================================

@app.get("/api/prompts", response_model=PromptLibrary)
async def list_prompts():
    """Get all saved prompts"""
    return _prompt_library


@app.post("/api/prompts", response_model=SavedPrompt)
async def create_prompt(request: CreatePromptRequest):
    """Create a new saved prompt"""
    global _prompt_library

    import uuid

    prompt = SavedPrompt(
        id=str(uuid.uuid4()),
        name=request.name,
        prompt=request.prompt,
        created_at=datetime.now().isoformat(),
    )

    _prompt_library.prompts.append(prompt)
    save_prompt_library(_prompt_library)

    return prompt


@app.get("/api/prompts/{prompt_id}", response_model=SavedPrompt)
async def get_prompt(prompt_id: str):
    """Get a specific saved prompt"""
    for prompt in _prompt_library.prompts:
        if prompt.id == prompt_id:
            return prompt
    raise HTTPException(status_code=404, detail="Prompt not found")


@app.put("/api/prompts/{prompt_id}", response_model=SavedPrompt)
async def update_prompt(prompt_id: str, request: UpdatePromptRequest):
    """Update a saved prompt"""
    global _prompt_library

    for i, prompt in enumerate(_prompt_library.prompts):
        if prompt.id == prompt_id:
            update_data = request.model_dump(exclude_unset=True)
            prompt_data = prompt.model_dump()
            prompt_data.update(update_data)
            _prompt_library.prompts[i] = SavedPrompt(**prompt_data)
            save_prompt_library(_prompt_library)
            return _prompt_library.prompts[i]

    raise HTTPException(status_code=404, detail="Prompt not found")


@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a saved prompt"""
    global _prompt_library

    for i, prompt in enumerate(_prompt_library.prompts):
        if prompt.id == prompt_id:
            deleted = _prompt_library.prompts.pop(i)
            save_prompt_library(_prompt_library)
            return {"success": True, "deleted": deleted.name}

    raise HTTPException(status_code=404, detail="Prompt not found")


# ============================================================================
# Video Endpoints
# ============================================================================

def get_media_info_fast(media_path: Path, working_dir: Path = None, media_type: MediaType = MediaType.VIDEO) -> VideoInfo:
    """Get media info without calling ffprobe - much faster for large libraries"""
    # Captions are saved in the same directory as the media file
    caption_path = media_path.parent / (media_path.stem + config.OUTPUT_EXTENSION)
    has_caption = caption_path.exists()
    caption_preview = None

    if has_caption:
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                text = f.read(200)  # Only read first 200 chars for preview
                caption_preview = text[:150] + "..." if len(text) > 150 else text
        except Exception:
            pass

    stat = media_path.stat()

    # Use relative path from working directory if provided, otherwise just the filename
    # Always use forward slashes for consistency with frontend URLs
    if working_dir:
        try:
            relative_path = media_path.relative_to(working_dir)
            name = str(relative_path).replace('\\', '/')
        except ValueError:
            name = media_path.name
    else:
        name = media_path.name

    return VideoInfo(
        name=name,
        path=str(media_path),
        size_mb=stat.st_size / (1024 * 1024),
        media_type=media_type,
        duration_sec=None,  # Skip ffprobe for speed
        width=None,
        height=None,
        frame_count=1 if media_type == MediaType.IMAGE else None,
        fps=None,
        has_caption=has_caption,
        caption_preview=caption_preview,
    )


# Backward compatibility alias
def get_video_info_fast(video_path: Path, working_dir: Path = None) -> VideoInfo:
    """Get video info without calling ffprobe - much faster for large libraries"""
    return get_media_info_fast(video_path, working_dir, MediaType.VIDEO)


@app.get("/api/videos/stream")
async def stream_videos():
    """Stream media files as Server-Sent Events for progressive loading"""
    # Shared list so we can pre-generate thumbnails after streaming
    captured_media_items = []

    async def generate():
        working_dir = config.get_working_directory()
        traverse = config.get_traverse_subfolders()
        include_videos = config.get_include_videos()
        include_images = config.get_include_images()

        # Single-pass file discovery (much faster than per-extension glob)
        videos_list, images_list = find_all_media(
            working_dir, traverse, include_videos, include_images
        )
        media_items = [(v, MediaType.VIDEO) for v in videos_list] + \
                      [(img, MediaType.IMAGE) for img in images_list]
        media_items.sort(key=lambda x: str(x[0]).lower())
        captured_media_items.extend(media_items)
        total = len(media_items)
        batch_size = 100

        # Send total count first
        yield f"data: {json.dumps({'type': 'total', 'count': total})}\n\n"

        # Send media in batches
        batch = []
        for i, (media_path, media_type) in enumerate(media_items):
            try:
                media_info = get_media_info_fast(media_path, working_dir, media_type)
                batch.append(media_info.model_dump())

                # Send batch when full or at end
                if len(batch) >= batch_size or i == total - 1:
                    yield f"data: {json.dumps({'type': 'batch', 'videos': batch, 'loaded': i + 1})}\n\n"
                    batch = []
                    await asyncio.sleep(0)  # Allow other tasks to run
            except Exception as e:
                print(f"[API] Error getting info for {media_path}: {e}")

        # Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    async def stream_and_pregenerate():
        async for chunk in generate():
            yield chunk
        # After stream completes, kick off background thumbnail pre-generation
        if captured_media_items:
            asyncio.create_task(_pregenerate_thumbnails(captured_media_items))

    return StreamingResponse(
        stream_and_pregenerate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/videos", response_model=VideoListResponse)
async def list_videos(fast: bool = True):
    """List all videos in working directory"""
    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()
    videos = find_videos(working_dir, traverse_subfolders=traverse)
    video_infos = []

    for video_path in videos:
        try:
            if fast:
                # Fast mode: skip ffprobe
                video_infos.append(get_video_info_fast(video_path, working_dir))
            else:
                # Slow mode: get full video metadata
                info = get_video_info(video_path)
                # Captions are saved in the same directory as the video
                caption_path = video_path.parent / (video_path.stem + config.OUTPUT_EXTENSION)
                has_caption = caption_path.exists()
                caption_preview = None

                if has_caption:
                    try:
                        with open(caption_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            caption_preview = text[:150] + "..." if len(text) > 150 else text
                    except Exception:
                        pass

                # Use relative path from working directory
                # Always use forward slashes for consistency with frontend URLs
                try:
                    relative_path = video_path.relative_to(working_dir)
                    name = str(relative_path).replace('\\', '/')
                except ValueError:
                    name = video_path.name

                video_infos.append(VideoInfo(
                    name=name,
                    path=str(video_path),
                    size_mb=video_path.stat().st_size / (1024 * 1024),
                    duration_sec=info.get("duration"),
                    width=info.get("width"),
                    height=info.get("height"),
                    frame_count=info.get("frame_count"),
                    fps=info.get("fps"),
                    has_caption=has_caption,
                    caption_preview=caption_preview,
                ))
        except Exception as e:
            print(f"[API] Error getting info for {video_path}: {e}")

    return VideoListResponse(videos=video_infos, total_count=len(video_infos))


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to working directory"""
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in config.VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(config.VIDEO_EXTENSIONS)}"
        )

    # Save file to working directory
    working_dir = config.get_working_directory()
    file_path = working_dir / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {"success": True, "filename": file.filename, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/videos/{video_name:path}")
async def delete_video(video_name: str):
    """Delete a video file from working directory"""
    working_dir = config.get_working_directory()
    file_path = working_dir / video_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        file_path.unlink()
        return {"success": True, "deleted": video_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Thumbnail Endpoints
# ============================================================================

# Thumbnail cache directory
THUMBNAIL_CACHE_DIR = config.PROJECT_ROOT / ".thumbnail_cache"
THUMBNAIL_CACHE_DIR.mkdir(exist_ok=True)

# Thread pool for parallel thumbnail generation
_thumbnail_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="thumb")


def get_thumbnail_cache_path(video_path: Path, size: int) -> Path:
    """Generate cache path for thumbnail based on video path and modification time"""
    stat = video_path.stat()
    cache_key = f"{video_path.name}_{stat.st_mtime}_{stat.st_size}_{size}"
    hash_name = hashlib.md5(cache_key.encode()).hexdigest()
    return THUMBNAIL_CACHE_DIR / f"{hash_name}.jpg"


def generate_thumbnail(video_path: Path, output_path: Path, size: int = 160) -> bool:
    """Generate thumbnail for a video file using ffmpeg"""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", "00:00:01",  # Seek to 1 second
            "-vframes", "1",    # Extract 1 frame
            "-vf", f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
            "-q:v", "3",        # Quality (2-5 is good)
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        return output_path.exists()
    except Exception as e:
        print(f"[API] Thumbnail generation failed for {video_path}: {e}")
        return False


def generate_image_thumbnail(image_path: Path, output_path: Path, size: int = 160) -> bool:
    """Generate thumbnail for an image file using PIL (much faster than ffmpeg)"""
    from PIL import Image as PILImage
    try:
        with PILImage.open(image_path) as img:
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            # Resize maintaining aspect ratio then center-crop to square
            img.thumbnail((size * 2, size * 2), PILImage.Resampling.LANCZOS)
            width, height = img.size
            left = (width - min(width, size)) // 2
            top = (height - min(height, size)) // 2
            right = left + min(width, size)
            bottom = top + min(height, size)
            img = img.crop((left, top, right, bottom))
            img.save(output_path, "JPEG", quality=85)
        return output_path.exists()
    except Exception as e:
        print(f"[API] Image thumbnail generation failed for {image_path}: {e}")
        return False


def _generate_any_thumbnail(media_path: Path, cache_path: Path, size: int) -> bool:
    """Route to the correct thumbnail generator based on file extension"""
    ext = media_path.suffix.lower()
    if ext in config.IMAGE_EXTENSIONS:
        return generate_image_thumbnail(media_path, cache_path, size)
    else:
        return generate_thumbnail(media_path, cache_path, size)


async def _pregenerate_thumbnails(media_items: list):
    """Background task: pre-generate thumbnails for all uncached media files."""
    default_size = 200  # Match VideoTile.vue thumbnailSize default

    uncached = []
    for media_path, media_type in media_items:
        try:
            cache_path = get_thumbnail_cache_path(media_path, default_size)
            if not cache_path.exists():
                uncached.append((media_path, cache_path, default_size))
        except Exception:
            continue

    if not uncached:
        return

    print(f"[API] Pre-generating {len(uncached)} thumbnails in background...")
    loop = asyncio.get_event_loop()
    generated = 0

    # Process in batches of 50 to avoid overwhelming I/O
    batch_size = 50
    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        futures = [
            loop.run_in_executor(
                _thumbnail_executor,
                _generate_any_thumbnail,
                media_path, cache_path, size
            )
            for media_path, cache_path, size in batch
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        generated += sum(1 for r in results if r is True)
        await asyncio.sleep(0)  # Yield to other tasks

    print(f"[API] Pre-generated {generated}/{len(uncached)} thumbnails")


@app.get("/api/videos/{video_name:path}/thumbnail")
async def get_video_thumbnail(video_name: str, size: int = 160):
    """Get thumbnail for a media file. Generates and caches if not exists."""
    # Clamp size to reasonable bounds
    size = max(64, min(320, size))

    working_dir = config.get_working_directory()
    video_path = working_dir / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Media not found")

    cache_path = get_thumbnail_cache_path(video_path, size)

    # Check cache - generate in thread to avoid blocking event loop
    if not cache_path.exists():
        success = await asyncio.to_thread(_generate_any_thumbnail, video_path, cache_path, size)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

    # Return cached thumbnail
    return FileResponse(
        cache_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            "ETag": cache_path.stem,
        }
    )


@app.delete("/api/thumbnails/cache")
async def clear_thumbnail_cache():
    """Clear the thumbnail cache"""
    try:
        count = 0
        for f in THUMBNAIL_CACHE_DIR.glob("*.jpg"):
            f.unlink()
            count += 1
        return {"success": True, "cleared": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos/{video_name:path}/stream")
async def stream_video(video_name: str):
    """Stream video file for preview playback"""
    working_dir = config.get_working_directory()
    video_path = working_dir / video_name

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Determine MIME type based on extension
    ext = video_path.suffix.lower()
    mime_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".wmv": "video/x-ms-wmv",
        ".flv": "video/x-flv",
    }
    content_type = mime_types.get(ext, "video/mp4")

    # Return the video file directly for streaming
    return FileResponse(
        path=video_path,
        media_type=content_type,
        filename=video_name,
    )


# ============================================================================
# Caption Endpoints
# ============================================================================

@app.get("/api/captions", response_model=CaptionListResponse)
async def list_captions():
    """List all generated captions in working directory"""
    captions = []
    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()

    # Use recursive glob if traverse is enabled
    glob_pattern = f"**/*{config.OUTPUT_EXTENSION}" if traverse else f"*{config.OUTPUT_EXTENSION}"

    if working_dir.exists():
        for caption_path in working_dir.glob(glob_pattern):
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    text = f.read()

                stat = caption_path.stat()

                # Use relative path from working directory if traversing
                # Always use forward slashes for consistency with frontend URLs
                try:
                    relative_path = caption_path.relative_to(working_dir)
                    video_name = str(relative_path.with_suffix('')).replace('\\', '/')
                except ValueError:
                    video_name = caption_path.stem

                captions.append(CaptionInfo(
                    video_name=video_name,
                    caption_path=str(caption_path),
                    caption_text=text,
                    created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                ))
            except Exception as e:
                print(f"[API] Error reading caption {caption_path}: {e}")

    return CaptionListResponse(captions=captions, total_count=len(captions))


@app.get("/api/captions/{video_name:path}")
async def get_caption(video_name: str):
    """Get caption for a specific video"""
    working_dir = config.get_working_directory()
    # Construct path relative to working directory, then get the caption path
    video_path = working_dir / video_name
    caption_path = video_path.parent / (video_path.stem + config.OUTPUT_EXTENSION)

    if not caption_path.exists():
        raise HTTPException(status_code=404, detail="Caption not found")

    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            text = f.read()

        return CaptionInfo(
            video_name=video_path.stem,
            caption_path=str(caption_path),
            caption_text=text,
            created_at=datetime.fromtimestamp(caption_path.stat().st_mtime).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/captions/{video_name:path}")
async def delete_caption(video_name: str):
    """Delete a caption file from working directory"""
    working_dir = config.get_working_directory()
    # Construct path relative to working directory, then get the caption path
    video_path = working_dir / video_name
    caption_path = video_path.parent / (video_path.stem + config.OUTPUT_EXTENSION)

    if not caption_path.exists():
        raise HTTPException(status_code=404, detail="Caption not found")

    try:
        caption_path.unlink()
        return {"success": True, "deleted": video_path.stem}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Endpoints
# ============================================================================

@app.get("/api/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model status"""
    if _processing_manager:
        status = _processing_manager.get_model_status()
        return ModelStatus(**status)
    return ModelStatus()


@app.post("/api/model/load")
async def load_model():
    """Pre-load the model"""
    global _processing_task

    if _processing_manager.is_processing:
        raise HTTPException(status_code=409, detail="Processing in progress")

    try:
        success = await _processing_manager.load_model(_settings)
        if success:
            return {"success": True, "message": "Model loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/unload")
async def unload_model():
    """Unload the model to free VRAM"""
    if _processing_manager.is_processing:
        raise HTTPException(status_code=409, detail="Processing in progress")

    try:
        await _processing_manager.unload_model()
        await broadcast_progress(_processing_manager.state.to_progress_update())
        return {"success": True, "message": "Model unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Processing Endpoints
# ============================================================================

@app.post("/api/process/start", response_model=ProcessingResponse)
async def start_processing(request: ProcessingRequest = None):
    """Start processing videos"""
    global _processing_task

    print(f"[API] /api/process/start called. request={request}")

    if _processing_manager.is_processing:
        print("[API] Already processing, returning 409")
        raise HTTPException(status_code=409, detail="Processing already in progress")

    # Get videos to process from working directory
    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()
    all_videos = find_videos(working_dir, traverse_subfolders=traverse)
    print(f"[API] Found {len(all_videos)} videos in {working_dir} (traverse={traverse})")

    if request and request.video_names:
        # Filter to specific videos - match on relative path when traversing subfolders
        # Normalize path separators to forward slashes for cross-platform compatibility
        video_names_lower = {n.replace('\\', '/').lower() for n in request.video_names}
        print(f"[API] Filtering to specific videos: {request.video_names}")

        def get_relative_name(v: Path) -> str:
            """Get relative path from working directory for matching"""
            try:
                # Use forward slashes for consistency with frontend
                return str(v.relative_to(working_dir)).replace('\\', '/').lower()
            except ValueError:
                return v.name.lower()

        videos = [v for v in all_videos if get_relative_name(v) in video_names_lower]
        print(f"[API] After filtering: {[get_relative_name(v) for v in videos]}")
    else:
        videos = all_videos

    if not videos:
        print("[API] No videos to process, returning 400")
        raise HTTPException(status_code=400, detail="No videos to process")

    # Start processing in background
    async def run_processing():
        print("[API] Background task started")
        try:
            await _processing_manager.process_videos(videos, _settings)
            print("[API] Background task completed successfully")
        except Exception as e:
            print(f"[API] Processing error: {e}")
            import traceback
            traceback.print_exc()
            _processing_manager.state.stage = ProcessingStage.ERROR
            _processing_manager.state.error_message = str(e)
            await broadcast_progress(_processing_manager.state.to_progress_update())

    _processing_task = asyncio.create_task(run_processing())
    print(f"[API] Created background task: {_processing_task}")

    return ProcessingResponse(
        success=True,
        message=f"Started processing {len(videos)} videos",
        videos_queued=len(videos),
    )


@app.post("/api/process/stop")
async def stop_processing():
    """Stop processing"""
    if not _processing_manager.is_processing:
        return {"success": True, "message": "Not processing"}

    _processing_manager.stop()
    return {"success": True, "message": "Stop requested"}


@app.get("/api/process/status")
async def get_processing_status():
    """Get current processing status"""
    return _processing_manager.state.to_progress_update()


# ============================================================================
# WebSocket for Real-time Progress
# ============================================================================

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()
    _connected_websockets.add(websocket)
    print(f"[API] WebSocket client connected. Total: {len(_connected_websockets)}")

    try:
        # Send current state immediately
        await websocket.send_text(
            _processing_manager.state.to_progress_update().model_dump_json()
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages (ping/pong or commands)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_text(
                        _processing_manager.state.to_progress_update().model_dump_json()
                    )
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[API] WebSocket error: {e}")
    finally:
        _connected_websockets.discard(websocket)
        print(f"[API] WebSocket client disconnected. Total: {len(_connected_websockets)}")


# ============================================================================
# Analytics Endpoints
# ============================================================================

from backend.analytics import (
    calculate_word_frequency,
    calculate_ngrams,
    calculate_word_correlations,
    get_caption_texts_from_directory,
    get_stopwords_for_preset,
    tokenize_text,
)
import time as time_module


@app.get("/api/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary():
    """Get quick analytics summary for all captions"""
    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()

    captions = get_caption_texts_from_directory(Path(working_dir), traverse)

    if not captions:
        return AnalyticsSummary(
            total_captions=0,
            total_words=0,
            unique_words=0,
            avg_words_per_caption=0.0,
            top_words=[]
        )

    # Combine all texts for analysis
    texts = [text for _, text in captions]

    # Calculate word frequency for summary
    stopwords = get_stopwords_for_preset('english')
    word_results = calculate_word_frequency(texts, stopwords=stopwords, top_n=10)

    # Count total words (including stopwords for accurate count)
    total_words = 0
    all_words = set()
    for text in texts:
        words = tokenize_text(text, lowercase=True, min_word_length=2)
        total_words += len(words)
        all_words.update(words)

    return AnalyticsSummary(
        total_captions=len(captions),
        total_words=total_words,
        unique_words=len(all_words),
        avg_words_per_caption=total_words / len(captions) if captions else 0,
        top_words=[WordFrequencyItem(
            word=r.word, count=r.count, frequency=r.frequency
        ) for r in word_results]
    )


@app.post("/api/analytics/wordfreq", response_model=WordFrequencyResponse)
async def analyze_word_frequency(request: WordFrequencyRequest):
    """Analyze word frequency across captions"""
    start_time = time_module.time()

    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()

    captions = get_caption_texts_from_directory(
        Path(working_dir), traverse, request.video_names
    )

    if not captions:
        raise HTTPException(status_code=404, detail="No captions found")

    texts = [text for _, text in captions]
    stopwords = get_stopwords_for_preset(request.stopword_preset.value, request.custom_stopwords)

    results = calculate_word_frequency(
        texts,
        stopwords=stopwords,
        min_word_length=request.min_word_length,
        top_n=request.top_n
    )

    # Calculate totals
    total_words = sum(r.count for r in results)
    unique_words = len(results)

    # Get actual total (not just top_n)
    all_word_results = calculate_word_frequency(
        texts, stopwords=stopwords, min_word_length=request.min_word_length, top_n=10000
    )
    actual_total = sum(r.count for r in all_word_results)
    actual_unique = len(all_word_results)

    return WordFrequencyResponse(
        words=[WordFrequencyItem(
            word=r.word, count=r.count, frequency=r.frequency
        ) for r in results],
        total_words=actual_total,
        total_unique_words=actual_unique,
        captions_analyzed=len(captions),
        analysis_time_ms=(time_module.time() - start_time) * 1000
    )


@app.post("/api/analytics/ngrams", response_model=NgramResponse)
async def analyze_ngrams(request: NgramRequest):
    """Analyze n-gram frequency across captions"""
    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()

    captions = get_caption_texts_from_directory(
        Path(working_dir), traverse, request.video_names
    )

    if not captions:
        raise HTTPException(status_code=404, detail="No captions found")

    texts = [text for _, text in captions]
    stopwords = get_stopwords_for_preset(request.stopword_preset.value)

    results = calculate_ngrams(
        texts,
        n=request.n,
        stopwords=stopwords,
        top_n=request.top_n,
        min_count=request.min_count
    )

    return NgramResponse(
        ngrams=[NgramItem(
            ngram=list(r.ngram),
            display=' '.join(r.ngram),
            count=r.count,
            frequency=r.frequency
        ) for r in results],
        n=request.n,
        total_ngrams=len(results),
        captions_analyzed=len(captions)
    )


@app.post("/api/analytics/correlations", response_model=CorrelationResponse)
async def analyze_correlations(request: CorrelationRequest):
    """Analyze word co-occurrence correlations using PMI"""
    working_dir = config.get_working_directory()
    traverse = config.get_traverse_subfolders()

    captions = get_caption_texts_from_directory(
        Path(working_dir), traverse, request.video_names
    )

    if not captions:
        raise HTTPException(status_code=404, detail="No captions found")

    texts = [text for _, text in captions]
    stopwords = get_stopwords_for_preset('english')

    results = calculate_word_correlations(
        texts,
        stopwords=stopwords,
        window_size=request.window_size,
        min_co_occurrence=request.min_co_occurrence,
        top_n=request.top_n
    )

    # Extract unique nodes for network visualization
    nodes = set()
    for r in results:
        nodes.add(r.word1)
        nodes.add(r.word2)

    return CorrelationResponse(
        correlations=[CorrelationItem(
            word1=r.word1,
            word2=r.word2,
            co_occurrence=r.co_occurrence_count,
            pmi_score=r.pmi_score
        ) for r in results],
        nodes=sorted(nodes),
        captions_analyzed=len(captions)
    )


# ============================================================================
# Static Files (for production - serve frontend)
# ============================================================================

# Check if frontend dist exists
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": _processing_manager.state.model_loaded if _processing_manager else False,
        "processing": _processing_manager.is_processing if _processing_manager else False,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  VIDEO CAPTION SUITE - API SERVER")
    print("=" * 60 + "\n")

    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
