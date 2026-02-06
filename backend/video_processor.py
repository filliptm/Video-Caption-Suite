"""
Video Processing Utilities
Extract and process frames from video files
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple
import time

from backend import config

# Pre-compute extension sets for fast lookup
_VIDEO_EXT_SET = {ext.lower() for ext in config.VIDEO_EXTENSIONS}
_IMAGE_EXT_SET = {ext.lower() for ext in config.IMAGE_EXTENSIONS}
_ALL_MEDIA_EXT_SET = _VIDEO_EXT_SET | _IMAGE_EXT_SET


def get_video_info(video_path: Path) -> dict:
    """
    Get information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dict with video metadata
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "path": str(video_path),
        "name": video_path.name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": None,
    }

    if info["fps"] > 0:
        info["duration"] = info["frame_count"] / info["fps"]

    cap.release()
    return info


def extract_frames(
    video_path: Path,
    max_frames: int = None,
    frame_size: int = None,
    sample_method: str = "uniform",
) -> List[Image.Image]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (default: from config)
        frame_size: Target frame size in pixels (default: from config)
        sample_method: "uniform" for even sampling, "first_last" to preserve endpoints

    Returns:
        List of PIL Images
    """
    max_frames = max_frames or config.MAX_FRAMES_PER_VIDEO
    frame_size = frame_size or config.FRAME_SIZE

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Determine which frames to extract
    if total_frames <= max_frames:
        # Use all frames
        frame_indices = list(range(total_frames))
    elif sample_method == "uniform":
        # Uniform sampling
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    elif sample_method == "first_last":
        # Keep first and last, uniform sample the rest
        if max_frames <= 2:
            frame_indices = [0, total_frames - 1][:max_frames]
        else:
            middle_count = max_frames - 2
            middle_indices = np.linspace(1, total_frames - 2, middle_count, dtype=int).tolist()
            frame_indices = [0] + middle_indices + [total_frames - 1]
    else:
        raise ValueError(f"Unknown sample method: {sample_method}")

    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        pil_frame = Image.fromarray(frame)

        # Resize if needed
        pil_frame = resize_image(pil_frame, max_size=frame_size)

        frames.append(pil_frame)

    cap.release()

    if not frames:
        raise ValueError(f"Could not extract any frames from: {video_path}")

    return frames


def resize_image(
    image: Image.Image,
    max_size: int = 448,
    min_size: int = 224,
) -> Image.Image:
    """
    Resize image while preserving aspect ratio.

    Args:
        image: PIL Image
        max_size: Maximum dimension
        min_size: Minimum dimension

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    # Calculate scale factor
    max_dim = max(width, height)
    min_dim = min(width, height)

    if max_dim > max_size:
        scale = max_size / max_dim
    elif min_dim < min_size:
        scale = min_size / min_dim
    else:
        return image  # No resize needed

    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def find_all_media(
    directory: Path = None,
    traverse_subfolders: bool = False,
    include_videos: bool = True,
    include_images: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """
    Single-pass file discovery for all media types.
    Uses os.scandir/os.walk instead of per-extension glob for much better
    performance with large directories (one traversal vs 16-28 glob calls).

    Args:
        directory: Directory to search
        traverse_subfolders: If True, recursively search subdirectories
        include_videos: Include video files in results
        include_images: Include image files in results

    Returns:
        Tuple of (video_paths, image_paths), each sorted by name
    """
    directory = directory or config.get_working_directory()
    if not directory or not directory.exists():
        return [], []

    videos = []
    images = []
    seen = set()

    if traverse_subfolders:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in _ALL_MEDIA_EXT_SET:
                    continue
                full_path = Path(root) / name
                key = str(full_path).lower()
                if key in seen:
                    continue
                seen.add(key)
                if include_videos and ext in _VIDEO_EXT_SET:
                    videos.append(full_path)
                elif include_images and ext in _IMAGE_EXT_SET:
                    images.append(full_path)
    else:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in _ALL_MEDIA_EXT_SET:
                        continue
                    key = entry.name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    full_path = Path(entry.path)
                    if include_videos and ext in _VIDEO_EXT_SET:
                        videos.append(full_path)
                    elif include_images and ext in _IMAGE_EXT_SET:
                        images.append(full_path)
        except PermissionError:
            pass

    videos.sort(key=lambda p: str(p).lower())
    images.sort(key=lambda p: str(p).lower())
    return videos, images


def find_videos(directory: Path = None, traverse_subfolders: bool = False) -> List[Path]:
    """Find all video files in a directory. Wrapper around find_all_media()."""
    videos, _ = find_all_media(directory, traverse_subfolders, include_videos=True, include_images=False)
    return videos


def find_images(directory: Path = None, traverse_subfolders: bool = False) -> List[Path]:
    """Find all image files in a directory. Wrapper around find_all_media()."""
    _, images = find_all_media(directory, traverse_subfolders, include_videos=False, include_images=True)
    return images


def process_image(
    image_path: Path,
    frame_size: int = None,
) -> Tuple[List[Image.Image], dict]:
    """
    Load and process an image file for captioning.

    Args:
        image_path: Path to image file
        frame_size: Target size for resize

    Returns:
        Tuple of ([image], metadata_dict)
    """
    frame_size = frame_size or config.FRAME_SIZE
    start_time = time.time()

    # Load image with PIL
    img = Image.open(image_path)

    # Get original dimensions before conversion
    width, height = img.size

    # Convert to RGB if needed (handles RGBA, P mode, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = resize_image(img, max_size=frame_size)

    process_time = time.time() - start_time

    metadata = {
        "path": str(image_path),
        "name": image_path.name,
        "width": width,
        "height": height,
        "frames_extracted": 1,
        "frame_size": frame_size,
        "process_time": process_time,
        "media_type": "image",
    }

    return [img], metadata


def process_video(
    video_path: Path,
    max_frames: int = None,
    frame_size: int = None,
) -> Tuple[List[Image.Image], dict]:
    """
    Process a video file: extract frames and return with metadata.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to extract
        frame_size: Target frame size

    Returns:
        Tuple of (frames_list, metadata_dict)
    """
    start_time = time.time()

    # Get video info
    info = get_video_info(video_path)

    # Extract frames
    frames = extract_frames(
        video_path,
        max_frames=max_frames,
        frame_size=frame_size,
    )

    process_time = time.time() - start_time

    metadata = {
        **info,
        "frames_extracted": len(frames),
        "frame_size": frame_size or config.FRAME_SIZE,
        "process_time": process_time,
    }

    return frames, metadata


if __name__ == "__main__":
    # Test video processing
    print("Searching for videos in input_videos/...")

    videos = find_videos()

    if not videos:
        print("No videos found. Place videos in input_videos/ directory.")
    else:
        print(f"Found {len(videos)} video(s):")
        for v in videos:
            print(f"  - {v.name}")

        # Test processing first video
        print(f"\nTesting frame extraction on: {videos[0].name}")
        frames, meta = process_video(videos[0])
        print(f"  Resolution: {meta['width']}x{meta['height']}")
        print(f"  Duration: {meta['duration']:.1f}s" if meta['duration'] else "  Duration: Unknown")
        print(f"  Total frames: {meta['frame_count']}")
        print(f"  Extracted: {meta['frames_extracted']} frames")
        print(f"  Process time: {meta['process_time']:.2f}s")
