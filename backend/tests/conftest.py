"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def mock_config():
    """Mock config module for tests"""
    mock_config = MagicMock()
    mock_config.INPUT_DIR = Path("/tmp/test_input")
    mock_config.OUTPUT_DIR = Path("/tmp/test_output")
    mock_config.OUTPUT_EXTENSION = ".txt"
    mock_config.VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

    with patch.dict('sys.modules', {'backend.config': mock_config}):
        yield mock_config


@pytest.fixture
def sample_video_info():
    """Sample VideoInfo data for tests"""
    return {
        "name": "test_video.mp4",
        "path": "/path/to/test_video.mp4",
        "size_mb": 50.5,
        "duration_sec": 120.0,
        "width": 1920,
        "height": 1080,
        "has_caption": False,
        "caption_preview": None,
    }


@pytest.fixture
def sample_settings():
    """Sample Settings data for tests"""
    return {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "device": "cuda",
        "dtype": "bfloat16",
        "max_frames": 16,
        "frame_size": 336,
        "max_tokens": 512,
        "temperature": 0.3,
        "use_sage_attention": False,
        "use_torch_compile": True,
        "include_metadata": False,
        "prompt": "Describe this video in detail.",
    }


@pytest.fixture
def sample_progress():
    """Sample ProgressUpdate data for tests"""
    return {
        "stage": "processing",
        "current_video": "test.mp4",
        "video_index": 2,
        "total_videos": 5,
        "tokens_generated": 250,
        "tokens_per_sec": 28.5,
        "model_loaded": True,
        "vram_used_gb": 8.2,
        "substage": "generating",
        "substage_progress": 0.6,
        "error_message": None,
        "elapsed_time": 45.5,
    }
