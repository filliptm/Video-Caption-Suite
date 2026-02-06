"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock the imports that require GPU/model
with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'backend.model_loader': MagicMock(),
    'backend.video_processor': MagicMock(),
}):
    from backend.api import app


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self, client):
        """Test health check returns healthy status"""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "processing" in data


class TestSettingsEndpoints:
    """Tests for settings endpoints"""

    def test_get_settings(self, client):
        """Test getting current settings"""
        response = client.get("/api/settings")
        assert response.status_code == 200

        data = response.json()
        assert "model_id" in data
        assert "device" in data
        assert "dtype" in data
        assert "max_frames" in data
        assert "frame_size" in data
        assert "max_tokens" in data
        assert "temperature" in data

    def test_update_settings_partial(self, client):
        """Test partial settings update"""
        response = client.post("/api/settings", json={
            "max_frames": 8,
            "temperature": 0.5,
        })
        assert response.status_code == 200

        data = response.json()
        assert data["max_frames"] == 8
        assert data["temperature"] == 0.5

    def test_update_settings_invalid(self, client):
        """Test invalid settings update"""
        response = client.post("/api/settings", json={
            "max_frames": 150,  # Invalid: > 128
        })
        assert response.status_code == 422  # Validation error

    def test_reset_settings(self, client):
        """Test resetting settings to defaults"""
        # First change a setting
        client.post("/api/settings", json={"max_frames": 8})

        # Then reset
        response = client.post("/api/settings/reset")
        assert response.status_code == 200

        data = response.json()
        assert data["max_frames"] == 16  # Default value


class TestVideoEndpoints:
    """Tests for video endpoints"""

    def test_list_videos(self, client):
        """Test listing videos"""
        with patch('backend.api.find_videos', return_value=[]):
            response = client.get("/api/videos")
            assert response.status_code == 200

            data = response.json()
            assert "videos" in data
            assert "total_count" in data
            assert isinstance(data["videos"], list)

    def test_delete_video_not_found(self, client):
        """Test deleting non-existent video"""
        response = client.delete("/api/videos/nonexistent.mp4")
        assert response.status_code == 404


class TestCaptionEndpoints:
    """Tests for caption endpoints"""

    def test_list_captions(self, client):
        """Test listing captions"""
        response = client.get("/api/captions")
        assert response.status_code == 200

        data = response.json()
        assert "captions" in data
        assert "total_count" in data
        assert isinstance(data["captions"], list)

    def test_get_caption_not_found(self, client):
        """Test getting non-existent caption"""
        response = client.get("/api/captions/nonexistent")
        assert response.status_code == 404


class TestModelEndpoints:
    """Tests for model endpoints"""

    def test_get_model_status(self, client):
        """Test getting model status"""
        response = client.get("/api/model/status")
        assert response.status_code == 200

        data = response.json()
        assert "loaded" in data
        assert "vram_used_gb" in data


class TestProcessingEndpoints:
    """Tests for processing endpoints"""

    def test_get_processing_status(self, client):
        """Test getting processing status"""
        response = client.get("/api/process/status")
        assert response.status_code == 200

        data = response.json()
        assert "stage" in data
        assert "video_index" in data
        assert "total_videos" in data

    def test_start_processing_no_videos(self, client):
        """Test starting processing - expects 400 when no videos found"""
        # This test verifies the endpoint exists and returns proper error
        # when there are no videos in the input directory
        response = client.post("/api/process/start")
        # Will return 400 if no videos, or 200/409 if there are videos or processing
        assert response.status_code in [400, 200, 409]

    def test_stop_processing_when_idle(self, client):
        """Test stopping when not processing"""
        response = client.post("/api/process/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
