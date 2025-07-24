"""Tests for the unified model management API endpoints."""
import os
from unittest.mock import Mock, patch
import pytest
from starlette.testclient import TestClient
from fastapi import FastAPI

# Ensure required env vars so face_api can import
os.environ.setdefault("DETECTOR_MODEL", "fake")
os.environ.setdefault("DETECTOR_FILE", "model.onnx")
os.environ.setdefault("EMBEDDER_MODEL_PATH", "fake")
os.environ.setdefault("EMBEDDER_FILE", "model.onnx")

from apps.manage_api.router import router as manage_router
from apps.shared.unified_model_registry import UnifiedModel, ModelFile

app = FastAPI()
app.include_router(manage_router, prefix="/manage")
client = TestClient(app)


class TestUnifiedManageAPI:
    """Test the unified model management API endpoints."""

    @patch('apps.manage_api.router.registry')
    def test_get_unified_models(self, mock_registry):
        """Test getting all unified models."""
        # Mock registry response
        mock_model = UnifiedModel(
            id="test-model",
            name="Test Model",
            server="chat",
            architecture="multi-component",
            description="Test model",
            files=[
                ModelFile(path="model.onnx", repo_id="test/repo", size_mb=100.0, downloaded=True)
            ],
            quantizations=["FP16", "Q4"],
            status="1/2-quants",
            total_size_mb=100.0
        )
        mock_registry.get_all_models.return_value = [mock_model]
        mock_registry.update_download_status.return_value = None

        response = client.get("/manage/unified-models")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        
        model_data = data[0]
        assert model_data["id"] == "test-model"
        assert model_data["name"] == "Test Model"
        assert model_data["server"] == "chat"
        assert model_data["architecture"] == "multi-component"
        assert model_data["status"] == "1/2-quants"
        assert len(model_data["files"]) == 1
        assert len(model_data["quantizations"]) == 2

    @patch('apps.manage_api.router.registry')
    def test_get_unified_model_by_id(self, mock_registry):
        """Test getting a specific unified model by ID."""
        mock_model = UnifiedModel(
            id="test-model",
            name="Test Model",
            server="face",
            architecture="single-file",
            description="Test face model",
            files=[
                ModelFile(path="model.onnx", repo_id="test/repo", size_mb=50.0, downloaded=False)
            ],
            quantizations=[],
            status="not-downloaded",
            total_size_mb=0.0
        )
        mock_registry.get_model.return_value = mock_model
        mock_registry.update_download_status.return_value = None

        response = client.get("/manage/unified-models/test-model")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-model"
        assert data["server"] == "face"
        assert data["architecture"] == "single-file"

    @patch('apps.manage_api.router.registry')
    def test_get_unified_model_not_found(self, mock_registry):
        """Test getting a non-existent model."""
        mock_registry.get_model.return_value = None
        mock_registry.update_download_status.return_value = None

        response = client.get("/manage/unified-models/non-existent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch('apps.manage_api.router.download_file')
    @patch('apps.manage_api.router.registry')
    def test_download_unified_model(self, mock_registry, mock_download):
        """Test downloading a unified model."""
        mock_model = UnifiedModel(
            id="test-model",
            name="Test Model",
            server="face",
            architecture="single-file",
            description="Test",
            files=[
                ModelFile(path="model.onnx", repo_id="test/repo")
            ]
        )
        mock_registry.get_model.return_value = mock_model

        response = client.post("/manage/unified-models/download", 
                             json={"model_id": "test-model"})
        
        assert response.status_code == 202
        assert "Download initiated" in response.json()["message"]
        mock_download.assert_called_once_with(repo_id="test/repo", file_path="model.onnx")

    @patch('apps.manage_api.router.download_file')
    @patch('apps.shared.model_types.get_curated_model_config')
    @patch('apps.manage_api.router.registry')
    def test_download_unified_model_with_quantization(self, mock_registry, mock_config, mock_download):
        """Test downloading a specific quantization of a model."""
        # Mock the model
        mock_model = UnifiedModel(
            id="chat-gemma-3n-e2b-it-onnx",
            name="Gemma 3n",
            server="chat",
            architecture="multi-component",
            description="Test",
            files=[
                ModelFile(path="onnx/model_fp16.onnx", repo_id="test/repo"),
                ModelFile(path="onnx/model_q4.onnx", repo_id="test/repo"),
            ],
            quantizations=["FP16", "Q4"]
        )
        mock_registry.get_model.return_value = mock_model
        
        # Mock the quantization config
        mock_config.return_value = {
            "decoder": "onnx/model_fp16.onnx"
        }

        response = client.post("/manage/unified-models/download", 
                             json={"model_id": "chat-gemma-3n-e2b-it-onnx", "quantization": "FP16"})
        
        assert response.status_code == 202
        # Should only download the FP16 file
        mock_download.assert_called_once()

    @patch('apps.manage_api.router.registry')
    def test_download_unified_model_not_found(self, mock_registry):
        """Test downloading a non-existent model."""
        mock_registry.get_model.return_value = None

        response = client.post("/manage/unified-models/download", 
                             json={"model_id": "non-existent"})
        
        assert response.status_code == 404

    @patch('apps.manage_api.router.delete_cached_file')
    @patch('apps.manage_api.router.registry')
    def test_delete_unified_model(self, mock_registry, mock_delete):
        """Test deleting a unified model."""
        mock_model = UnifiedModel(
            id="test-model",
            name="Test Model",
            server="face",
            architecture="single-file",
            description="Test",
            files=[
                ModelFile(path="model.onnx", repo_id="test/repo")
            ]
        )
        mock_registry.get_model.return_value = mock_model

        response = client.delete("/manage/unified-models/test-model")
        
        assert response.status_code == 200
        assert "Deleted 1 files" in response.json()["message"]
        mock_delete.assert_called_once_with(repo_id="test/repo", file_path="model.onnx")

    @patch('apps.manage_api.router.delete_cached_file')
    @patch('apps.shared.model_types.get_curated_model_config')
    @patch('apps.manage_api.router.registry')
    def test_delete_unified_model_with_quantization(self, mock_registry, mock_config, mock_delete):
        """Test deleting a specific quantization of a model."""
        mock_model = UnifiedModel(
            id="chat-test",
            name="Test",
            server="chat",
            architecture="multi-component",
            description="Test",
            files=[
                ModelFile(path="onnx/model_fp16.onnx", repo_id="test/repo"),
                ModelFile(path="onnx/model_q4.onnx", repo_id="test/repo"),
            ],
            quantizations=["FP16", "Q4"]
        )
        mock_registry.get_model.return_value = mock_model
        
        # Mock the quantization config
        mock_config.return_value = {
            "decoder": "onnx/model_fp16.onnx"
        }

        response = client.delete("/manage/unified-models/chat-test?quantization=FP16")
        
        assert response.status_code == 200
        # Should only delete the FP16 file
        mock_delete.assert_called_once()

    @patch('apps.manage_api.router.registry')
    def test_delete_unified_model_not_found(self, mock_registry):
        """Test deleting a non-existent model."""
        mock_registry.get_model.return_value = None

        response = client.delete("/manage/unified-models/non-existent")
        
        assert response.status_code == 404

    @patch('apps.manage_api.router.download_file')
    @patch('apps.manage_api.router.registry')
    def test_download_error_handling(self, mock_registry, mock_download):
        """Test download error handling."""
        mock_model = UnifiedModel(
            id="test-model",
            name="Test Model",
            server="face",
            architecture="single-file",
            description="Test",
            files=[
                ModelFile(path="model.onnx", repo_id="test/repo")
            ]
        )
        mock_registry.get_model.return_value = mock_model
        mock_download.side_effect = Exception("Download failed")

        response = client.post("/manage/unified-models/download", 
                             json={"model_id": "test-model"})
        
        assert response.status_code == 500
        assert "Download failed" in response.json()["detail"]

    @patch('apps.manage_api.router.delete_cached_file')
    @patch('apps.manage_api.router.registry')
    def test_delete_error_handling(self, mock_registry, mock_delete):
        """Test delete error handling."""
        mock_model = UnifiedModel(
            id="test-model",
            name="Test Model",
            server="face",
            architecture="single-file", 
            description="Test",
            files=[
                ModelFile(path="model.onnx", repo_id="test/repo")
            ]
        )
        mock_registry.get_model.return_value = mock_model
        mock_delete.side_effect = Exception("Delete failed")

        response = client.delete("/manage/unified-models/test-model")
        
        assert response.status_code == 500
        assert "Delete failed" in response.json()["detail"]