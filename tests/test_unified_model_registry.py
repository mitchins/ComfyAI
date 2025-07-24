"""Tests for the unified model registry."""
import pytest
from unittest.mock import Mock, patch
from apps.shared.unified_model_registry import UnifiedModelRegistry, UnifiedModel, ModelFile


class TestUnifiedModelRegistry:
    """Test the unified model registry functionality."""

    def test_initialization(self):
        """Test that registry initializes with expected models."""
        registry = UnifiedModelRegistry()
        models = registry.get_all_models()
        
        # Should have chat and face models
        assert len(models) > 0
        
        chat_models = registry.get_models_by_server("chat")
        face_models = registry.get_models_by_server("face")
        
        assert len(chat_models) > 0
        assert len(face_models) > 0
        assert len(chat_models) + len(face_models) == len(models)

    def test_chat_model_discovery(self):
        """Test that chat models are discovered correctly."""
        registry = UnifiedModelRegistry()
        chat_models = registry.get_models_by_server("chat")
        
        # Check for expected chat models
        model_names = [m.name for m in chat_models]
        assert any("Qwen2" in name for name in model_names)
        assert any("Gemma" in name for name in model_names)
        
        # All chat models should be multi-component
        for model in chat_models:
            assert model.architecture == "multi-component"
            assert len(model.quantizations) > 0
            assert len(model.files) > 0

    def test_face_model_discovery(self):
        """Test that face models are discovered correctly."""
        registry = UnifiedModelRegistry()
        face_models = registry.get_models_by_server("face")
        
        # Check for expected face models
        model_names = [m.name for m in face_models]
        assert any("Face Detection" in name for name in model_names)
        assert any("ArcFace" in name for name in model_names)
        assert any("CLIP" in name for name in model_names)
        
        # All face models should be single-file
        for model in face_models:
            assert model.architecture == "single-file"
            assert len(model.files) == 1

    def test_get_model_by_id(self):
        """Test getting a specific model by ID."""
        registry = UnifiedModelRegistry()
        models = registry.get_all_models()
        
        # Get first model and test retrieval
        first_model = models[0]
        retrieved = registry.get_model(first_model.id)
        
        assert retrieved is not None
        assert retrieved.id == first_model.id
        assert retrieved.name == first_model.name
        
        # Test non-existent model
        assert registry.get_model("non-existent") is None

    def test_companion_file_discovery(self):
        """Test companion file discovery patterns."""
        registry = UnifiedModelRegistry()
        
        # Test the different patterns
        test_cases = [
            ("model.onnx", ["model.onnx_data"], ["model.onnx_data"]),
            ("model_fp16.onnx", ["model_fp16.onnx_data"], ["model_fp16.onnx_data"]),
            ("model.onnx", [], []),  # No companions exist
        ]
        
        for main_file, existing_files, expected in test_cases:
            with patch('huggingface_hub.list_repo_files', return_value=existing_files):
                companions = registry._discover_actual_companion_files("test/repo", main_file)
                assert companions == expected

    def test_quantization_file_mapping(self):
        """Test that quantization files are mapped correctly."""
        registry = UnifiedModelRegistry()
        
        # Create a test model
        test_model = UnifiedModel(
            id="test",
            name="Test Model",
            server="chat",
            architecture="multi-component",
            description="Test",
            files=[
                ModelFile(path="model_fp16.onnx", repo_id="test/repo"),
                ModelFile(path="model_q4.onnx", repo_id="test/repo"),
                ModelFile(path="model.onnx", repo_id="test/repo"),
            ],
            quantizations=["FP16", "Q4", "FP32"]
        )
        
        # Test quantization mapping
        fp16_files = registry._get_files_for_quantization(test_model, "FP16")
        q4_files = registry._get_files_for_quantization(test_model, "Q4")
        fp32_files = registry._get_files_for_quantization(test_model, "FP32")
        
        assert len(fp16_files) == 1
        assert "fp16" in fp16_files[0].path
        
        assert len(q4_files) == 1
        assert "q4" in q4_files[0].path
        
        assert len(fp32_files) == 1
        assert fp32_files[0].path == "model.onnx"

    @patch('huggingface_hub.scan_cache_dir')
    def test_download_status_update(self, mock_scan):
        """Test download status detection."""
        # Mock cache scan result
        mock_repo = Mock()
        mock_repo.repo_id = "test/repo"
        mock_revision = Mock()
        mock_file = Mock()
        mock_file.file_path = "model.onnx"
        mock_file.size_on_disk = 1024 * 1024  # 1MB
        mock_revision.files = [mock_file]
        mock_repo.revisions = [mock_revision]
        
        mock_cache_info = Mock()
        mock_cache_info.repos = [mock_repo]
        mock_scan.return_value = mock_cache_info
        
        registry = UnifiedModelRegistry()
        registry.update_download_status()
        
        # Check that status was updated
        models = registry.get_all_models()
        assert all(model.status in ["not-downloaded", "downloaded", "partial-files"] or 
                  model.status.endswith("-quants") for model in models)

    def test_quantization_status_counting(self):
        """Test that quantization counting works correctly."""
        registry = UnifiedModelRegistry()
        
        # Create test model with mixed download status
        test_model = UnifiedModel(
            id="test",
            name="Test Model", 
            server="chat",
            architecture="multi-component",
            description="Test",
            files=[
                ModelFile(path="model_fp16.onnx", repo_id="test/repo", downloaded=True),
                ModelFile(path="model_q4.onnx", repo_id="test/repo", downloaded=True),
                ModelFile(path="model.onnx", repo_id="test/repo", downloaded=False),
            ],
            quantizations=["FP16", "Q4", "FP32"]
        )
        
        # Manually set the model for testing
        registry.models["test"] = test_model
        
        # Test quantization counting logic
        downloaded_quants = 0
        for quant in test_model.quantizations:
            quant_files = registry._get_files_for_quantization(test_model, quant)
            if len(quant_files) == 0:
                continue
            quant_downloaded = sum(1 for f in quant_files if f.downloaded)
            if quant_downloaded == len(quant_files) and len(quant_files) > 0:
                downloaded_quants += 1
        
        assert downloaded_quants == 2  # FP16 and Q4 should be "downloaded"