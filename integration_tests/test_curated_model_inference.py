"""Integration tests for curated model inference without mocking."""

import pytest
import asyncio

pytestmark = pytest.mark.integration

# Integration tests require real dependencies - fail fast if not available
from apps.shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine, ONNX_AVAILABLE
from apps.shared.model_types import get_smallest_quant_for_model, get_available_model_quants


class TestCuratedModelInference:
    """Integration tests for curated model inference with real models."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.loader = ONNXModelLoader()
    
    def test_qwen2_vl_2b_instruct_inference(self):
        """Test Qwen2-VL-2B-Instruct with actual inference - Capital of France test."""
        # Fail fast if ONNX not available
        assert ONNX_AVAILABLE, "ONNX runtime required for integration tests"
        
        model_quant = get_smallest_quant_for_model("Qwen2-VL-2B-Instruct")
        assert model_quant is not None, "Qwen2-VL-2B-Instruct should have at least one quantization"
        
        # Real integration test - downloads ~2GB model and runs inference
        sessions, tokenizer, config = self.loader.load_model(model_quant)
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        # Test real inference
        question = "What is the capital of France?"
        response = engine.generate_text(question, max_tokens=20)
        
        # Verify real response
        assert isinstance(response, str), f"Response should be string, got {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        assert "paris" in response.lower(), f"Expected 'Paris' in response, got: {response}"
    
    def test_gemma_3n_e2b_inference(self):
        """Test Gemma-3n-E2B-it-ONNX with actual inference - Capital of France test."""
        # Fail fast if ONNX not available
        assert ONNX_AVAILABLE, "ONNX runtime required for integration tests"
        
        # Known issue: Gemma-3n-E2B-it-ONNX has ONNX runtime compatibility issues with FP16/FP32 models
        # This test will fail until the compatibility issues are resolved
        model_quant = get_smallest_quant_for_model("Gemma-3n-E2B-it-ONNX")
        assert model_quant is not None, "Gemma-3n-E2B-it-ONNX should have at least one quantization"
        
        # Real integration test - will likely fail due to known compatibility issues
        sessions, tokenizer, config = self.loader.load_model(model_quant)
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        # Test real inference
        question = "What is the capital of France?"
        response = engine.generate_text(question, max_tokens=20)
        
        # Verify real response
        assert isinstance(response, str), f"Response should be string, got {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        assert "paris" in response.lower(), f"Expected 'Paris' in response, got: {response}"
    
    def test_phi_35_vision_inference(self):
        """Test Phi-3.5-vision-instruct with actual inference - Capital of France test."""
        # Fail fast if ONNX not available
        assert ONNX_AVAILABLE, "ONNX runtime required for integration tests"
        
        # Known issue: Phi-3.5-vision-instruct requires image_features input even for text-only inference
        # This test will fail until vision inputs are properly implemented
        model_quant = get_smallest_quant_for_model("Phi-3.5-vision-instruct")
        assert model_quant is not None, "Phi-3.5-vision-instruct should have at least one quantization"
        
        # Real integration test - will likely fail due to missing image_features input
        sessions, tokenizer, config = self.loader.load_model(model_quant)
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        # Test real inference
        question = "What is the capital of France?"
        response = engine.generate_text(question, max_tokens=20)
        
        # Verify real response
        assert isinstance(response, str), f"Response should be string, got {type(response)}"
        assert len(response) > 0, "Response should not be empty"
        assert "paris" in response.lower(), f"Expected 'Paris' in response, got: {response}"
    
    def test_all_curated_models_have_smallest_quant(self):
        """Test that all curated models have a smallest quantization defined."""
        model_families = ["Qwen2-VL-2B-Instruct", "Gemma-3n-E2B-it-ONNX", "Phi-3.5-vision-instruct"]
        
        for model_family in model_families:
            smallest_quant = get_smallest_quant_for_model(model_family)
            assert smallest_quant is not None, f"Model {model_family} should have a smallest quantization"
            assert "/" in smallest_quant, f"Smallest quant should be in format 'Model/Quant', got: {smallest_quant}"
            assert smallest_quant.startswith(model_family), f"Smallest quant should start with model name: {smallest_quant}"
    
    def test_curated_model_configs_exist(self):
        """Test that all expected curated model configurations exist."""
        available_configs = get_available_model_quants()
        
        # Check that we have configs for all three model families
        qwen_configs = [c for c in available_configs if c.startswith("Qwen2-VL-2B-Instruct/")]
        gemma_configs = [c for c in available_configs if c.startswith("Gemma-3n-E2B-it-ONNX/")]
        phi_configs = [c for c in available_configs if c.startswith("Phi-3.5-vision-instruct/")]
        
        assert len(qwen_configs) > 0, "Should have at least one Qwen2-VL-2B-Instruct configuration"
        assert len(gemma_configs) > 0, "Should have at least one Gemma-3n-E2B-it-ONNX configuration"
        assert len(phi_configs) > 0, "Should have at least one Phi-3.5-vision-instruct configuration"
        
        # Check specific expected configurations
        assert "Qwen2-VL-2B-Instruct/Q4" in available_configs, "Should have Qwen2-VL-2B-Instruct/Q4"
        assert "Gemma-3n-E2B-it-ONNX/FP16" in available_configs, "Should have Gemma-3n-E2B-it-ONNX/FP16"
        assert "Phi-3.5-vision-instruct/Q4" in available_configs, "Should have Phi-3.5-vision-instruct/Q4"


class TestCuratedModelErrorHandling:
    """Test error handling for curated models."""
    
    def test_invalid_model_quant_raises_error(self):
        """Test that invalid model/quant combinations raise appropriate errors."""
        loader = ONNXModelLoader()
        
        invalid_configs = [
            "NonExistent-Model/Q4",
            "Qwen2-VL-2B-Instruct/InvalidQuant",
            "Invalid/Invalid",
            "JustInvalid"
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError, match="not in curated list"):
                loader.load_curated_model(invalid_config)
    
    def test_model_name_without_quant_falls_back_to_legacy(self):
        """Test that model names without quant fall back to legacy loading."""
        loader = ONNXModelLoader()
        
        # This should try legacy loading (which will likely fail due to network/download issues)
        # but the important thing is that it doesn't fail with a curated config error
        try:
            loader.load_model("onnx-community/Qwen2-VL-2B-Instruct")
            # If it succeeds, that's fine too - means the legacy path worked
        except ValueError as e:
            error_msg = str(e)
            # Should get a legacy error, not a curated config error
            assert "not in curated list" not in error_msg, f"Got curated error when expecting legacy error: {error_msg}"
        except Exception as e:
            # Other exceptions are expected in integration tests (network issues, file not found, etc.)
            # The key is that we don't get a curated config error
            error_msg = str(e)
            assert "not in curated list" not in error_msg, f"Got curated error when expecting network/download error: {error_msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])