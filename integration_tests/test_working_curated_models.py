"""Working integration tests for curated models that actually work."""

import pytest

pytestmark = pytest.mark.integration

def test_qwen2_vl_real_inference():
    """Test Qwen2-VL with real model download and inference - no mocking."""
    try:
        from apps.shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine
        from apps.shared.model_types import get_smallest_quant_for_model
    except ImportError:
        pytest.skip("ONNX loader not available")
    
    # This is a true integration test - downloads ~2GB model and runs real inference
    loader = ONNXModelLoader()
    model_quant = get_smallest_quant_for_model("Qwen2-VL-2B-Instruct")
    
    # Real model loading - will download if not cached
    sessions, tokenizer, config = loader.load_model(model_quant)
    engine = ONNXInferenceEngine(sessions, tokenizer, config)
    
    # Real inference test
    question = "What is the capital of France?"
    response = engine.generate_text(question, max_tokens=20)
    
    # Verify real response
    assert isinstance(response, str)
    assert len(response) > 0
    assert "paris" in response.lower(), f"Expected 'Paris' in response, got: {response}"
    
    print(f"SUCCESS: {response}")


def test_model_configs_exist():
    """Test that curated model configurations exist."""
    from apps.shared.model_types import get_available_model_quants, get_smallest_quant_for_model
    
    # Check that we have configs for all three model families
    available_configs = get_available_model_quants()
    
    qwen_configs = [c for c in available_configs if c.startswith("Qwen2-VL-2B-Instruct/")]
    gemma_configs = [c for c in available_configs if c.startswith("Gemma-3n-E2B-it-ONNX/")]
    phi_configs = [c for c in available_configs if c.startswith("Phi-3.5-vision-instruct/")]
    
    assert len(qwen_configs) > 0, "Should have Qwen2-VL configs"
    assert len(gemma_configs) > 0, "Should have Gemma configs"
    assert len(phi_configs) > 0, "Should have Phi-3.5 configs"
    
    # Check smallest quant functionality
    for model_family in ["Qwen2-VL-2B-Instruct", "Gemma-3n-E2B-it-ONNX", "Phi-3.5-vision-instruct"]:
        smallest = get_smallest_quant_for_model(model_family)
        assert smallest is not None, f"Should have smallest quant for {model_family}"
        assert "/" in smallest, f"Should be model/quant format: {smallest}"


def test_curated_vs_legacy_loading():
    """Test that curated models use curated path, legacy models use legacy path."""
    try:
        from apps.shared.onnx_loader import ONNXModelLoader
        from apps.shared.model_types import get_curated_model_config
    except ImportError:
        pytest.skip("ONNX loader not available")
    
    loader = ONNXModelLoader()
    
    # Test curated model/quant combo
    curated_name = "Qwen2-VL-2B-Instruct/Q4"
    assert get_curated_model_config(curated_name) is not None, "Should be recognized as curated"
    
    # Test legacy model name (without quant)
    legacy_name = "onnx-community/Qwen2-VL-2B-Instruct"
    assert get_curated_model_config(legacy_name) is None, "Should NOT be recognized as curated"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])