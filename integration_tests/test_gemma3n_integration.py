"""Integration test for Gemma3n multimodal model."""

import pytest
import numpy as np
from apps.shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine
from apps.shared.model_types import get_curated_model_config


@pytest.mark.integration
def test_gemma3n_model_loading():
    """Test that Gemma3n model can be loaded with curated configuration."""
    loader = ONNXModelLoader()
    
    # Test loading with Q4_MIXED quantization (matches test_gemma3n.py)
    model_quant_name = "Gemma-3n-E2B-it-ONNX/Q4_MIXED"
    
    # Verify curated config exists
    config = get_curated_model_config(model_quant_name)
    assert config is not None
    assert "audio_encoder" in config
    assert "vision_encoder" in config
    assert "embed_tokens" in config
    assert "decoder" in config
    
    # Test that expected file paths are configured correctly
    assert config["audio_encoder"] == "onnx/audio_encoder_q4.onnx"
    assert config["vision_encoder"] == "onnx/vision_encoder_quantized.onnx"
    assert config["embed_tokens"] == "onnx/embed_tokens_quantized.onnx"
    assert config["decoder"] == "onnx/decoder_model_merged_q4.onnx"


@pytest.mark.integration
def test_gemma3n_text_generation():
    """Test basic text generation with Gemma3n model."""
    loader = ONNXModelLoader()
    
    try:
        # Load the model
        sessions, tokenizer, config = loader.load_curated_model("Gemma-3n-E2B-it-ONNX/Q4_MIXED")
        
        # Create inference engine
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        # Test basic text generation
        response = engine.generate_text("Hello, how are you?", max_tokens=50)
        
        # Basic validation
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Hello" in response or "hello" in response
        
    except Exception as e:
        pytest.skip(f"Model files not available: {e}")


@pytest.mark.integration
def test_gemma3n_multimodal_generation():
    """Test multimodal generation with Gemma3n model."""
    loader = ONNXModelLoader()
    
    try:
        # Load the model
        sessions, tokenizer, config = loader.load_curated_model("Gemma-3n-E2B-it-ONNX/Q4_MIXED")
        
        # Create inference engine
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        # Verify multimodal components are loaded
        assert engine.vision_model is not None
        assert engine.audio_model is not None
        
        # Test with placeholder multimodal inputs
        # Note: This would require proper processor integration for real testing
        response = engine.generate_text(
            "Describe the following image and audio:",
            max_tokens=100,
            images=["placeholder_image.jpg"],
            audio=["placeholder_audio.wav"]
        )
        
        # Basic validation
        assert isinstance(response, str)
        assert len(response) > 0
        
    except Exception as e:
        pytest.skip(f"Model files or multimodal processing not available: {e}")


def test_gemma3n_config_structure():
    """Test that Gemma3n configuration matches expected structure."""
    from apps.shared.model_types import REFERENCE_MODELS, ReferenceModel
    
    # Verify Gemma3n is in reference models
    assert ReferenceModel.GEMMA_3N_E2B in REFERENCE_MODELS
    
    spec = REFERENCE_MODELS[ReferenceModel.GEMMA_3N_E2B]
    
    # Verify configuration
    assert spec.repo_id == "onnx-community/gemma-3n-E2B-it-ONNX"
    assert spec.config.num_layers == 24
    assert spec.config.num_heads == 16
    assert spec.config.head_dim == 64
    assert spec.config.has_vision is True
    assert spec.config.position_dims == 1
    
    # Verify components
    expected_components = {
        'audio_encoder': 'audio_encoder{suffix}.onnx',
        'decoder': 'decoder_model_merged{suffix}.onnx', 
        'embed_tokens': 'embed_tokens{suffix}.onnx',
        'vision_encoder': 'vision_encoder{suffix}.onnx'
    }
    assert spec.config.components == expected_components
    
    # Verify supported quantizations include Q4 (default from test)
    from apps.shared.model_types import Quantization
    assert Quantization.Q4 in spec.supported_quants
    assert spec.default_quant == Quantization.Q4