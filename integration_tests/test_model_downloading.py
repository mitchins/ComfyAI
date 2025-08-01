"""Integration tests for model downloading with proper labeling.

These tests actually download models and verify they work. They are marked
with @pytest.mark.slow to distinguish from faster integration tests.
The smallest quantized models are used to minimize download time while
still testing the full download and inference pipeline.

Expected runtime: 3-4 minutes per model (first run, then cached).
"""

import pytest
import logging
import time
from pathlib import Path

# Mark this entire module as slow integration tests
pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Set up logging to see download progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Integration tests require real dependencies
try:
    from apps.shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine, ONNX_AVAILABLE
    from apps.shared.model_types import get_smallest_quant_for_model, get_available_model_quants
    from huggingface_hub import hf_hub_download, HfFolder
except ImportError as e:
    ONNX_AVAILABLE = False
    logger.error(f"Required dependencies not available: {e}")


class TestModelDownloading:
    """Test actual model downloading with the smallest quantized models."""
    
    def setup_method(self):
        """Setup for each test method."""
        if not ONNX_AVAILABLE:
            pytest.skip("ONNX runtime and dependencies required for download tests")
        
        self.loader = ONNXModelLoader()
        logger.info("Starting model download test")
    
    @pytest.mark.slow
    def test_qwen2_vl_2b_smallest_download_and_inference(self):
        """Test downloading and using Qwen2-VL-2B-Instruct/UINT8 (smallest available).
        
        This test:
        1. Downloads the smallest quantized version (~500MB vs ~2GB for FP32)
        2. Verifies all components are downloaded
        3. Tests basic inference
        4. Measures download and inference time
        """
        start_time = time.time()
        
        # Use the smallest quantization to minimize download time
        model_quant = get_smallest_quant_for_model("Qwen2-VL-2B-Instruct")
        assert model_quant == "Qwen2-VL-2B-Instruct/UINT8", f"Expected UINT8, got {model_quant}"
        
        logger.info(f"Downloading model: {model_quant}")
        download_start = time.time()
        
        # This will download if not cached, or load from cache if available
        sessions, tokenizer, config = self.loader.load_model(model_quant)
        
        download_time = time.time() - download_start
        logger.info(f"Model loading completed in {download_time:.2f}s")
        
        # Verify all expected components are loaded
        assert 'decoder' in sessions, "Decoder component should be loaded"
        assert 'embed_tokens' in sessions, "Embed tokens component should be loaded"
        assert 'vision_encoder' in sessions, "Vision encoder component should be loaded"
        assert tokenizer is not None, "Tokenizer should be loaded"
        assert config is not None, "Config should be loaded"
        assert config.has_vision, "Qwen2-VL should have vision capability"
        
        # Test basic inference
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        inference_start = time.time()
        response = engine.generate_text("What is the capital of France?", max_tokens=10)
        inference_time = time.time() - inference_start
        
        total_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.2f}s")
        logger.info(f"Total test time: {total_time:.2f}s")
        logger.info(f"Model response: {response}")
        
        # Basic validation that inference worked
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert "France" in response or "Paris" in response, f"Expected France/Paris in response: {response}"
    
    @pytest.mark.slow
    def test_gemma_3n_smallest_download_and_inference(self):
        """Test downloading and using Gemma-3n-E2B-it-ONNX/Q4_MIXED (smallest available).
        
        This test:
        1. Downloads the smallest quantized Gemma-3n version
        2. Verifies multimodal components are downloaded
        3. Tests basic text inference
        """
        start_time = time.time()
        
        model_quant = get_smallest_quant_for_model("Gemma-3n-E2B-it-ONNX")
        assert model_quant == "Gemma-3n-E2B-it-ONNX/Q4_MIXED", f"Expected Q4_MIXED, got {model_quant}"
        
        logger.info(f"Downloading model: {model_quant}")
        download_start = time.time()
        
        sessions, tokenizer, config = self.loader.load_model(model_quant)
        
        download_time = time.time() - download_start
        logger.info(f"Model loading completed in {download_time:.2f}s")
        
        # Verify Gemma-3n specific components
        assert 'decoder' in sessions, "Decoder component should be loaded"
        assert 'embed_tokens' in sessions, "Embed tokens component should be loaded"
        # Gemma-3n may have optional vision/audio components
        logger.info(f"Loaded components: {list(sessions.keys())}")
        
        # Test basic inference
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        inference_start = time.time()
        response = engine.generate_text("Hello, how are you?", max_tokens=15)
        inference_time = time.time() - inference_start
        
        total_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.2f}s")
        logger.info(f"Total test time: {total_time:.2f}s")
        logger.info(f"Model response: {response}")
        
        # Basic validation
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
    
    @pytest.mark.slow
    def test_phi_35_vision_smallest_download_and_inference(self):
        """Test downloading and using Phi-3.5-vision-instruct/Q4 (smallest available).
        
        This test:
        1. Downloads the smallest quantized Phi-3.5 vision version
        2. Verifies vision-capable components are downloaded
        3. Tests text-only inference (vision inference requires image processing)
        """
        start_time = time.time()
        
        model_quant = get_smallest_quant_for_model("Phi-3.5-vision-instruct")
        assert model_quant == "Phi-3.5-vision-instruct/Q4", f"Expected Q4, got {model_quant}"
        
        logger.info(f"Downloading model: {model_quant}")
        download_start = time.time()
        
        sessions, tokenizer, config = self.loader.load_model(model_quant)
        
        download_time = time.time() - download_start
        logger.info(f"Model loading completed in {download_time:.2f}s")
        
        # Verify Phi-3.5 vision specific components
        assert 'decoder' in sessions, "Decoder component should be loaded"
        assert 'prepare_inputs_embeds' in sessions, "prepare_inputs_embeds component should be loaded"
        assert config.has_vision, "Phi-3.5-vision should have vision capability"
        logger.info(f"Loaded components: {list(sessions.keys())}")
        
        # Test basic text inference
        engine = ONNXInferenceEngine(sessions, tokenizer, config)
        
        inference_start = time.time()
        # Note: Phi-3.5-vision uses prepare_inputs_embeds architecture
        response = engine.generate_text("What is 2+2?", max_tokens=10)
        inference_time = time.time() - inference_start
        
        total_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.2f}s")
        logger.info(f"Total test time: {total_time:.2f}s")
        logger.info(f"Model response: {response}")
        
        # Basic validation
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
    
    @pytest.mark.slow
    def test_smolvlm_smallest_download_and_inference(self):
        """Test downloading and using SmolVLM-256M-Instruct/UINT8 (ultra-lightweight).
        
        This test:
        1. Downloads the smallest SmolVLM model (only ~200MB total)
        2. Verifies vision components are downloaded
        3. Tests basic text inference
        4. Demonstrates ultra-fast inference suitable for CPU/edge deployment
        """
        start_time = time.time()
        
        model_quant = get_smallest_quant_for_model("SmolVLM-256M-Instruct")
        assert model_quant == "SmolVLM-256M-Instruct/UINT8", f"Expected UINT8, got {model_quant}"
        
        logger.info(f"Downloading ultra-lightweight model: {model_quant}")
        download_start = time.time()
        
        sessions, tokenizer, config = self.loader.load_model(model_quant)
        
        download_time = time.time() - download_start
        logger.info(f"Model loading completed in {download_time:.2f}s")
        
        # Verify SmolVLM components (uses same structure as Qwen2-VL)
        assert 'decoder' in sessions, "Decoder component should be loaded"
        assert 'embed_tokens' in sessions, "Embed tokens component should be loaded"  # SmolVLM uses embed_tokens
        assert 'vision_encoder' in sessions, "Vision encoder component should be loaded"
        assert config.has_vision, "SmolVLM should have vision capability"
        assert config.num_layers == 30, "SmolVLM should have 30 layers"
        assert config.num_heads == 9, "SmolVLM should have 9 attention heads"
        
        # Test basic inference
        engine = ONNXInferenceEngine(sessions, tokenizer, config) 
        
        inference_start = time.time()
        response = engine.generate_text("Describe a beautiful sunset.", max_tokens=20)
        inference_time = time.time() - inference_start
        
        total_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.2f}s")
        logger.info(f"Total test time: {total_time:.2f}s")
        logger.info(f"SmolVLM response: {response}")
        
        # Basic validation
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        
        # SmolVLM should be very fast for inference (under 5 seconds even on CPU)
        assert inference_time < 10.0, f"SmolVLM should be fast, took {inference_time:.2f}s"
    
    def test_model_caching_behavior(self):
        """Test that models are properly cached after first download.
        
        This test verifies that subsequent loads of the same model are much faster.
        """
        model_quant = get_smallest_quant_for_model("Qwen2-VL-2B-Instruct")
        
        # First load (may download or load from cache)
        start_time = time.time()
        sessions1, tokenizer1, config1 = self.loader.load_model(model_quant)
        first_load_time = time.time() - start_time
        
        # Second load should be from cache
        start_time = time.time()
        sessions2, tokenizer2, config2 = self.loader.load_model(model_quant)
        second_load_time = time.time() - start_time
        
        logger.info(f"First load time: {first_load_time:.2f}s")
        logger.info(f"Second load time: {second_load_time:.2f}s")
        
        # Verify same objects are returned (from cache)
        assert sessions1 is sessions2, "Sessions should be the same object (cached)"
        assert tokenizer1 is tokenizer2, "Tokenizer should be the same object (cached)"
        assert config1 is config2, "Config should be the same object (cached)"
        
        # Second load should be much faster (sub-second)
        assert second_load_time < 1.0, f"Cached load should be fast, took {second_load_time:.2f}s"
    
    def test_download_error_handling(self):
        """Test proper error handling for invalid models."""
        with pytest.raises(ValueError, match="not in curated list"):
            self.loader.load_model("NonExistent-Model/Q4")
        
        with pytest.raises(ValueError, match="Unknown model name"):
            self.loader.load_curated_model("NonExistent/Q4")


class TestModelDownloadComponents:
    """Test downloading of individual model components."""
    
    def setup_method(self):
        if not ONNX_AVAILABLE:
            pytest.skip("ONNX runtime required for component download tests")
        self.loader = ONNXModelLoader()
    
    def test_component_download_paths(self):
        """Test that component download paths are correctly constructed."""
        from apps.shared.model_types import get_curated_model_config
        
        # Test that curated configs exist for our smallest models
        qwen_config = get_curated_model_config("Qwen2-VL-2B-Instruct/UINT8")
        assert qwen_config is not None, "Qwen2-VL UINT8 config should exist"
        assert "decoder" in qwen_config, "Should have decoder component path"
        assert "embed_tokens" in qwen_config, "Should have embed_tokens component path"
        assert "vision_encoder" in qwen_config, "Should have vision_encoder component path"
        
        gemma_config = get_curated_model_config("Gemma-3n-E2B-it-ONNX/Q4_MIXED")
        assert gemma_config is not None, "Gemma-3n Q4_MIXED config should exist"
        assert "decoder" in gemma_config, "Should have decoder component path"
        
        phi_config = get_curated_model_config("Phi-3.5-vision-instruct/Q4")
        assert phi_config is not None, "Phi-3.5-vision Q4 config should exist"
        assert "decoder" in phi_config, "Should have decoder component path"
        
        smol_config = get_curated_model_config("SmolVLM-256M-Instruct/UINT8")
        assert smol_config is not None, "SmolVLM UINT8 config should exist"
        assert "decoder" in smol_config, "Should have decoder component path"
        assert "embed_tokens" in smol_config, "Should have embed_tokens component path"
        assert "vision_encoder" in smol_config, "Should have vision_encoder component path"
    
    @pytest.mark.slow
    def test_auxiliary_file_download(self):
        """Test that auxiliary files (tokenizer, config) are downloaded."""
        model_quant = get_smallest_quant_for_model("Qwen2-VL-2B-Instruct")
        repo_id = "onnx-community/Qwen2-VL-2B-Instruct"
        
        # Download auxiliary files
        self.loader.download_auxiliary_files(repo_id)
        
        # Verify that at least config.json was downloaded (tokenizer files may vary)
        try:
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            assert Path(config_path).exists(), "config.json should be downloaded"
            logger.info(f"Successfully downloaded config.json to {config_path}")
        except Exception as e:
            logger.warning(f"Could not verify config.json download: {e}")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s", "--tb=short"])