"""Integration tests for ONNX model loading and inference."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from apps.shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine
from apps.shared.model_types import get_onnx_model_config, ONNX_MODEL_CONFIGS


class TestONNXModelLoader:
    """Test ONNX model loader functionality."""
    
    def setup_method(self):
        self.loader = ONNXModelLoader()
    
    def test_parse_model_name_with_colon(self):
        """Test parsing model name with colon syntax."""
        repo_id, filename = self.loader.parse_model_name("onnx-community/Qwen2-VL-2B-Instruct:decoder_model_merged_q4.onnx")
        assert repo_id == "onnx-community/Qwen2-VL-2B-Instruct"
        assert filename == "decoder_model_merged_q4.onnx"
    
    def test_parse_model_name_repo_only(self):
        """Test parsing model name with repo only."""
        repo_id, filename = self.loader.parse_model_name("onnx-community/Qwen2-VL-2B-Instruct")
        assert repo_id == "onnx-community/Qwen2-VL-2B-Instruct"
        assert filename == "model.onnx"
    
    def test_extract_quantization_suffix(self):
        """Test extracting quantization suffix from filename."""
        suffix = self.loader.extract_quantization_suffix("decoder_model_merged_q4.onnx")
        assert suffix == "_q4"
        
        suffix = self.loader.extract_quantization_suffix("decoder_model_merged.onnx")
        assert suffix == ""
        
        suffix = self.loader.extract_quantization_suffix("model.onnx")
        assert suffix == ""
    
    @patch('apps.shared.onnx_loader.hf_hub_download')
    @patch('apps.shared.onnx_loader.ort.InferenceSession')
    @patch('apps.shared.onnx_loader.AutoTokenizer')
    def test_load_model_success(self, mock_tokenizer, mock_session, mock_download):
        """Test successful model loading."""
        # Setup mocks
        mock_download.return_value = "/fake/path/model.onnx"
        mock_session.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        # Test loading
        model_name = "onnx-community/Qwen2-VL-2B-Instruct"
        sessions, tokenizer, config = self.loader.load_model(model_name)
        
        # Verify results
        assert 'embed' in sessions
        assert 'decoder' in sessions
        assert tokenizer is not None
        assert config == ONNX_MODEL_CONFIGS['qwen2-vl-2b']
        
        # Verify caching
        assert model_name in self.loader.sessions_cache
        assert model_name in self.loader.tokenizers_cache
        assert model_name in self.loader.configs_cache
    
    def test_load_unsupported_model(self):
        """Test loading unsupported model raises error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            self.loader.load_model("unknown/model")


class TestONNXInferenceEngine:
    """Test ONNX inference engine functionality."""
    
    def setup_method(self):
        # Create mock sessions
        self.mock_embed = Mock()
        self.mock_decoder = Mock()
        self.mock_tokenizer = Mock()
        
        # Setup mock decoder inputs
        mock_input = Mock()
        mock_input.name = 'inputs_embeds'
        self.mock_decoder.get_inputs.return_value = [
            mock_input,
            Mock(name='attention_mask'),
            Mock(name='position_ids'),
            Mock(name='past_key_values.0.key'),
            Mock(name='past_key_values.0.value')
        ]
        
        sessions = {
            'embed': self.mock_embed,
            'decoder': self.mock_decoder
        }
        
        config = ONNX_MODEL_CONFIGS['qwen2-vl-2b']
        
        self.engine = ONNXInferenceEngine(sessions, self.mock_tokenizer, config)
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.embed_model == self.mock_embed
        assert self.engine.decoder_model == self.mock_decoder
        assert self.engine.tokenizer == self.mock_tokenizer
        assert self.engine.config.num_layers == 28
        assert self.engine.config.num_heads == 2
        assert self.engine.config.head_dim == 128
    
    def test_generate_text_basic(self):
        """Test basic text generation."""
        # Setup mocks
        self.mock_tokenizer.return_value = {
            'input_ids': np.array([[1, 2, 3]]),
            'attention_mask': np.array([[1, 1, 1]])
        }
        self.mock_tokenizer.eos_token_id = 2
        
        # Mock embed model output
        self.mock_embed.run.return_value = [np.random.rand(1, 3, 768)]
        
        # Mock decoder outputs - simulate generating one token then EOS
        logits_first = np.zeros((1, 3, 32000))
        logits_first[0, -1, 4] = 10.0  # High score for token 4
        
        logits_second = np.zeros((1, 1, 32000))
        logits_second[0, -1, 2] = 10.0  # High score for EOS token
        
        # Create a function that returns appropriate outputs for each call
        def mock_decoder_run(*args, **kwargs):
            if not hasattr(mock_decoder_run, 'call_count'):
                mock_decoder_run.call_count = 0
            mock_decoder_run.call_count += 1
            
            if mock_decoder_run.call_count == 1:
                # First call - generate token 4
                return [logits_first] + [np.random.rand(1, 2, 3, 128)] * 56
            else:
                # Second call - generate EOS and stop
                return [logits_second] + [np.random.rand(1, 2, 1, 128)] * 56
        
        self.mock_decoder.run.side_effect = mock_decoder_run
        
        self.mock_tokenizer.decode.return_value = "Hello world"
        
        # Test generation
        result = self.engine.generate_text("Hello", max_tokens=10)
        
        # Verify
        assert result == "Hello world"
        assert self.mock_tokenizer.called
        assert self.mock_embed.run.called
        assert self.mock_decoder.run.called
    
    def test_generate_text_with_images_unsupported(self):
        """Test that providing images to non-vision model raises error."""
        # Create non-vision config
        non_vision_config = ONNX_MODEL_CONFIGS['granite-3.0-2b']  # This is non-vision
        
        # Create a mock single model
        mock_single_model = Mock()
        mock_input = Mock()
        mock_input.name = 'input_ids'
        mock_single_model.get_inputs.return_value = [mock_input]
        
        engine = ONNXInferenceEngine(
            {'model': mock_single_model},  # Single model architecture
            self.mock_tokenizer,
            non_vision_config
        )
        
        with pytest.raises(ValueError, match="Images provided but model doesn't support vision"):
            engine.generate_text("Hello", images=["base64image"])


class TestModelConfigs:
    """Test model configuration functionality."""
    
    def test_get_qwen2_vl_config(self):
        """Test getting Qwen2-VL configuration."""
        config = get_onnx_model_config("onnx-community/Qwen2-VL-2B-Instruct")
        assert config == ONNX_MODEL_CONFIGS['qwen2-vl-2b']
        assert config.num_layers == 28
        assert config.has_vision == True
        assert config.position_dims == 3
    
    def test_get_qwen2_vl_7b_config(self):
        """Test getting Qwen2-VL-7B configuration."""
        config = get_onnx_model_config("onnx-community/Qwen2-VL-7B-Instruct")
        assert config == ONNX_MODEL_CONFIGS['qwen2-vl-7b']
        assert config.num_layers == 32
        assert config.num_heads == 4
    
    def test_get_unknown_config(self):
        """Test getting configuration for unknown model."""
        config = get_onnx_model_config("unknown/model")
        assert config is None
    
    def test_all_configs_have_required_fields(self):
        """Test that all model configs have required fields."""
        for name, config in ONNX_MODEL_CONFIGS.items():
            assert hasattr(config, 'num_layers')
            assert hasattr(config, 'num_heads') 
            assert hasattr(config, 'head_dim')
            assert hasattr(config, 'has_vision')
            assert hasattr(config, 'position_dims')
            assert hasattr(config, 'components')
            
            # Check components - different architectures have different components
            if 'granite' in name:
                # Single model architecture
                assert 'model' in config.components
            else:
                # Multi-component architecture
                assert 'embed' in config.components
                assert 'decoder' in config.components
                
            if config.has_vision:
                assert 'vision' in config.components


@pytest.mark.integration
class TestONNXChatIntegration:
    """Integration tests for ONNX chat functionality."""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_mock(self):
        """Test chat endpoint with mocked ONNX inference."""
        from apps.onnx_chat.main import app
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            # Mock the inference engine
            with patch('apps.onnx_chat.main.get_inference_engine') as mock_get_engine:
                mock_engine = Mock()
                mock_engine.generate_text.return_value = "Hello! How can I help you?"
                mock_get_engine.return_value = mock_engine
                
                response = client.post("/v1/chat/completions", json={
                    "model": "onnx-community/Qwen2-VL-2B-Instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 50
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"
                assert data["model"] == "onnx-community/Qwen2-VL-2B-Instruct"
    
    @pytest.mark.asyncio 
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from apps.onnx_chat.main import app
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])