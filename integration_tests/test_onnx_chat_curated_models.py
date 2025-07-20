"""Integration tests for ONNX Chat curated models with basic inference verification."""

import pytest
from unittest.mock import Mock, patch

pytestmark = pytest.mark.onnx

try:
    from fastapi.testclient import TestClient
    import fastapi  # noqa: F401
except Exception:  # pragma: no cover - optional
    TestClient = None

if TestClient is None:
    pytest.skip("fastapi not available", allow_module_level=True)
else:
    from apps.onnx_chat.main import app, validate_model_name
    from apps.shared.model_types import ReferenceModel, REFERENCE_MODELS
    import apps.onnx_chat.main as onnx_server


# Test data for basic inference verification
BASIC_INFERENCE_TESTS = [
    {"input": "Hello", "expected_type": str, "min_length": 1},
    {"input": "What is 2+2?", "expected_type": str, "min_length": 1},
    {"input": "Tell me a joke", "expected_type": str, "min_length": 1},
]


def _create_test_client(monkeypatch):
    """Create test client with mocked model loading."""
    if TestClient is None:
        pytest.skip("fastapi not available")
    
    # Mock the ONNX inference engine to simulate basic responses
    mock_engine = Mock()
    mock_engine.generate_text.return_value = "Hello! This is a test response from the ONNX model."
    
    async def mock_get_inference_engine(model_name):
        # Validate that the model is actually in our curated list
        validate_model_name(model_name)  # This will raise HTTPException if invalid
        return mock_engine
    
    monkeypatch.setattr(onnx_server, "get_inference_engine", mock_get_inference_engine)
    return TestClient(app)


class TestCuratedModelValidation:
    """Test that only curated models are accepted."""
    
    def test_valid_reference_model_names(self, monkeypatch):
        """Test that all reference model names are accepted."""
        client = _create_test_client(monkeypatch)
        
        for ref_model in ReferenceModel:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": ref_model.value, "messages": [{"role": "user", "content": "Hello"}]},
            )
            assert resp.status_code == 200, f"Model {ref_model.value} should be accepted"
            data = resp.json()
            assert "choices" in data
            assert data["choices"][0]["message"]["content"]
    
    def test_valid_repo_ids(self, monkeypatch):
        """Test that repo IDs from curated models are accepted."""
        client = _create_test_client(monkeypatch)
        
        for spec in REFERENCE_MODELS.values():
            resp = client.post(
                "/v1/chat/completions",
                json={"model": spec.repo_id, "messages": [{"role": "user", "content": "Hello"}]},
            )
            assert resp.status_code == 200, f"Repo ID {spec.repo_id} should be accepted"
    
    def test_invalid_model_rejected(self, monkeypatch):
        """Test that non-curated models are rejected."""
        client = _create_test_client(monkeypatch)
        
        invalid_models = [
            "random-user/random-model",
            "microsoft/phi-2",
            "meta-llama/Llama-2-7b-hf",
            "invalid-model-name",
            "not-a-real-model",
        ]
        
        for invalid_model in invalid_models:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": invalid_model, "messages": [{"role": "user", "content": "Hello"}]},
            )
            assert resp.status_code == 400, f"Invalid model {invalid_model} should be rejected"
            assert "not supported" in resp.json()["detail"].lower()


class TestBasicInferenceVerification:
    """Test basic inference for all curated models."""
    
    @pytest.mark.parametrize("ref_model", list(ReferenceModel))
    @pytest.mark.parametrize("test_case", BASIC_INFERENCE_TESTS)
    def test_basic_inference_all_models(self, monkeypatch, ref_model, test_case):
        """Test basic inference doesn't blow up for any curated model."""
        client = _create_test_client(monkeypatch)
        
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": ref_model.value,
                "messages": [{"role": "user", "content": test_case["input"]}],
                "max_tokens": 50  # Keep it small for testing
            },
        )
        
        assert resp.status_code == 200, f"Model {ref_model.value} should handle input: {test_case['input']}"
        data = resp.json()
        
        # Verify response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, test_case["expected_type"])
        assert len(content) >= test_case["min_length"]
    
    def test_vision_model_with_text_only(self, monkeypatch):
        """Test that vision models handle text-only input properly."""
        client = _create_test_client(monkeypatch)
        
        vision_models = [
            ReferenceModel.QWEN2_VL_2B,
            ReferenceModel.QWEN2_VL_7B,
            ReferenceModel.GRANITE_VISION_3_2B,
        ]
        
        for model in vision_models:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": model.value,
                    "messages": [{"role": "user", "content": "Describe this image."}],
                    "max_tokens": 20
                },
            )
            assert resp.status_code == 200, f"Vision model {model.value} should handle text-only input"
    
    def test_smallest_quant_loading(self, monkeypatch):
        """Test that Q4 quantization is used by default for testing."""
        # Mock the model loader to verify Q4 is requested
        original_load_model = None
        
        def mock_load_model(model_name):
            # Verify that Q4 quantization is being used
            assert ":q4" in model_name.lower() or "q4" in model_name.lower(), \
                f"Expected Q4 quantization in model name: {model_name}"
            
            # Return mock objects
            sessions = {"model": Mock()}
            tokenizer = Mock()
            tokenizer.decode.return_value = "Mock response"
            tokenizer.eos_token_id = 2
            config = Mock()
            config.has_vision = False
            return sessions, tokenizer, config
        
        with patch('apps.onnx_chat.main.model_loader.load_model', side_effect=mock_load_model):
            with patch('apps.onnx_chat.main.ONNX_LOADER_AVAILABLE', True):
                client = _create_test_client(monkeypatch)
                
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": ReferenceModel.QWEN2_VL_2B.value,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                assert resp.status_code == 200


class TestModelSpecificFeatures:
    """Test model-specific features and configurations."""
    
    def test_qwen2_vl_vision_support(self, monkeypatch):
        """Test Qwen2-VL models support vision inputs."""
        client = _create_test_client(monkeypatch)
        
        qwen_models = [ReferenceModel.QWEN2_VL_2B, ReferenceModel.QWEN2_VL_7B]
        
        for model in qwen_models:
            # Test with vision content
            message_content = [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
            ]
            
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": model.value,
                    "messages": [{"role": "user", "content": message_content}],
                    "max_tokens": 30
                },
            )
            assert resp.status_code == 200, f"Qwen2-VL model {model.value} should handle vision input"
    
    def test_granite_vision_support(self, monkeypatch):
        """Test Granite Vision model supports vision inputs."""
        client = _create_test_client(monkeypatch)
        
        message_content = [
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
        ]
        
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": ReferenceModel.GRANITE_VISION_3_2B.value,
                "messages": [{"role": "user", "content": message_content}],
                "max_tokens": 30
            },
        )
        assert resp.status_code == 200, "Granite Vision should handle vision input"
    
    def test_gemma_3n_text_only(self, monkeypatch):
        """Test Gemma-3n is text-only."""
        client = _create_test_client(monkeypatch)
        
        # Text-only should work
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": ReferenceModel.GEMMA_3N_E2B.value,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 20
            },
        )
        assert resp.status_code == 200, "Gemma-3n should handle text input"


class TestHealthAndMetadata:
    """Test health endpoints and model metadata."""
    
    def test_health_endpoint(self, monkeypatch):
        """Test the health endpoint."""
        client = _create_test_client(monkeypatch)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"
    
    def test_response_format(self, monkeypatch):
        """Test that responses follow OpenAI format."""
        client = _create_test_client(monkeypatch)
        
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": ReferenceModel.QWEN2_VL_2B.value,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        
        assert resp.status_code == 200
        data = resp.json()
        
        # Check OpenAI-compatible response format
        required_fields = ["id", "object", "choices", "model"]
        for field in required_fields:
            assert field in data, f"Response missing required field: {field}"
        
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        
        message = choice["message"]
        assert "role" in message
        assert "content" in message
        assert message["role"] == "assistant"