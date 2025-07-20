import pytest

pytestmark = pytest.mark.onnx

try:
    from fastapi.testclient import TestClient
    import fastapi  # noqa: F401
except Exception:  # pragma: no cover - optional
    TestClient = None

if TestClient is None:
    pytest.skip("fastapi not available", allow_module_level=True)
else:
    from apps.onnx_chat.main import app
    import apps.onnx_chat.main as onnx_server


def _client(monkeypatch):
    if TestClient is None:
        pytest.skip("fastapi not available")
    
    # Mock the model loader to handle "test" model
    from unittest.mock import Mock
    
    mock_engine = Mock()
    mock_engine.generate_text.return_value = "positive"
    
    async def mock_get_inference_engine(model_name):
        return mock_engine
    
    monkeypatch.setattr(onnx_server, "get_inference_engine", mock_get_inference_engine)
    # Also keep the old classify for any legacy tests
    monkeypatch.setattr(onnx_server, "classify", lambda text, model=None: "positive")
    return TestClient(app)


def test_chat_endpoint_success(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "positive"
    assert "id" in data


def test_chat_endpoint_missing_messages(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/v1/chat/completions", json={"model": "x"})
    assert resp.status_code == 400


def test_chat_endpoint_invalid_json(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        content="notjson",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_chat_endpoint_with_two_images(monkeypatch):
    client = _client(monkeypatch)
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,b"}},
        ],
    }
    resp = client.post("/v1/chat/completions", json={"model": "test", "messages": [msg]})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "positive"