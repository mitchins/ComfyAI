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
    from apps import onnx_server
    from apps.onnx_server import app


def _client(monkeypatch):
    if TestClient is None:
        pytest.skip("fastapi not available")
    # ensure classify returns predictable output
    monkeypatch.setattr(onnx_server, "classify", lambda text: "positive")
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
        data="notjson",
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
