import pytest
from fastapi.testclient import TestClient
import onnx_vllm_server
import base64


def _client(monkeypatch):
    # ensure classify returns predictable output
    monkeypatch.setattr(onnx_vllm_server, "classify", lambda text: "positive")
    return TestClient(onnx_vllm_server.app)


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


def test_chat_endpoint_two_images(monkeypatch):
    client = _client(monkeypatch)
    img1 = base64.b64encode(b"img1").decode()
    img2 = base64.b64encode(b"img2").decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2}"}},
        ],
    }]
    resp = client.post("/v1/chat/completions", json={"model": "test", "messages": messages})
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "positive"


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
