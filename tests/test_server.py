import pytest
from fastapi.testclient import TestClient
import onnx_vllm_server


def _client(monkeypatch, capture=None):
    def _fake(text):
        if capture is not None:
            capture.append(text)
        return "positive"

    monkeypatch.setattr(onnx_vllm_server, "classify", _fake)
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


def test_chat_endpoint_with_images(monkeypatch):
    captured = []
    client = _client(monkeypatch, capture=captured)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "good"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,a"}},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,b"}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert captured[0] == "good"
