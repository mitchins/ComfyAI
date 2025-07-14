import pytest
from fastapi.testclient import TestClient
import onnx_vllm_server


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


def test_chat_endpoint_with_images(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "images": ["img1", "img2"],
        },
    )
    assert resp.status_code == 200


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
