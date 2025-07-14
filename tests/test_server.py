import json
from fastapi.testclient import TestClient
import onnx_vllm_server


def make_client(monkeypatch):
    monkeypatch.setattr(onnx_vllm_server, "classify", lambda text: "mocked")
    return TestClient(onnx_vllm_server.app)


def test_chat_completions_ok(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "hello"}]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "mocked"


def test_missing_messages(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.post("/v1/chat/completions", json={"model": "test"})
    assert resp.status_code == 400


def test_invalid_json(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.post("/v1/chat/completions", data="{bad", headers={"content-type": "application/json"})

    assert resp.status_code == 400

