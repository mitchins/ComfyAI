from fastapi.testclient import TestClient
import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import onnx_vllm_server as srv


@pytest.fixture(autouse=True)
def patch_classifier(monkeypatch):
    # avoid loading real ONNX sessions
    monkeypatch.setattr(srv, "load_session", lambda: None)
    monkeypatch.setattr(srv, "classify", lambda text: "positive")


def get_client():
    return TestClient(srv.app)


def test_chat_completion_ok():
    client = get_client()
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data and data["choices"][0]["message"]["content"] == "positive"


def test_chat_completion_missing_messages():
    client = get_client()
    resp = client.post("/v1/chat/completions", json={"model": "dummy"})
    assert resp.status_code == 422


def test_chat_completion_invalid_json():
    client = get_client()
    resp = client.post(
        "/v1/chat/completions",
        data="{bad",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400 or resp.status_code == 422
