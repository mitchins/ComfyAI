from fastapi.testclient import TestClient
import onnx_vllm_server as server


def make_client(monkeypatch):
    # prevent any session loading
    monkeypatch.setattr(server, "load_session", lambda: None)
    return TestClient(server.app)


def test_chat_completion_success(monkeypatch):
    client = make_client(monkeypatch)
    monkeypatch.setattr(server, "classify", lambda text: "positive")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"]
    assert data["choices"][0]["message"]["content"] == "positive"


def test_chat_missing_messages(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.post("/v1/chat/completions", json={"model": "dummy"})
    assert resp.status_code == 422


def test_chat_invalid_json(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.post(
        "/v1/chat/completions", data="notjson", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422
