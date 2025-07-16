import pytest

try:
    from fastapi.testclient import TestClient
    from apps.face_api.main import app
except Exception:  # pragma: no cover - optional heavy deps
    TestClient = None

@pytest.mark.skipif(TestClient is None, reason="fastapi not available")
def test_health_and_compare(monkeypatch):
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200

    monkeypatch.setattr("apps.face_api.main.get_embedding", lambda x: None)
    with open("Example01.png", "rb") as f:
        data = f.read()
    res = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", data), "image_b": ("b.png", data)},
    )
    assert res.status_code in (200, 422)
