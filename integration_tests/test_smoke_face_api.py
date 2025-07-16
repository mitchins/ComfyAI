import io
import pytest, os

pytestmark = pytest.mark.onnx

try:
    from fastapi.testclient import TestClient
    import fastapi  # noqa: F401
except Exception:  # pragma: no cover - optional
    TestClient = None

if TestClient is None:
    pytest.skip("fastapi not available", allow_module_level=True)
else:
    os.environ.setdefault("DETECTOR_MODEL", "dummy")
    os.environ.setdefault("DETECTOR_FILE", "model.onnx")
    os.environ.setdefault("EMBEDDER_MODEL_PATH", "dummy")
    os.environ.setdefault("EMBEDDER_FILE", "embed.onnx")
    from apps.face_api.main import app
    import apps.face_api.main as api_main


def _client(monkeypatch):
    if TestClient is None:
        pytest.skip("fastapi not available")
    monkeypatch.setattr(api_main, "get_embedding", lambda data: None)
    return TestClient(app)


def test_smoke_health(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_smoke_compare_faces(dummy_image, monkeypatch):
    client = _client(monkeypatch)
    buf = io.BytesIO()
    dummy_image.save(buf, format="PNG")
    data = buf.getvalue()
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", data), "image_b": ("b.png", data)},
    )
    assert resp.status_code == 422
