import pytest

pytestmark = pytest.mark.onnx

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional
    TestClient = None

if TestClient is None:
    pytest.skip("fastapi not available", allow_module_level=True)
else:
    from apps.face_api.main import app
    import apps.face_api.face_model as face_model
    from apps.face_api.utils import cosine_similarity
    import numpy as np


def _client(monkeypatch):
    if TestClient is None:
        pytest.skip("fastapi not available")
    return TestClient(app)


def test_compare_faces_success(monkeypatch):
    client = _client(monkeypatch)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    monkeypatch.setattr(face_model, "get_embedding", lambda b: v1 if b == b"a" else v2)
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", b"a", "image/png"), "image_b": ("b.png", b"b", "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["similarity"] == pytest.approx(1.0)


def test_compare_faces_no_face(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(face_model, "get_embedding", lambda b: None)
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", b"a", "image/png"), "image_b": ("b.png", b"b", "image/png")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "face_not_detected"
