import sys
sys.modules.pop("fastapi", None)
sys.modules.pop("uvicorn", None)
sys.modules.pop("onnxruntime", None)
import io
import numpy as np
import pytest

pytestmark = pytest.mark.onnx

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional
    TestClient = None

from apps.face_api import main, face_model


@pytest.fixture
def dummy_image_bytes():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buf, format="PNG")
    return buf.getvalue()


def _client(monkeypatch, embeds):
    if TestClient is None:
        pytest.skip("fastapi not available")
    seq = iter(embeds)

    def fake_get(_):
        return next(seq)

    monkeypatch.setattr(face_model, "get_embedding", fake_get)
    return TestClient(main.app)


def test_compare_faces_success(monkeypatch, dummy_image_bytes):
    emb = np.array([1.0, 0.0])
    client = _client(monkeypatch, [emb, emb])
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", dummy_image_bytes, "image/png"),
               "image_b": ("b.png", dummy_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["similarity"] == pytest.approx(1.0)


def test_compare_faces_low(monkeypatch, dummy_image_bytes):
    client = _client(monkeypatch, [np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", dummy_image_bytes, "image/png"),
               "image_b": ("b.png", dummy_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["similarity"] == pytest.approx(0.0)


def test_face_not_detected(monkeypatch, dummy_image_bytes):
    client = _client(monkeypatch, [None, np.array([1.0, 0.0])])
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", dummy_image_bytes, "image/png"),
               "image_b": ("b.png", dummy_image_bytes, "image/png")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "face_not_detected"
