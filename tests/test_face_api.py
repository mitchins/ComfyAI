import numpy as np
from fastapi.testclient import TestClient

from apps.face_api.main import app, face_model, utils

client = TestClient(app)


def test_compare_faces_ok(monkeypatch):
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([1.0, 0.0, 0.0])

    def fake_get_embedding(data):
        return emb_a if data == b"a" else emb_b

    monkeypatch.setattr(face_model, "get_embedding", fake_get_embedding)

    resp = client.post(
        "/v1/image/compare_faces",
        files={
            "image_a": ("a.png", b"a", "image/png"),
            "image_b": ("b.png", b"b", "image/png"),
        },
    )
    assert resp.status_code == 200
    expected = utils.cosine_similarity(emb_a, emb_b)
    assert abs(resp.json()["similarity"] - expected) < 1e-6


def test_compare_faces_no_face(monkeypatch):
    def fake_get_embedding(data):
        return None

    monkeypatch.setattr(face_model, "get_embedding", fake_get_embedding)

    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", b"a", "image/png"), "image_b": ("b.png", b"b", "image/png")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "face_not_detected"
