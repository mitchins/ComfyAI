import numpy as np
import importlib, sys
sys.modules.pop("fastapi", None)
from fastapi.testclient import TestClient
from apps.face_api.main import app
import apps.face_api.face_model as face_model
import apps.face_api.main as api_main
from apps.face_api.utils import cosine_similarity

client = TestClient(app)


def test_compare_faces_success(monkeypatch):
    emb_a = np.array([1.0, 0.0])
    emb_b = np.array([0.5, 0.5])

    def fake_get_embedding(data):
        if data == b"A":
            return emb_a
        if data == b"B":
            return emb_b
        return None

    monkeypatch.setattr(face_model, "get_embedding", fake_get_embedding)
    monkeypatch.setattr(api_main, "get_embedding", fake_get_embedding)
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", b"A"), "image_b": ("b.png", b"B")},
    )
    assert resp.status_code == 200
    sim = resp.json()["similarity"]
    assert abs(sim - cosine_similarity(emb_a, emb_b)) < 1e-6


def test_face_not_detected(monkeypatch):
    monkeypatch.setattr(face_model, "get_embedding", lambda data: None)
    monkeypatch.setattr(api_main, "get_embedding", lambda data: None)
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", b"A"), "image_b": ("b.png", b"B")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "face_not_detected"
