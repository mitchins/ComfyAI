import numpy as np
import importlib, sys, os
sys.modules.pop("fastapi", None)
os.environ.setdefault("DETECTOR_MODEL", "dummy")
os.environ.setdefault("DETECTOR_FILE", "model.onnx")
os.environ.setdefault("EMBEDDER_MODEL_PATH", "dummy")
os.environ.setdefault("EMBEDDER_FILE", "embed.onnx")
from fastapi.testclient import TestClient
from apps.face_api.main import app
import apps.face_api.face_model as face_model
import apps.face_api.main as api_main


def _set_env(monkeypatch):
    monkeypatch.setenv("DETECTOR_MODEL", "dummy")
    monkeypatch.setenv("DETECTOR_FILE", "model.onnx")
    monkeypatch.setenv("EMBEDDER_MODEL_PATH", "dummy")
    monkeypatch.setenv("EMBEDDER_FILE", "embed.onnx")
from apps.face_api.utils import cosine_similarity

client = TestClient(app)


def test_compare_faces_success(monkeypatch):
    _set_env(monkeypatch)
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
    _set_env(monkeypatch)
    monkeypatch.setattr(face_model, "get_embedding", lambda data: None)
    monkeypatch.setattr(api_main, "get_embedding", lambda data: None)
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", b"A"), "image_b": ("b.png", b"B")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "face_not_detected"


def test_blank_image_fixture(monkeypatch, dummy_image, tmp_path):
    _set_env(monkeypatch)
    monkeypatch.setattr(face_model, "get_embedding", lambda data: None)
    monkeypatch.setattr(api_main, "get_embedding", lambda data: None)
    path = tmp_path / "blank.png"
    dummy_image.save(path)
    with open(path, "rb") as f:
        img_bytes = f.read()
    resp = client.post(
        "/v1/image/compare_faces",
        files={"image_a": ("a.png", img_bytes), "image_b": ("b.png", img_bytes)},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "face_not_detected"


def test_provider_fallback(monkeypatch):
    _set_env(monkeypatch)
    calls = {}

    class FakeFace:
        def __init__(self, name, providers):
            calls["name"] = name
            calls["providers"] = providers

        def prepare(self, ctx_id=0):
            calls["prepared"] = True

    class FakeORT:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(face_model, "FaceAnalysis", FakeFace)
    monkeypatch.setattr(face_model, "ort", FakeORT())
    monkeypatch.setattr(face_model, "_model", None, raising=False)
    monkeypatch.setenv("FACE_MODEL_NAME", "foo")
    model = face_model._load_model()
    assert calls["name"] == "foo"
    assert calls["providers"] == ["CPUExecutionProvider"]
    assert calls.get("prepared") is True
