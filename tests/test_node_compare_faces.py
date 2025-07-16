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
from nodes.compare_faces import CompareFacesNode, fetch_face_api_response
import apps.face_api.main as api_main

client = TestClient(app)


def _setup_fake_embeddings(monkeypatch):
    emb_a = np.array([1.0, 0.0])
    emb_b = np.array([0.5, 0.5])

    calls = []
    def fake_get_embedding(data):
        calls.append(data)
        return emb_a if len(calls) == 1 else emb_b

    monkeypatch.setattr(face_model, "get_embedding", fake_get_embedding)
    monkeypatch.setattr(api_main, "get_embedding", fake_get_embedding)
    return emb_a, emb_b


def test_node_compare_faces(dummy_image, monkeypatch):
    emb_a, emb_b = _setup_fake_embeddings(monkeypatch)

    def fake_send(url, img_a, img_b, timeout=5.0, retries=3, backoff=0.5):
        with open(img_a, "rb") as f1, open(img_b, "rb") as f2:
            resp = client.post("/v1/image/compare_faces", files={"image_a": f1, "image_b": f2})
        return resp.json()

    monkeypatch.setattr("nodes.compare_faces.fetch_face_api_response", fake_send)
    monkeypatch.setattr("nodes.compare_faces.tensor_to_pil", lambda x: x)

    node = CompareFacesNode()
    sim, same = node.run(dummy_image, dummy_image, api_url="http://test", threshold=0.7)
    from apps.face_api.utils import cosine_similarity
    assert abs(sim - cosine_similarity(emb_a, emb_b)) < 1e-6
    assert same is True


def test_node_error(dummy_image, monkeypatch):
    monkeypatch.setattr(face_model, "get_embedding", lambda d: None)
    monkeypatch.setattr(api_main, "get_embedding", lambda d: None)

    def fake_send(url, img_a, img_b, timeout=5.0, retries=3, backoff=0.5):
        with open(img_a, "rb") as f1, open(img_b, "rb") as f2:
            resp = client.post("/v1/image/compare_faces", files={"image_a": f1, "image_b": f2})
        return resp.json()

    monkeypatch.setattr("nodes.compare_faces.fetch_face_api_response", fake_send)
    monkeypatch.setattr("nodes.compare_faces.tensor_to_pil", lambda x: x)

    node = CompareFacesNode()
    sim, same = node.run(dummy_image, dummy_image, api_url="http://test")
    assert sim == 0.0 and same is False

