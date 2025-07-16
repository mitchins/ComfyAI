from PIL import Image
import pytest

from nodes.compare_faces import CompareFacesNode


def test_node_compare_faces_success(monkeypatch):
    node = CompareFacesNode()

    # Stub tensor_to_pil to avoid torch dependency
    from nodes import compare_faces as cf
    monkeypatch.setattr(cf, "tensor_to_pil", lambda img: Image.new("RGB", (1,1)))

    class DummyResp:
        status_code = 200
        def json(self):
            return {"similarity": 0.8}
    monkeypatch.setattr(cf.requests, "post", lambda url, files=None, timeout=30: DummyResp())

    sim, same = node.compare("a", "b", api_url="http://x", threshold=0.72)
    assert sim == 0.8
    assert same is True


def test_node_compare_faces_failure(monkeypatch):
    node = CompareFacesNode()
    from nodes import compare_faces as cf
    monkeypatch.setattr(cf, "tensor_to_pil", lambda img: Image.new("RGB", (1,1)))
    class DummyResp:
        status_code = 422
        def json(self):
            return {"error": "face_not_detected"}
    monkeypatch.setattr(cf.requests, "post", lambda *a, **k: DummyResp())

    sim, same = node.compare("a", "b", api_url="http://x", threshold=0.72)
    assert sim == 0.0
    assert same is False
