import types

from nodes.compare_faces import CompareFacesNode
import nodes.compare_faces as cf


def test_node_success(monkeypatch, dummy_image):
    node = CompareFacesNode()

    monkeypatch.setattr(cf, "tensor_to_pil", lambda img: dummy_image)

    resp = types.SimpleNamespace()
    resp.status_code = 200
    resp.json = lambda: {"similarity": 0.8}

    monkeypatch.setattr(cf.requests, "post", lambda *a, **k: resp)

    sim, same = node.compare(dummy_image, dummy_image, api_url="http://x", threshold=0.72)
    assert sim == 0.8
    assert same is True


def test_node_error(monkeypatch, dummy_image):
    node = CompareFacesNode()

    monkeypatch.setattr(cf, "tensor_to_pil", lambda img: dummy_image)

    resp = types.SimpleNamespace()
    resp.status_code = 422
    resp.json = lambda: {"error": "face_not_detected"}

    monkeypatch.setattr(cf.requests, "post", lambda *a, **k: resp)

    sim, same = node.compare(dummy_image, dummy_image, api_url="http://x")
    assert sim == 0.0
    assert same is False
