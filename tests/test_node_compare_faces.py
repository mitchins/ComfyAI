import types
import numpy as np
import requests

from nodes.compare_faces import CompareFacesNode


class DummyImage:
    def __init__(self):
        self._arr = np.zeros((1, 3, 8, 8), dtype="float32")

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape


def test_node_success(monkeypatch):
    node = CompareFacesNode()

    def fake_post(url, files=None, timeout=None):
        resp = types.SimpleNamespace(status_code=200, json=lambda: {"similarity": 0.8})
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    img = DummyImage()
    result = node.run(img, img, api_url="http://x", threshold=0.72)
    assert result == (0.8, True)


def test_node_error(monkeypatch):
    node = CompareFacesNode()

    def fake_post(url, files=None, timeout=None):
        resp = types.SimpleNamespace(status_code=422, json=lambda: {"error": "face_not_detected"})
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    img = DummyImage()
    result = node.run(img, img, api_url="http://x")
    assert result == (0.0, False)
