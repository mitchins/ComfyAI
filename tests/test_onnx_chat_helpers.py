import os
import sys
import types
import numpy as np
import pytest

import apps.onnx_chat.main as chat
from apps.face_api.utils import cosine_similarity as utils_cosine_similarity


def test_ensure_model_path_download(monkeypatch, tmp_path):
    called = {}

    def fake_download(repo, filename, cache):
        called["repo"] = repo
        called["filename"] = filename
        return str(tmp_path / filename)

    monkeypatch.setattr(chat, "download_model", fake_download)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
    p = chat._ensure_model_path("hf/repo:model.onnx")
    assert p == str(tmp_path / "model.onnx")
    assert called["repo"] == "hf/repo"
    assert called["filename"] == "model.onnx"


def test_ensure_model_path_existing(tmp_path):
    file_path = tmp_path / "exist.onnx"
    file_path.write_text("x")
    assert chat._ensure_model_path(str(file_path)) == str(file_path)


def test_cosine_similarity_utils():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert utils_cosine_similarity(a, a) == pytest.approx(1.0)
    assert utils_cosine_similarity(a, b) == pytest.approx(0.0)
