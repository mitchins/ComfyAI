import os
import pytest

import apps.shared.hf_utils as hf


def test_download_model(monkeypatch, tmp_path):
    src = tmp_path / "src" / "model.onnx"
    src.parent.mkdir()
    src.write_text("data")

    calls = {"count": 0}

    def fake_download(repo_id: str, filename: str):
        calls["count"] += 1
        assert repo_id == "user/repo"
        assert filename == "model.onnx"
        return str(src)

    monkeypatch.setattr(hf, "hf_hub_download", fake_download)

    cache = tmp_path / "cache"
    path1 = hf.download_model("user/repo", "model.onnx", str(cache))
    assert path1 == str(cache / "model.onnx")
    assert (cache / "model.onnx").read_text() == "data"

    # Second call should not trigger download
    path2 = hf.download_model("user/repo", "model.onnx", str(cache))
    assert path2 == path1
    assert calls["count"] == 1


