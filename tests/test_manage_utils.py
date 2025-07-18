# Tests for apps.manage_api router utilities
import os
import sys
import types
import pytest
from fastapi import HTTPException

import apps.manage_api.router as router


def test_normalize_repo_id():
    fn = router._normalize_repo_id
    assert fn("https://huggingface.co/foo/bar") == "foo/bar"
    assert fn("https://huggingface.co/foo/bar/tree/main") == "foo/bar"
    assert fn("foo/bar/") == "foo/bar"
    assert fn("foo/bar") == "foo/bar"


def test_validate_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
    router._validate_paths("repo", "file.onnx")
    with pytest.raises(router.HTTPException):
        router._validate_paths("bad repo", "file.onnx")
    with pytest.raises(router.HTTPException):
        router._validate_paths("repo", "/etc/passwd")

