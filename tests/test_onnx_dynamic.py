import os
import types
import pytest

import apps.onnx_chat.main as chat


def test_legacy_load_session_compatibility(monkeypatch, tmp_path):
    """Test that legacy load_session function exists for backward compatibility."""
    # The legacy load_session function should return None to trigger fallback logic
    sess = chat.load_session('repo/model:onx.onnx')
    assert sess is None


def test_legacy_download_model_compatibility(monkeypatch, tmp_path):
    """Test that legacy download_model function exists for backward compatibility."""
    # The legacy download_model should return a path
    result = chat.download_model('repo/model', 'model.onnx')
    assert result is not None
    assert isinstance(result, str)


def test_legacy_classify_function(monkeypatch):
    """Test that legacy classify function works for backward compatibility."""
    # Test the rule-based fallback
    assert chat.classify("This is good", "/fake/path") == "positive"
    assert chat.classify("This is bad", "/fake/path") == "negative"  
    assert chat.classify("This is okay", "/fake/path") == "neutral"

