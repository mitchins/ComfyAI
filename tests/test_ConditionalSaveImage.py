import os
import sys
import types
import pytest

# Import after setting UNIT_TEST_MODE so the module stubs heavy deps
os.environ["UNIT_TEST_MODE"] = "1"
import conditional_save_image


class DummyImage:
    def __init__(self):
        self._arr = __import__('numpy').zeros((1, 16, 16, 3), dtype='float32')

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape


def test_contract():
    cls = conditional_save_image.ConditionalSaveImage
    assert isinstance(cls.CATEGORY, str)
    inputs = cls.INPUT_TYPES()
    assert "condition" in inputs["required"]
    assert hasattr(cls, cls.FUNCTION)


def test_skip_save(monkeypatch):
    dummy = DummyImage()
    # Stub folder_paths.get_output_directory and get_save_image_path
    fp = types.SimpleNamespace(
        get_output_directory=lambda: "/tmp",
        get_save_image_path=lambda *a, **k: ("/tmp", "img", 0, "", "img")
    )
    monkeypatch.setattr(conditional_save_image, "folder_paths", fp)
    node = conditional_save_image.ConditionalSaveImage()
    result = node.save_images(False, [dummy])
    assert result == {"ui": {"images": []}}
