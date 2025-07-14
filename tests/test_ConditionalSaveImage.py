import os
os.environ["UNIT_TEST_MODE"] = "1"
import conditional_save_image as csi


def test_contract():
    assert isinstance(csi.ConditionalSaveImage.CATEGORY, str)
    inputs = csi.ConditionalSaveImage.INPUT_TYPES()
    assert "required" in inputs
    assert hasattr(csi.ConditionalSaveImage, csi.ConditionalSaveImage.FUNCTION)


def test_no_save_when_false(monkeypatch):
    os.environ["UNIT_TEST_MODE"] = "1"
    monkeypatch.setattr(csi.folder_paths, "get_output_directory", lambda: ".", raising=False)
    monkeypatch.setattr(csi.folder_paths, "get_save_image_path", lambda *a: (".", "file", 0, "", "pre"), raising=False)
    node = csi.ConditionalSaveImage()
    result = node.save_images(False, [None])
    assert result == {"ui": {"images": []}}
