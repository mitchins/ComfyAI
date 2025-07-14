import pytest

try:
    from custom_nodes.YourPluginName.ImageSelector import ImageSelector
except Exception:  # pragma: no cover - optional node
    ImageSelector = None

if ImageSelector is None:
    pytest.skip("ImageSelector node not available", allow_module_level=True)


def test_contract():
    assert isinstance(ImageSelector.CATEGORY, str)
    inputs = ImageSelector.INPUT_TYPES()
    assert isinstance(inputs, dict)
    assert "required" in inputs
    assert isinstance(ImageSelector.RETURN_TYPES, tuple)
    assert hasattr(ImageSelector, ImageSelector.FUNCTION)


@pytest.mark.parametrize("fixture_name,arg_name", [("dummy_image", "images")])
def test_choose_image(fixture_name, arg_name, request):
    kwargs = {arg_name: request.getfixturevalue(fixture_name)}
    node = ImageSelector()
    fn = getattr(node, ImageSelector.FUNCTION)
    result = fn(**kwargs)
    from PIL.Image import Image as PILImage
    assert isinstance(result, tuple) and len(result) == 1
    assert isinstance(result[0], PILImage)
