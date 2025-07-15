import os
import sys
import types
import pytest
from PIL import Image
import numpy as np

os.environ.setdefault("UNIT_TEST_MODE", "1")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

apps_path = os.path.join(repo_root, "apps")
if apps_path not in sys.path:
    sys.path.insert(0, apps_path)

for mod in ("torch", "onnxruntime", "fastapi", "uvicorn"):
    sys.modules.setdefault(mod, types.ModuleType(mod))

@pytest.fixture
def dummy_image():
    return Image.fromarray(np.zeros((16,16,3), dtype=np.uint8))

@pytest.fixture
def dummy_string():
    return "test"

@pytest.fixture
def dummy_int():
    return 0


def pytest_runtest_setup(item):
    if "onnx" in item.keywords:
        import importlib.util
        if (
            importlib.util.find_spec("onnxruntime") is None
            or importlib.util.find_spec("fastapi") is None
        ):
            pytest.skip("Skipping ONNX tests; optional extra not installed")
