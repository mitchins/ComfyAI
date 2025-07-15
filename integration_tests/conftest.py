import pytest
from PIL import Image
import numpy as np
import sys, types, os, pathlib

# Path tweaks so the workflow runner can resolve nodes
ROOT = pathlib.Path(__file__).parent.parent.resolve()
comfy_src = ROOT / "third_party" / "ComfyUI"
sys.path.insert(0, str(comfy_src))
sys.path.insert(0, str(ROOT / "custom_nodes"))

os.environ.setdefault("UNIT_TEST_MODE", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

@pytest.fixture
def run_graph():
    from comfyai.testing.workflow_runner import run_workflow_from_json
    return run_workflow_from_json


def pytest_runtest_setup(item):
    if "onnx" in item.keywords:
        import importlib.util
        if (
            importlib.util.find_spec("onnxruntime") is None
            or importlib.util.find_spec("fastapi") is None
        ):
            pytest.skip("Skipping ONNX tests; optional extra not installed")
