import pytest
from PIL import Image
import numpy as np
import sys, types
import os

# Ensure unit test mode for modules that check this env var
os.environ.setdefault("UNIT_TEST_MODE", "1")

# Add repository root to sys.path so local modules resolve
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Stub heavy modules so imports never fail
import importlib.machinery
for mod in ("torch", "onnxruntime", "fastapi", "uvicorn"):
    if mod not in sys.modules:
        module = types.ModuleType(mod)
        module.__spec__ = importlib.machinery.ModuleSpec(mod, loader=None)
        sys.modules[mod] = module

@pytest.fixture
def dummy_image():
    """16×16 black RGB image"""
    return Image.fromarray(np.zeros((16,16,3), dtype=np.uint8))

@pytest.fixture
def dummy_string():
    return "test"

@pytest.fixture
def dummy_int():
    return 0


@pytest.fixture
def run_graph():
    """Execute a mini-graph via the workflow runner."""
    from integration_tests.support.workflow_runner import run_workflow_from_json
    return run_workflow_from_json


# Pytest hook: skip ONNX tests without optional packages
def pytest_runtest_setup(item):
    """Skip tests marked with 'onnx' if optional deps aren't installed."""
    if "onnx" in item.keywords:
        import importlib.util
        if (
            importlib.util.find_spec("onnxruntime") is None
            or importlib.util.find_spec("fastapi") is None
        ):
            pytest.skip("Skipping ONNX tests; optional extra not installed")

