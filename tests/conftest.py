import pytest

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import sys, types
import os

# Ensure unit test mode for modules that check this env var
os.environ.setdefault("UNIT_TEST_MODE", "1")

# Add repository root to sys.path so local modules resolve
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Stub heavy modules for unit tests only (not integration tests)
import importlib.machinery
for mod in ("torch", "onnxruntime", "uvicorn", "onnx"):
    if mod not in sys.modules:
        module = types.ModuleType(mod)
        module.__spec__ = importlib.machinery.ModuleSpec(mod, loader=None)
        sys.modules[mod] = module

@pytest.fixture
def dummy_image():
    """16×16 black RGB image"""
    if PIL_AVAILABLE:
        return Image.fromarray(np.zeros((16,16,3), dtype=np.uint8))
    else:
        # Return a mock image object for tests
        class MockImage:
            def __init__(self):
                self.size = (16, 16)
        return MockImage()

@pytest.fixture
def dummy_string():
    return "test"

@pytest.fixture
def dummy_int():
    return 0


@pytest.fixture
def run_graph():
    """Execute a mini-graph via the workflow runner."""
    # Import the helper from the integration test support package
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

