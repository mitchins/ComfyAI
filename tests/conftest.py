import pytest
from PIL import Image
import numpy as np
import sys, types
import os

# Ensure unit test mode for modules that check this env var
os.environ.setdefault("UNIT_TEST_MODE", "1")
os.environ.setdefault("DETECTOR_MODEL", "stub-model")
os.environ.setdefault("DETECTOR_FILE", "stub.onnx")
os.environ.setdefault("EMBEDDER_MODEL_PATH", "stub-embedder")
os.environ.setdefault("EMBEDDER_FILE", "embed.onnx")

# Add repository root to sys.path so local modules resolve
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Stub heavy modules so imports never fail
import importlib.machinery
for mod in ("torch", "onnxruntime", "fastapi", "uvicorn", "imgutils"):
    if mod not in sys.modules:
        module = types.ModuleType(mod)
        module.__spec__ = importlib.machinery.ModuleSpec(mod, loader=None)
        sys.modules[mod] = module

if "onnxruntime" in sys.modules:
    ort = sys.modules["onnxruntime"]
    class _Session:
        def __init__(self, *a, **kw):
            pass
        def run(self, *a, **kw):
            return [[0]]
        def get_inputs(self):
            class _I: name="x"; pass
            return [_I()]
    ort.InferenceSession = _Session
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]

if "imgutils" in sys.modules:
    detect_mod = types.ModuleType("imgutils.detect")
    face_mod = types.ModuleType("imgutils.detect.face")
    def _noop(*args, **kwargs):
        return []
    face_mod.detect_faces = _noop
    detect_mod.face = face_mod
    sys.modules["imgutils.detect"] = detect_mod
    sys.modules["imgutils.detect.face"] = face_mod

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
    from comfyai.testing.workflow_runner import run_workflow_from_json
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

