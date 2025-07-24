import sys
import pathlib
import pytest

ROOT = pathlib.Path(__file__).parent.parent.resolve()
comfy_src = ROOT / "ComfyUI_repo"
sys.path.insert(0, str(comfy_src))
sys.path.insert(0, str(ROOT / "nodes"))

# Add repository root to sys.path so local modules resolve
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Integration tests need REAL dependencies - remove any stubs that may have been loaded
stub_modules = ["torch", "onnxruntime", "uvicorn", "onnx"]
for mod in stub_modules:
    if mod in sys.modules and not hasattr(sys.modules[mod], '__file__'):
        # This is a stub module (no __file__ attribute), remove it
        del sys.modules[mod]

@pytest.fixture
def run_graph():
    """Execute a mini-graph via the workflow runner."""
    # Import the helper from the integration test support package
    from integration_tests.support.workflow_runner import run_workflow_from_json
    return run_workflow_from_json
