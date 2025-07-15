import sys
import pathlib
import os
import types
import pytest

ROOT = pathlib.Path(__file__).parent.parent.resolve()
comfy_src = ROOT / "third_party" / "ComfyUI"
sys.path.insert(0, str(comfy_src))
sys.path.insert(0, str(ROOT / "custom_nodes"))

# Ensure unit test mode for modules that check this env var
os.environ.setdefault("UNIT_TEST_MODE", "1")

# Stub heavy modules
for mod in ("torch", "onnxruntime", "fastapi", "uvicorn"):
    sys.modules.setdefault(mod, types.ModuleType(mod))

from comfyai.testing.workflow_runner import run_workflow_from_json

@pytest.fixture
def run_graph():
    return run_workflow_from_json
