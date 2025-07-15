import sys
import pathlib
import types
import os
import pytest
from PIL import Image
import numpy as np

# Ensure unit test mode
os.environ.setdefault("UNIT_TEST_MODE", "1")

ROOT = pathlib.Path(__file__).parent.parent.resolve()

# 1) Add ComfyUI core source (submodule or CI-clone)
comfy_src_candidates = [
    ROOT / "third_party" / "ComfyUI",
    ROOT / "comfyui-core",
    ROOT / "ComfyUI_repo",
]
for path in comfy_src_candidates:
    if path.exists():
        sys.path.insert(0, str(path))
        break

# 2) Add plugin dir
sys.path.insert(0, str(ROOT / "custom_nodes"))
# Also ensure repo root import
sys.path.insert(0, str(ROOT))

# Stub heavy optional modules
for mod in ("torch", "onnxruntime", "fastapi", "uvicorn"):
    sys.modules.setdefault(mod, types.ModuleType(mod))


@pytest.fixture
def dummy_image():
    return Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))


@pytest.fixture
def run_graph():
    from comfyai.testing.workflow_runner import run_workflow_from_json
    return run_workflow_from_json
