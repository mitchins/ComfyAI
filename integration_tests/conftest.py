import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()

# 1) Path to a local clone of ComfyUI
comfy_src = ROOT / "third_party" / "ComfyUI"
sys.path.insert(0, str(comfy_src))

# 2) Add the project custom node directory
sys.path.insert(0, str(ROOT / "custom_nodes"))

pytest_plugins = ["tests.conftest"]
