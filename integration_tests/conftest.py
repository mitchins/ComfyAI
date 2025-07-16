import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
comfy_src = ROOT / "ComfyUI_repo"
sys.path.insert(0, str(comfy_src))
sys.path.insert(0, str(ROOT / "nodes"))

pytest_plugins = ["tests.conftest"]
