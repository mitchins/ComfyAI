"""Pytest config for integration tests.

Adds the ComfyUI clone and this repo's custom nodes directory to ``sys.path``
so that workflows can import nodes.
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
comfy_src = ROOT / "ComfyUI_repo"
sys.path.insert(0, str(comfy_src))
sys.path.insert(0, str(ROOT / "custom_nodes"))

pytest_plugins = ["tests.conftest"]
