"""Pytest configuration for integration tests.

Prepends the local ComfyUI checkout and the ``custom_nodes`` directory to
``sys.path`` so node imports resolve correctly."""

import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
comfy_src = ROOT / "ComfyUI_repo"
sys.path.insert(0, str(comfy_src))
sys.path.insert(0, str(ROOT / "custom_nodes"))

pytest_plugins = ["tests.conftest"]
