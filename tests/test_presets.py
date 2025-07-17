import importlib
import os
import sys

from apps.face_api.presets import PRESETS
from fastapi.testclient import TestClient


def load_app(preset: str):
    os.environ["PRESET"] = preset
    for var in (
        "DETECTOR_MODEL",
        "DETECTOR_FILE",
        "EMBEDDER_MODEL_PATH",
        "EMBEDDER_FILE",
        "DETECTOR_THRESHOLD",
    ):
        os.environ.pop(var, None)
    sys.modules.pop("apps.face_api.main", None)
    import apps.face_api.main as main
    importlib.reload(main)
    return main.app


def check_preset(preset_name: str):
    app = load_app(preset_name)
    TestClient(app)  # trigger startup
    preset = PRESETS[preset_name]
    assert app.state.detector_path == f"{preset['detector_repo']}/{preset['detector_file']}"
    assert app.state.embedder_path == f"{preset['embedder_repo']}/{preset['embedder_file']}"
    assert app.state.threshold == preset['threshold']


def test_photo_preset():
    check_preset("photo")


def test_anime_preset():
    check_preset("anime")


def test_cg_preset():
    check_preset("cg")

