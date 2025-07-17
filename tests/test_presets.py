import importlib
import os
import sys

from apps.face_api.presets import PRESETS


def load_app(preset_name):
    for key in [
        "DETECTOR_MODEL",
        "DETECTOR_FILE",
        "EMBEDDER_MODEL_PATH",
        "EMBEDDER_FILE",
        "DETECTOR_THRESHOLD",
    ]:
        os.environ.pop(key, None)
    os.environ["PRESET"] = preset_name
    sys.modules.pop("fastapi", None)
    mod = importlib.reload(importlib.import_module("apps.face_api.main"))
    return mod.app


for name in PRESETS.keys():
    def _test(name=name):
        app = load_app(name)
        preset = PRESETS[name]
        assert app.state.detector_path == f"{preset['detector_repo']}/{preset['detector_file']}"
        assert app.state.embedder_path == f"{preset['embedder_repo']}/{preset['embedder_file']}"
        assert abs(app.state.threshold - preset['threshold']) < 1e-6
    globals()[f"test_preset_{name}"] = _test
