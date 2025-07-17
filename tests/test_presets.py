import importlib
import sys
import os
from apps.face_api.presets import PRESETS


def _load_app(preset):
    for var in [
        "DETECTOR_MODEL",
        "DETECTOR_FILE",
        "EMBEDDER_MODEL_PATH",
        "EMBEDDER_FILE",
        "DETECTOR_THRESHOLD",
    ]:
        os.environ.pop(var, None)
    os.environ["PRESET"] = preset
    sys.modules.pop("fastapi", None)
    sys.modules.pop("apps.face_api.main", None)
    import apps.face_api.main as api_main
    importlib.reload(api_main)
    return api_main.app


def test_presets(monkeypatch):
    for name, preset in PRESETS.items():
        app = _load_app(name)
        assert app.state.detector_path == f"{preset['detector_repo']}/{preset['detector_file']}"
        assert app.state.embedder_path == f"{preset['embedder_repo']}/{preset['embedder_file']}"
        assert abs(app.state.threshold - preset['threshold']) < 1e-6


