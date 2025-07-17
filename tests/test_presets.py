import importlib
import os

import pytest

from apps.face_api.presets import PRESETS


def _load_app(monkeypatch, preset):
    monkeypatch.setenv("PRESET", preset)
    for var in [
        "DETECTOR_MODEL",
        "DETECTOR_FILE",
        "EMBEDDER_MODEL_PATH",
        "EMBEDDER_FILE",
        "DETECTOR_THRESHOLD",
    ]:
        monkeypatch.delenv(var, raising=False)
    import apps.face_api.main as api_main
    importlib.reload(api_main)
    return api_main.app


@pytest.mark.parametrize("preset", ["photo", "anime", "cg"])
def test_presets(monkeypatch, preset):
    app = _load_app(monkeypatch, preset)
    cfg = PRESETS[preset]
    assert app.state.detector_path == f"{cfg['detector_repo']}/{cfg['detector_file']}"
    assert app.state.embedder_path == f"{cfg['embedder_repo']}/{cfg['embedder_file']}"
    assert app.state.threshold == pytest.approx(cfg['threshold'])
