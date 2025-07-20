from __future__ import annotations
import os
import sys
import argparse
import logging
from dataclasses import dataclass

from .presets import PRESETS


class PresetLoader:
    def __init__(self, presets: dict[str, dict]):
        self.presets = presets

    def get(self, name: str) -> dict:
        if name not in self.presets:
            valid = ", ".join(self.presets)
            raise KeyError(f"Unknown preset '{name}'. Valid options: {valid}")
        return self.presets[name]


Preset = PresetLoader(PRESETS)


@dataclass
class Config:
    detector_model: str
    detector_file: str
    embedder_model_path: str
    embedder_file: str
    threshold: float
    preload_models: bool
    preset_name: str | None = None
    log_level: str = "info"


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


_ENV_VARS = [
    "DETECTOR_MODEL",
    "DETECTOR_FILE",
    "EMBEDDER_MODEL_PATH",
    "EMBEDDER_FILE",
    "DETECTOR_THRESHOLD",
]


def _cli_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-file")
    parser.add_argument("--embedder-model-path")
    parser.add_argument("--embedder-file")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--preset")
    args, _ = parser.parse_known_args(argv)
    return args


def load_config(argv: list[str] | None = None) -> Config:
    args = _cli_args(argv if argv is not None else sys.argv[1:])

    preset_name = os.getenv("PRESET") or args.preset or "photo"  # Default to photo preset
    preset = {}
    if preset_name:
        try:
            preset = Preset.get(preset_name)
        except KeyError as e:
            raise SystemExit(str(e))

    def pick(env_name: str, arg_val: str | None, key: str, default: str = "") -> str:
        return os.getenv(env_name) or arg_val or preset.get(key, default)

    detector_model = pick("DETECTOR_MODEL", args.detector_model, "detector_repo", "deepghs/real_face_detection")
    detector_file = pick("DETECTOR_FILE", args.detector_file, "detector_file", "face_detect_v1.4_s/model.onnx")
    embedder_model_path = pick("EMBEDDER_MODEL_PATH", args.embedder_model_path, "embedder_repo", "openailab/onnx-arcface-resnet100-ms1m")
    embedder_file = pick("EMBEDDER_FILE", args.embedder_file, "embedder_file", "model.onnx")

    thr_env = os.getenv("DETECTOR_THRESHOLD")
    if thr_env is not None:
        threshold = float(thr_env)
    elif args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = float(preset.get("threshold", 0.5))

    preload = os.getenv("PRELOAD_MODELS", "false").lower() in ("1", "true", "yes")

    log_level = os.getenv("FACE_API_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).lower()

    return Config(
        detector_model=detector_model,
        detector_file=detector_file,
        embedder_model_path=embedder_model_path,
        embedder_file=embedder_file,
        threshold=threshold,
        preload_models=preload,
        preset_name=preset_name,
        log_level=log_level,
    )

