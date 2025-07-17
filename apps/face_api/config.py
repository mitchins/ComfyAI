from __future__ import annotations
import os
import argparse
import logging
from dataclasses import dataclass

from .presets import PRESETS, PresetLoader

Preset = PresetLoader(PRESETS)


@dataclass
class Config:
    detector_model: str
    detector_file: str
    embedder_model: str
    embedder_file: str
    threshold: float
    preset: str


def load_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset")
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-file")
    parser.add_argument("--embedder-model-path")
    parser.add_argument("--embedder-file")
    parser.add_argument("--threshold", type=float)
    args, _ = parser.parse_known_args(argv)

    preset_name = os.getenv("PRESET") or args.preset or "photo"
    try:
        preset = Preset.get(preset_name)
    except KeyError as e:
        raise SystemExit(str(e))

    detector_model = os.getenv("DETECTOR_MODEL") or args.detector_model or preset["detector_repo"]
    detector_file = os.getenv("DETECTOR_FILE") or args.detector_file or preset["detector_file"]
    embedder_model = os.getenv("EMBEDDER_MODEL_PATH") or args.embedder_model_path or preset["embedder_repo"]
    embedder_file = os.getenv("EMBEDDER_FILE") or args.embedder_file or preset["embedder_file"]

    threshold_env = os.getenv("DETECTOR_THRESHOLD")
    if threshold_env is not None:
        threshold = float(threshold_env)
    elif args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = float(preset["threshold"])

    return Config(
        detector_model=detector_model,
        detector_file=detector_file,
        embedder_model=embedder_model,
        embedder_file=embedder_file,
        threshold=threshold,
        preset=preset_name,
    )


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
