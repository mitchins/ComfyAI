import os
import argparse
from .presets import PRESETS, PresetLoader

Preset = PresetLoader(PRESETS)


def _parse_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset")
    parser.add_argument("--detector_model")
    parser.add_argument("--detector_file")
    parser.add_argument("--embedder_repo")
    parser.add_argument("--embedder_file")
    parser.add_argument("--threshold", type=float)
    args, _ = parser.parse_known_args()
    return args


def load_config():
    args = _parse_cli()
    env = os.environ

    preset_name = env.get("PRESET") or args.preset or "photo"
    try:
        preset = Preset.get(preset_name)
    except KeyError as e:
        raise ValueError(str(e)) from e

    detector_model = env.get("DETECTOR_MODEL") or args.detector_model or preset["detector_repo"]
    detector_file = env.get("DETECTOR_FILE") or args.detector_file or preset["detector_file"]
    embedder_repo = env.get("EMBEDDER_MODEL_PATH") or args.embedder_repo or preset["embedder_repo"]
    embedder_file = env.get("EMBEDDER_FILE") or args.embedder_file or preset["embedder_file"]

    if env.get("DETECTOR_THRESHOLD") is not None:
        threshold = float(env["DETECTOR_THRESHOLD"])
    elif args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(preset["threshold"])

    return {
        "DETECTOR_MODEL": detector_model,
        "DETECTOR_FILE": detector_file,
        "EMBEDDER_MODEL_PATH": embedder_repo,
        "EMBEDDER_FILE": embedder_file,
        "DEFAULT_THRESHOLD": threshold,
        "PRESET_NAME": preset_name,
    }
