import argparse
import os
from .presets import PRESETS, PresetLoader

Preset = PresetLoader(PRESETS)


def load_config(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset")
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-file")
    parser.add_argument("--embedder-model-path")
    parser.add_argument("--embedder-file")
    parser.add_argument("--threshold", type=float)
    args, _ = parser.parse_known_args(argv)

    env = os.environ

    preset_name = env.get("PRESET")
    if preset_name is None:
        preset_name = args.preset or "photo"

    cli_overrides = {
        "detector_repo": args.detector_model,
        "detector_file": args.detector_file,
        "embedder_repo": args.embedder_model_path,
        "embedder_file": args.embedder_file,
        "threshold": args.threshold,
    }

    preset = Preset.get(preset_name)

    def choose(key, env_name):
        if env.get(env_name) is not None:
            return env[env_name]
        if cli_overrides.get(key) is not None:
            return cli_overrides[key]
        return preset[key]

    cfg = {
        "PRESET_NAME": preset_name,
        "DETECTOR_MODEL": choose("detector_repo", "DETECTOR_MODEL"),
        "DETECTOR_FILE": choose("detector_file", "DETECTOR_FILE"),
        "EMBEDDER_MODEL_PATH": choose("embedder_repo", "EMBEDDER_MODEL_PATH"),
        "EMBEDDER_FILE": choose("embedder_file", "EMBEDDER_FILE"),
    }

    thr_env = env.get("DETECTOR_THRESHOLD")
    if thr_env is not None:
        cfg["DEFAULT_THRESHOLD"] = float(thr_env)
    elif cli_overrides.get("threshold") is not None:
        cfg["DEFAULT_THRESHOLD"] = cli_overrides["threshold"]
    else:
        cfg["DEFAULT_THRESHOLD"] = preset["threshold"]

    return cfg
