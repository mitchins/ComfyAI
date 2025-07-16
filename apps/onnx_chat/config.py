from __future__ import annotations
import os
import logging
from dataclasses import dataclass

@dataclass
class Config:
    model_path: str | None
    port: int
    log_level: str
    host: str = "0.0.0.0"


def load_config() -> Config:
    """Read environment variables and validate any required ones."""
    model_path = os.getenv("ONNX_MODEL_PATH")
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model path '{model_path}' does not exist")
    port = int(os.getenv("COMFYAI_ONNX_PORT", "8000"))
    log_level = os.getenv("ONNX_LOG_LEVEL", "info")
    return Config(model_path=model_path, port=port, log_level=log_level)


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
