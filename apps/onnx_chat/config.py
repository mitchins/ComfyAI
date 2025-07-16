from dataclasses import dataclass
import os
import logging

@dataclass
class Config:
    model_path: str | None
    port: int
    log_level: str


def load_config() -> Config:
    """Load configuration from environment variables."""
    port = int(os.getenv("COMFYAI_ONNX_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", os.getenv("ONNX_LOG_LEVEL", "info"))
    model_path = os.getenv("ONNX_MODEL_PATH")
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model '{model_path}' not found")
    return Config(model_path=model_path, port=port, log_level=log_level)
