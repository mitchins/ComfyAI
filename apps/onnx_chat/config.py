from dataclasses import dataclass
import os
import logging

@dataclass
class Config:
    detector_model: str
    embedder_model: str
    threshold: float
    onnx_model_path: str | None
    port: int
    log_level: str


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logging.critical(f"Missing required environment variable: {name}")
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def load_config() -> Config:
    detector_model = _required("DETECTOR_MODEL")
    embedder_model = _required("EMBEDDER_MODEL")
    threshold = float(os.getenv("THRESHOLD", "0.5"))
    onnx_model_path = os.getenv("ONNX_MODEL_PATH")
    port = int(os.getenv("COMFYAI_ONNX_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "INFO")
    return Config(
        detector_model=detector_model,
        embedder_model=embedder_model,
        threshold=threshold,
        onnx_model_path=onnx_model_path,
        port=port,
        log_level=log_level,
    )
