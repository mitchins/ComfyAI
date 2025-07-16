from __future__ import annotations
import io
import os
from typing import Optional

import numpy as np
from PIL import Image

try:
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional heavy deps
    FaceAnalysis = None  # type: ignore
    ort = None  # type: ignore

_model: FaceAnalysis | None = None


def _load_model() -> FaceAnalysis | None:
    global _model
    if _model is None and FaceAnalysis is not None:
        prov_env = os.getenv("FACE_MODEL_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
        providers = [p.strip() for p in prov_env.split(",") if p.strip()]
        if ort is not None:
            available = getattr(ort, "get_available_providers", lambda: providers)()
            if "CUDAExecutionProvider" not in available:
                providers = [p for p in providers if p != "CUDAExecutionProvider"] or ["CPUExecutionProvider"]
        model_name = os.getenv("FACE_MODEL_NAME", "buffalo_l")
        _model = FaceAnalysis(name=model_name, providers=providers)
        _model.prepare(ctx_id=0)
    return _model


def get_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Return face embedding for the first detected face or None."""
    model = _load_model()
    if model is None:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    faces = model.get(np.array(img))
    if not faces:
        return None
    return faces[0].normed_embedding
