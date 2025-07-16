from __future__ import annotations
import io
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
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort is not None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                providers = ["CPUExecutionProvider"]
        _model = FaceAnalysis(name="buffalo_l", providers=providers)
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
