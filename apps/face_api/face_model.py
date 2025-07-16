import io
import os
from typing import Optional

import numpy as np
from PIL import Image

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - optional heavy dep
    FaceAnalysis = None  # type: ignore

_model: Optional["FaceAnalysis"] = None


def _load_model() -> Optional["FaceAnalysis"]:
    """Load the InsightFace model if available."""
    global _model
    if _model is None and FaceAnalysis is not None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _model = FaceAnalysis(name="buffalo_l", providers=providers)
        _model.prepare(ctx_id=0, det_size=(640, 640))
    return _model


def get_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Return face embedding for the first detected face, or None."""
    model = _load_model()
    if model is None:
        return None

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.asarray(img)
    faces = model.get(img_np)
    if not faces:
        return None
    return faces[0].embedding
