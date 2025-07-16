import io
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - optional dependency
    FaceAnalysis = None  # type: ignore


@lru_cache(maxsize=1)
def get_model():
    if FaceAnalysis is None:
        raise RuntimeError("InsightFace is not available")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Return the embedding vector for the first detected face."""
    if FaceAnalysis is None:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    img_np = np.array(img)[..., ::-1]  # RGB to BGR
    model = get_model()
    faces = model.get(img_np)
    if not faces:
        return None
    return faces[0].normed_embedding
