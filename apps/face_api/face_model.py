import numpy as np
from io import BytesIO
from PIL import Image

from insightface.app import FaceAnalysis
import onnxruntime as ort

_model = None

def _init_model():
    global _model
    if _model is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                providers = ["CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]
        _model = FaceAnalysis(name="buffalo_l", providers=providers)
        _model.prepare(ctx_id=0)
    return _model

def get_embedding(image_bytes: bytes) -> np.ndarray | None:
    model = _init_model()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    faces = model.get(arr)
    if not faces:
        return None
    face = faces[0]
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = face.embedding
    return emb
