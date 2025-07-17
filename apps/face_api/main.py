from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import asyncio
import io
import logging
import os
from typing import Optional
import numpy as np
from PIL import Image
try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None
from apps.shared.hf_utils import download_model

# Requires: pip install dghs-imgutils
try:
    from imgutils.detect.face import detect_faces
except Exception:  # pragma: no cover - optional dependency
    detect_faces = None

from .config import load_config, setup_logging

from contextlib import asynccontextmanager

setup_logging()
config = load_config()
logger = logging.getLogger(__name__)

# Constants and defaults
DEFAULT_THRESHOLD_MAP = {
    "face_detect_v1.4_s/model.onnx": 0.307,
    "face_detect_v1.4_n/model.onnx": 0.278,
    "face_detect_v1.3_n/model.onnx": 0.305,
    "face_detect_v1.2_s/model.onnx": 0.222,
    "face_detect_v1.3_s/model.onnx": 0.259,
    "face_detect_v1_s/model.onnx": 0.446,
    "face_detect_v1_n/model.onnx": 0.458,
    "face_detect_v0_n/model.onnx": 0.428,
    "face_detect_v1.1_n/model.onnx": 0.373,
    "face_detect_v1.1_s/model.onnx": 0.405,
}

DEFAULT_LEVEL = "s"
DEFAULT_VERSION = "v1.4"

DETECTOR_MODEL = config.detector_model
DETECTOR_FILE = config.detector_file
EMBEDDER_MODEL_PATH = config.embedder_model_path
EMBEDDER_FILE = config.embedder_file

DEFAULT_THRESHOLD = config.threshold

# Model loader class for modularity
class ModelLoader:
    def __init__(self):
        self._detector: Optional[object] = None
        self._embedder: Optional[object] = None

    def _get_providers(self):
        providers = []
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers


    def load_detector(self):
        if self._detector is None:
            try:
                cache_dir = os.path.expanduser("~/.cache/face_api/detector")
                model_path = download_model(DETECTOR_MODEL, DETECTOR_FILE, cache_dir)
                providers = self._get_providers()
                self._detector = ort.InferenceSession(model_path, providers=providers)
                logger.info(f"Loaded detector model from {model_path}")
            except Exception:
                logger.critical("Detector failed to load; aborting startup")
                raise
        return self._detector

    def load_embedder(self):
        if self._embedder is None:
            try:
                cache_dir = os.path.expanduser("~/.cache/face_api/embedder")
                model_path = download_model(EMBEDDER_MODEL_PATH, EMBEDDER_FILE, cache_dir)
                providers = self._get_providers()
                self._embedder = ort.InferenceSession(model_path, providers=providers)
                logger.info(f"Loaded embedder model from {model_path}")
            except Exception:
                logger.critical("Embedder failed to load; aborting startup")
                raise
        return self._embedder

# Preprocessing functions
def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    target_size = (640, 640)
    image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize(target_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_for_embedding(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    image = np.array(Image.fromarray(image).resize(target_size))
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def extract_face_region(image: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return image[y1:y2, x1:x2]

# Cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# FastAPI app and model loader instance

PRELOAD_MODELS = config.preload_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD_MODELS:
        logger.info("Preloading models at startup...")
        try:
            model_loader.load_detector()
            model_loader.load_embedder()
            logger.info("Models preloaded successfully")
        except Exception:
            logger.critical("Preloading failed; aborting")
            raise
    yield

app = FastAPI(lifespan=lifespan)
model_loader = ModelLoader()

app.state.detector_path = f"{DETECTOR_MODEL}/{DETECTOR_FILE}"
app.state.embedder_path = f"{EMBEDDER_MODEL_PATH}/{EMBEDDER_FILE}"
app.state.threshold = DEFAULT_THRESHOLD


def get_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    logger.debug("Starting embedding extraction")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        logger.debug(f"Image converted to array of shape {img_array.shape}")

        raw_faces = detect_faces(
            img,
            level=DEFAULT_LEVEL,
            version=DEFAULT_VERSION,
            conf_threshold=DEFAULT_THRESHOLD
        )
        faces = [bbox for (bbox, _, _) in raw_faces]

        logger.info(f"Faces detected: {len(faces)}")
        if not faces:
            logger.warning("No faces detected; returning None embedding")
            return None

        face_bbox = faces[0]
        face_region = extract_face_region(img_array, face_bbox)

        if face_region.size == 0:
            logger.warning("Extracted face region is empty")
            return None

        embedder = model_loader.load_embedder()
        if embedder is None:
            logger.error("Embedder session is None; cannot extract embedding")
            return None

        processed_face = preprocess_for_embedding(face_region)
        input_name = embedder.get_inputs()[0].name
        outputs = embedder.run(None, {input_name: processed_face})
        embedding = outputs[0].flatten()
        embedding = embedding / np.linalg.norm(embedding)
        logger.debug(f"Extracted embedding vector of length {len(embedding)}")

        return embedding

    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return None

@app.post("/v1/image/compare_faces")
async def compare_faces(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...)
):
    logger.info("Received compare_faces request")
    try:
        data_a = await image_a.read()
        data_b = await image_b.read()
        logger.debug(f"Read image sizes: a={len(data_a)} bytes, b={len(data_b)} bytes")

        emb_a, emb_b = await asyncio.gather(
            asyncio.to_thread(get_embedding, data_a),
            asyncio.to_thread(get_embedding, data_b),
        )

        logger.info(f"Embeddings: A={'None' if emb_a is None else emb_a.shape}, B={'None' if emb_b is None else emb_b.shape}")

        if emb_a is None or emb_b is None:
            logger.error("One or both embeddings are None; returning 422")
            return JSONResponse(
                status_code=422,
                content={"error": "face_not_detected", "message": "Could not detect face in one or both images"}
            )

        similarity = float(cosine_similarity(emb_a, emb_b))
        logger.info(f"Similarity score: {similarity}")

        return {"similarity": similarity}

    except Exception as e:
        logger.error(f"Error in compare_faces: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": DETECTOR_MODEL}

@app.get("/models/info")
async def models_info():
    return {
        "detector_model": f"{DETECTOR_MODEL}/{DETECTOR_FILE}",
        "embedder_model": f"{EMBEDDER_MODEL_PATH}/{EMBEDDER_FILE}",
        "detector_loaded": model_loader._detector is not None,
        "embedder_loaded": model_loader._embedder is not None
    }

def main():  # pragma: no cover
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Face comparison API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-file")
    parser.add_argument("--embedder-model-path")
    parser.add_argument("--embedder-file")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--preset")
    args = parser.parse_args()

    # reload config with CLI overrides
    global config, DETECTOR_MODEL, DETECTOR_FILE, EMBEDDER_MODEL_PATH, EMBEDDER_FILE, DEFAULT_THRESHOLD, PRELOAD_MODELS
    config = load_config([
        f"--detector-model={args.detector_model}" if args.detector_model else "",
        f"--detector-file={args.detector_file}" if args.detector_file else "",
        f"--embedder-model-path={args.embedder_model_path}" if args.embedder_model_path else "",
        f"--embedder-file={args.embedder_file}" if args.embedder_file else "",
        f"--threshold={args.threshold}" if args.threshold is not None else "",
        f"--preset={args.preset}" if args.preset else "",
    ])
    DETECTOR_MODEL = config.detector_model
    DETECTOR_FILE = config.detector_file
    EMBEDDER_MODEL_PATH = config.embedder_model_path
    EMBEDDER_FILE = config.embedder_file
    DEFAULT_THRESHOLD = config.threshold
    PRELOAD_MODELS = config.preload_models
    app.state.detector_path = f"{DETECTOR_MODEL}/{DETECTOR_FILE}"
    app.state.embedder_path = f"{EMBEDDER_MODEL_PATH}/{EMBEDDER_FILE}"
    app.state.threshold = DEFAULT_THRESHOLD

    uvicorn.run(
        "apps.face_api.main:app",
        host=args.host,
        port=args.port,
        log_level=config.log_level,
    )

if __name__ == "__main__":  # pragma: no cover
    main()
