from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import asyncio
import logging
import os
import io
from typing import Optional, Union
import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import shutil
import math

 # Requires: pip install dghs-imgutils
from imgutils.detect.face import detect_faces

import logging

# Create a dedicated logger for the Face API
logger = logging.getLogger(__name__)

# Configure handler for our logger
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# Set logger level based on FACE_API_LOG_LEVEL env var at import time
import os
level_name = os.getenv("FACE_API_LOG_LEVEL", "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
logger.setLevel(level)
handler.setLevel(level)

DEFAULT_DETECTOR_REPO = "deepghs/anime_face_detection"
DEFAULT_DETECTOR_FILE = "face_detect_v1.4_s/model.onnx"
# Use the CLIP vision (image) model for embeddings
DEFAULT_EMBEDDER_REPO = "Xenova/clip-vit-base-patch32"
DEFAULT_EMBEDDER_FILE = "onnx/vision_model.onnx"

# Environment-driven model selection:
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", DEFAULT_DETECTOR_REPO)
DETECTOR_FILE = os.getenv("DETECTOR_FILE", DEFAULT_DETECTOR_FILE)
EMBEDDER_MODEL_PATH = os.getenv("EMBEDDER_MODEL_PATH", DEFAULT_EMBEDDER_REPO)
EMBEDDER_FILE = os.getenv("EMBEDDER_FILE", DEFAULT_EMBEDDER_FILE)

# Confidence thresholds recommended per model variant
_THRESHOLD_MAP = {
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
# Override via env var if needed
DEFAULT_THRESHOLD = float(os.getenv("DETECTOR_THRESHOLD",
    _THRESHOLD_MAP.get(DETECTOR_FILE, 0.5)
))

DEFAULT_LEVEL = os.getenv("DETECTOR_LEVEL", "s")
DEFAULT_VERSION = os.getenv("DETECTOR_VERSION", "v1.4")


app = FastAPI()

# Control preload of models at startup
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in ("1", "true", "yes")

@app.on_event("startup")
async def startup_event():
    if PRELOAD_MODELS:
        logger.info("Preloading models at startup...")
        try:
            _load_detector()
            _load_embedder()
            logger.info("Models preloaded successfully")
        except Exception:
            logger.critical("Preloading failed; aborting")
            raise

# Global model instances
_detector: Optional[ort.InferenceSession] = None
_embedder: Optional[ort.InferenceSession] = None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _get_providers():
    """Get available ONNX providers."""
    providers = []
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers

def _download_model(model_id: str, filename: str, cache_dir: str) -> str:
    """Download model from HuggingFace and return local path."""
    local_path = os.path.join(cache_dir, filename)
    # Ensure full path for nested filenames
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if not os.path.exists(local_path):
        try:
            downloaded_path = hf_hub_download(repo_id=model_id, filename=filename)
            shutil.copy(downloaded_path, local_path)
            logger.info(f"Downloaded {model_id}/{filename} to {local_path}")
        except Exception as e:
            logger.exception(f"Failed to download model {model_id}/{filename}; aborting startup")
            raise
    
    return local_path

def _load_detector() -> Optional[ort.InferenceSession]:
    """Load face detection model."""
    global _detector
    if _detector is None:
        try:
            cache_dir = os.path.expanduser("~/.cache/face_api/detector")
            model_path = _download_model(DETECTOR_MODEL, DETECTOR_FILE, cache_dir)
            providers = _get_providers()
            _detector = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"Loaded detector model from {model_path}")
        except Exception as e:
            logger.exception("Detector failed to load; aborting startup")
            raise
    return _detector

def _load_embedder() -> Optional[ort.InferenceSession]:
    """Load face embedding model."""
    global _embedder
    if _embedder is None:
        try:
            cache_dir = os.path.expanduser("~/.cache/face_api/embedder")
            model_path = _download_model(EMBEDDER_MODEL_PATH, EMBEDDER_FILE, cache_dir)
            providers = _get_providers()
            _embedder = ort.InferenceSession(model_path, providers=providers)
            logger.info(f"Loaded embedder model from {model_path}")
        except Exception as e:
            logger.exception("Embedder failed to load; aborting startup")
            raise
    return _embedder

def _preprocess_image_for_detection(image: np.ndarray) -> np.ndarray:
    """Preprocess image for face detection."""
    # Normalize to 0-1 range
    image = image.astype(np.float32) / 255.0
    
    # Resize to standard size (adjust based on your model requirements)
    target_size = (640, 640)
    image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize(target_size))
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension and transpose to NCHW format
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    return image

def _preprocess_image_for_embedding(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for CLIP embedding."""
    # Resize
    image = np.array(Image.fromarray(image).resize(target_size))
    
    # Normalize using CLIP's normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    image = image.astype(np.float32)
    
    # Transpose to CHW format and add batch dimension
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    return image

def _extract_face_region(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """Extract face region from image using bounding box."""
    x1, y1, x2, y2 = bbox
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    return image[y1:y2, x1:x2]

def get_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Return face embedding for the first detected face or None."""
    logger.debug("Starting embedding extraction")
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        logger.debug(f"Image converted to array of shape {img_array.shape}")
        
        # Detect faces using imported detect_faces with threshold and other parameters
        raw_faces = detect_faces(
            img,
            level=os.getenv("DETECTOR_LEVEL", DEFAULT_LEVEL),
            version=os.getenv("DETECTOR_VERSION", DEFAULT_VERSION),
            conf_threshold=float(os.getenv("DETECTOR_THRESHOLD", DEFAULT_THRESHOLD))
        )
        # Extract bounding boxes from raw detections
        faces = [bbox for (bbox, _, _) in raw_faces]
        
        logger.info(f"Faces detected: {len(faces)}")
        if not faces:
            logger.warning(f"No faces detected ({len(faces)}); returning None embedding")
            return None
        
        # Extract first face
        face_bbox = faces[0]
        face_region = _extract_face_region(img_array, face_bbox)
        
        if face_region.size == 0:
            logger.warning("Extracted face region is empty")
            return None
        
        # Get embedding using CLIP
        embedder = _load_embedder()
        if embedder is None:
            logger.error("Embedder session is None; cannot extract embedding")
            return None
        
        # Preprocess face for embedding
        processed_face = _preprocess_image_for_embedding(face_region)
        
        # Run embedding inference
        input_name = embedder.get_inputs()[0].name
        outputs = embedder.run(None, {input_name: processed_face})
        
        # Extract embedding (usually the first output)
        embedding = outputs[0].flatten()
        
        # Normalize embedding
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

        # Get embeddings concurrently
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
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/models/info")
async def models_info():
    """Get information about loaded models."""
    return {
        "detector_model": f"{DETECTOR_MODEL}/{DETECTOR_FILE}",
        "embedder_model": f"{EMBEDDER_MODEL_PATH}/{EMBEDDER_FILE}",
        "detector_loaded": _detector is not None,
        "embedder_loaded": _embedder is not None
    }

def main():  # pragma: no cover
    import uvicorn
    # Note: Models are preloaded on startup if PRELOAD_MODELS env var is set to true (1, yes)
    # This ensures models are loaded before accepting requests.
    log_level_name = os.getenv("FACE_API_LOG_LEVEL", "INFO").upper()
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level=log_level_name.lower())

if __name__ == "__main__":  # pragma: no cover
    main()