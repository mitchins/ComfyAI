# Face Comparison API

A lightweight FastAPI service that compares two face images. It downloads ONNX models from Hugging Face and returns an embedding similarity score.

## Installation

```bash
pip install fastapi uvicorn insightface onnxruntime[-gpu] python-multipart
```

## Usage

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

### Configuration

You can configure models and runtime behaviour using environment variables or CLI flags. Values are resolved in this order: **environment variables → CLI flags → presets**.

- `PRESET` – name from `apps/face_api/presets.py` (e.g. `photo`)
- `DETECTOR_MODEL` – HuggingFace repository for the face detector
- `DETECTOR_FILE` – path to the detector ONNX file
- `EMBEDDER_MODEL_PATH` – repository for the embedding model
- `EMBEDDER_FILE` – embedding ONNX file
- `DETECTOR_THRESHOLD` – similarity threshold override
- `FACE_MODEL_PROVIDERS` – comma separated ONNX providers (default: `CUDAExecutionProvider,CPUExecutionProvider`)
- `FACE_MODEL_NAME` – InsightFace model name (default: `buffalo_l`)
- `PRELOAD_MODELS` – set to `1` to load models at startup
- `FACE_API_LOG_LEVEL` / `LOG_LEVEL` – logging level

### Examples

Zero‑config preset:

```bash
uvicorn apps.face_api.main:app --preset photo
```

Custom models:

```bash
uvicorn apps.face_api.main:app \
  --detector-model deepghs/real_face_detection \
  --detector-file face_detect_v1.4_s/model.onnx \
  --embedder-model-path openailab/onnx-arcface-resnet100-ms1m \
  --embedder-file model.onnx
```

Add or remove presets by editing `apps/face_api/presets.py`.
