# Face Comparison API

This FastAPI application compares two face images and returns a similarity score.
It uses ONNX Runtime and optional InsightFace models.

## Installation

Install dependencies if running outside Docker:

```bash
pip install fastapi uvicorn insightface onnxruntime[-gpu] python-multipart
```

## Running

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

Configuration values are resolved in this order:
**environment variables → CLI flags → presets**.

Zero‑config example using the built-in *photo* preset:

```bash
uvicorn apps.face_api.main:app --preset photo
```

Custom models via CLI flags:

```bash
uvicorn apps.face_api.main:app \
  --detector-model deepghs/real_face_detection \
  --detector-file face_detect_v1.4_s/model.onnx \
  --embedder-model-path openailab/onnx-arcface-resnet100-ms1m \
  --embedder-file model.onnx
```

Add or remove presets by editing `presets.py`.

### Environment Variables

- `FACE_MODEL_PROVIDERS` – comma separated ONNX providers (default:
  `CUDAExecutionProvider,CPUExecutionProvider`)
- `FACE_MODEL_NAME` – InsightFace model name (default: `buffalo_l`)
- `DETECTOR_MODEL` – Hugging Face repo for the detector
- `DETECTOR_FILE` – path to the detector ONNX file
- `EMBEDDER_MODEL_PATH` – repository containing the embedder
- `EMBEDDER_FILE` – path to the embedder ONNX model
- `DETECTOR_THRESHOLD` – similarity threshold override
- `PRELOAD_MODELS` – "true" to load models at startup
- `FACE_API_LOG_LEVEL` – log level for the server

