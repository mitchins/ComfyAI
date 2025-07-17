# Face Comparison API

This FastAPI service compares two face images and returns a similarity score using InsightFace and ONNX Runtime.
Configuration can be supplied via environment variables, CLI flags or presets.

## Installation

```bash
pip install fastapi uvicorn insightface onnxruntime[-gpu] python-multipart
```

## Running

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

Use `--preset` to pick a model preset from `presets.py`. Values are resolved in this order:
**environment variables → CLI flags → presets**.

### Environment Variables

- `FACE_MODEL_PROVIDERS` – comma separated ONNX providers
- `FACE_MODEL_NAME` – InsightFace model name
- `DETECTOR_MODEL` – repository with the detector model
- `DETECTOR_FILE` – detector ONNX path
- `EMBEDDER_MODEL_PATH` – repository with the embedding model
- `EMBEDDER_FILE` – embedder ONNX path
- `DETECTOR_THRESHOLD` – similarity threshold override
- `PRELOAD_MODELS` – preload detector and embedder at startup
- `PRESET` – preset name to load

Edit `presets.py` to add or remove presets.
