# Face Comparison API

This service compares two face images and returns a similarity score. It relies on ONNX models for detection and embedding.

## Installation

```bash
pip install fastapi uvicorn insightface onnxruntime[-gpu] python-multipart
```

## Running

```bash
uvicorn apps.face_api.main:app --preset photo
```

Configuration values are resolved in this order: **environment variables → CLI flags → presets**. Use a preset for zero-config startup or override individual settings via flags or environment variables.

Custom models example:

```bash
uvicorn apps.face_api.main:app \
  --detector-model deepghs/real_face_detection \
  --detector-file face_detect_v1.4_s/model.onnx \
  --embedder-model-path openailab/onnx-arcface-resnet100-ms1m \
  --embedder-file model.onnx
```

### Environment Variables

- `FACE_MODEL_PROVIDERS` – comma-separated ONNX providers (default: `CUDAExecutionProvider,CPUExecutionProvider`)
- `FACE_MODEL_NAME` – InsightFace model name (default: `buffalo_l`)
- `DETECTOR_MODEL` – HuggingFace repository for the face detector
- `DETECTOR_FILE` – path to the detector ONNX file within the repo
- `EMBEDDER_MODEL_PATH` – repository containing the embedding model
- `EMBEDDER_FILE` – path to the embedder ONNX file
- `DETECTOR_THRESHOLD` – similarity threshold override
- `LOG_LEVEL` / `FACE_API_LOG_LEVEL` – Python and server logging levels

Add or remove presets by editing `presets.py` in this directory.
