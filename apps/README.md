# Embedded Servers

The `apps/` directory contains optional HTTP services that expose lightweight APIs used by some nodes.

## ONNX Chat Completion Server

`onnx_chat/` hosts a minimal OpenAI compatible endpoint. It runs a toy ONNX model if available and otherwise falls back to a couple of simple rules.

Run it with:

```bash
uvicorn apps.onnx_chat.main:app --host 0.0.0.0 --port 8000
```

Set `ONNX_MODEL_PATH` to point at an ONNX model file if you have one. The server listens on port `8000` by default.
A `Dockerfile` is provided for convenience if you prefer containerised runs.

## Face Comparison API

`face_api/` hosts a small FastAPI application that compares two face images. It relies on InsightFace and ONNX Runtime to extract embeddings.

### Installation

Install the dependencies if you plan to run the API outside Docker:

```bash
pip install fastapi uvicorn insightface onnxruntime[-gpu] python-multipart
```

Launch it using:

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

Several environment variables let you tune which provider or model is used:

- `FACE_MODEL_PROVIDERS` – comma separated list of ONNX providers (default: `CUDAExecutionProvider,CPUExecutionProvider`)
- `FACE_MODEL_NAME` – name of the InsightFace model (default: `buffalo_l`)

The `CompareFacesNode` sends images to this API and receives a similarity score.
