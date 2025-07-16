# Embedded Servers

The `apps/` directory contains optional HTTP services that expose lightweight APIs used by some nodes.

## ONNX Chat Completion Server

`onnx_server.py` starts a minimal OpenAI compatible endpoint. It can run a toy ONNX model or fall back to very simple rules if no model is provided. Use it when you need a local endpoint for the query nodes.

Run it with:

```bash
python -m apps.onnx_server
```

Set `ONNX_MODEL_PATH` to point at an ONNX model file if you have one. The server listens on port `8000` by default.

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
