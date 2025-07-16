# ONNX Chat Completion Server

This directory provides a small FastAPI application exposing an OpenAI compatible `/v1/chat/completions` endpoint.  If the optional `ONNX_MODEL_PATH` environment variable points to a local ONNX model, it will be loaded via `onnxruntime`; otherwise a tiny rule based fallback is used.

## Running locally

```bash
uvicorn apps.onnx_chat.main:app --host 0.0.0.0 --port 8000
```

Set `ONNX_MODEL_PATH` to the path of an ONNX model if available. The server listens on port `8000` by default.

## Docker

A minimal `Dockerfile` is provided for convenience:

```bash
docker build -t comfyai-onnx-chat .
docker run -p 8000:8000 comfyai-onnx-chat
```
