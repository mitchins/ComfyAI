# ONNX Chat Completion Server

This lightweight server exposes an OpenAI compatible `/chat/completions` API.
It can run a small ONNX model for quick, offline testing.

## Running

```bash
python -m apps.onnx_chat.main
```

Set `ONNX_MODEL_PATH` to point at an ONNX file if you have one.
The server listens on `COMFYAI_ONNX_PORT` (default `8000`) and
uses `ONNX_LOG_LEVEL` for uvicorn's log level.

A Dockerfile is provided:

```bash
docker build -t onnx-chat apps/onnx_chat
docker run -p 8000:8000 onnx-chat
```
