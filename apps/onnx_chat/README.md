# ONNX Chat Completion Server

A minimal OpenAI compatible endpoint that can optionally load an ONNX model. Useful when you need a local service for the query nodes.

## Installation

```bash
pip install -r apps/onnx_chat/requirements.txt
```

## Running

```bash
python -m apps.onnx_chat.main
```

By default the server listens on `0.0.0.0:8000`. You can override settings with:

- `ONNX_MODEL_PATH` – path to an ONNX model file
- `COMFYAI_ONNX_PORT` – port to listen on
- `ONNX_LOG_LEVEL` – uvicorn log level
- `LOG_LEVEL` – common logging level

A Dockerfile is provided for containerised usage.
