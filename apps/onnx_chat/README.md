# ONNX Chat Completion Server

A lightweight OpenAI-compatible endpoint. It can load a toy ONNX model or fall back to simple rules.

## Running

```bash
python -m apps.onnx_chat.main
```

You can also build and run the Docker image:

```bash
docker build -t onnx-chat apps/onnx_chat
docker run -p 8000:8000 onnx-chat
```

### Environment Variables

- `ONNX_MODEL_PATH` – path to an ONNX model file to load (optional)
- `COMFYAI_ONNX_PORT` – server port (default: `8000`)
- `ONNX_LOG_LEVEL` – uvicorn log level (default: `info`)
- `LOG_LEVEL` – Python logging level used for setup

