# Embedded Servers

The `apps/` directory contains optional HTTP services used by some nodes.

## onnx_chat

A minimal OpenAI-compatible endpoint that can run a small ONNX model or fall back to simple rules.
Run it with:

```bash
python -m apps.onnx_chat.main
```

See [onnx_chat/README.md](onnx_chat/README.md) for configuration details and environment variables.

## face_api

FastAPI service for comparing two face images. Start it with:

```bash
uvicorn apps.face_api.main:app
```

See [face_api/README.md](face_api/README.md) for installation steps, presets, and override order.
