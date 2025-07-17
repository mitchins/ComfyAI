# Embedded Servers

This folder contains optional HTTP services used by the nodes.

## ONNX Chat Completion Server

Lightweight OpenAI compatible endpoint for testing locally.
Run it with:

```bash
python -m apps.onnx_chat.main
```

See [onnx_chat/README.md](onnx_chat/README.md) for configuration and Docker usage.

## Face Comparison API

Compares two face images and returns a similarity score.
Start it with:

```bash
uvicorn apps.face_api.main:app
```

See [face_api/README.md](face_api/README.md) for installation and advanced options.
