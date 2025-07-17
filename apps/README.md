# Embedded Servers

The `apps/` folder bundles optional HTTP services used by some nodes.

## ONNX Chat Completion Server

A lightweight OpenAI-compatible endpoint. Start it with:

```bash
python -m apps.onnx_chat.main
```

See [onnx_chat/README.md](onnx_chat/README.md) for configuration options.

## Face Comparison API

Compares two face images and returns a similarity score. Run it with:

```bash
uvicorn apps.face_api.main:app --port 7860
```

Full documentation is available in [face_api/README.md](face_api/README.md).
