# Embedded Servers

The `apps/` folder bundles optional HTTP services used by some nodes.

Install server requirements first:

```bash
pip install -r onnx_chat/requirements.txt
pip install -r face_api/requirements.txt
```

## ONNX Chat Completion Server

A lightweight OpenAI-compatible endpoint. Start it with:

```bash
python -m apps.onnx_chat.main  # or: uvicorn apps.onnx_chat.main:app
```

See [onnx_chat/README.md](onnx_chat/README.md) for configuration options.

## Face Comparison API

Compares two face images and returns a similarity score. Run it with:

```bash
PRESET=photo|anime|cg uvicorn apps.face_api.main:app
```

Full documentation is available in [face_api/README.md](face_api/README.md).
