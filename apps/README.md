# Embedded Servers

This directory hosts optional HTTP services used by some ComfyAI nodes.
Each service can run locally for lightweight inference.

## onnx_chat

A minimal OpenAI-compatible chat completion server.

Run it with:

```bash
python -m apps.onnx_chat.main
```

See [onnx_chat/README.md](onnx_chat/README.md) for all configuration options and
Docker usage.

## face_api

A FastAPI application that compares two faces and returns a similarity score.

Start it with:

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

See [face_api/README.md](face_api/README.md) for presets, environment variables
and CLI examples.
