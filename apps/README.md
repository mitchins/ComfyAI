# Embedded Servers

The `apps/` directory contains lightweight HTTP services used by the nodes. Each service is self contained so you can run them independently or inside Docker.

## ONNX Chat Completion Server

The `onnx_chat/` folder exposes a tiny OpenAI compatible endpoint. It can load a toy ONNX model if `ONNX_MODEL_PATH` is set, otherwise a simple rule based reply is returned.

Start the server locally with:

```bash
python -m apps.onnx_chat.main
```

Or via the convenience entrypoint:

```bash
comfyai-onnx-server
```

By default the server listens on port `8000`. Supply `PORT` to override. A `Dockerfile` is also provided so you can build an image with:

```bash
docker build -t onnx-chat apps/onnx_chat
```

Run it using:

```bash
docker run -p 8000:8000 onnx-chat
```

## Face Comparison API

`face_api/` hosts a FastAPI application to compare two face images. It uses InsightFace and ONNX Runtime to extract embeddings.

Install the dependencies if running outside Docker:

```bash
pip install -r apps/face_api/requirements.txt
```

Launch it with uvicorn:

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

A `Dockerfile` is provided in the same directory. Build it with `docker build -t face-api apps/face_api` and run with `docker run -p 7860:7860 face-api`.

Environment variables allow you to select models or execution providers. Refer to `apps/face_api/main.py` for details.
