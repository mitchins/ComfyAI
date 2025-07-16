# Embedded Servers

The `apps/` directory contains optional HTTP services that expose lightweight APIs used by some nodes.

## ONNX Chat Completion Server

`onnx_chat/` contains a minimal OpenAI compatible endpoint. It can run a toy ONNX model or fall back to very simple rules if no model is provided. Use it when you need a local endpoint for the query nodes.

Run it with:

```bash
python -m apps.onnx_chat.main
```

Set `ONNX_MODEL_PATH` to point at an ONNX model file if you have one. The server listens on port `8000` by default and honours `COMFYAI_ONNX_PORT` and `LOG_LEVEL`.

Build a container using the supplied Dockerfile if preferred:

```bash
docker build -t onnx-chat apps/onnx_chat
docker run -p 8000:8000 onnx-chat
```

## Face Comparison API

`face_api/` hosts a small FastAPI application that compares two face images. It relies on InsightFace and ONNX Runtime to extract embeddings.

### Installation

Install the dependencies if you plan to run the API outside Docker:

```bash
pip install fastapi uvicorn insightface onnxruntime[-gpu] python-multipart
```

Launch it using:

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

Or build and run the Docker image:

```bash
docker build -t face-api apps/face_api
docker run -p 7860:7860 face-api
```

To bake defaults into the image supply build arguments:

```bash
docker build \
  --build-arg DETECTOR_MODEL=deepghs/real_face_detection \
  --build-arg DETECTOR_FILE=face_detect_v1.4_s/model.onnx \
  --build-arg EMBEDDER_MODEL_PATH=Xenova/clip-vit-base-patch32 \
  --build-arg EMBEDDER_FILE=open_clip_pytorch_model.bin \
  apps/face_api
```
The Dockerfile uses a multi-stage build so the final image remains under 600MB.

Example overriding models and threshold:

```bash
DETECTOR_MODEL=deepghs/real_face_detection \
EMBEDDER_MODEL_PATH=Xenova/clip-vit-base-patch32 \
DETECTOR_FILE=face_detect_v1.4_s/model.onnx \
EMBEDDER_FILE=open_clip_pytorch_model.bin \
DETECTOR_THRESHOLD=0.65 \
uvicorn apps.face_api.main:app --reload
```

`DETECTOR_THRESHOLD` defaults per preset but can always be overridden.

Several environment variables let you tune which provider or model is used:

- `FACE_MODEL_PROVIDERS` – comma separated list of ONNX providers (default: `CUDAExecutionProvider,CPUExecutionProvider`)
- `FACE_MODEL_NAME` – name of the InsightFace model (default: `buffalo_l`)
- `DETECTOR_MODEL` – HuggingFace repo containing the face detector
- `DETECTOR_FILE` – detector file path inside the repo
- `EMBEDDER_MODEL_PATH` – repo containing the embedder model
- `EMBEDDER_FILE` – embedder file path inside the repo
- `DETECTOR_THRESHOLD` – override the similarity threshold (preset provides defaults)

The `CompareFacesNode` sends images to this API and receives a similarity score.
