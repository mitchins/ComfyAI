# Face Comparison API

A lightweight FastAPI service that compares two face images. It downloads ONNX models from Hugging Face and returns an embedding similarity score.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, you may replace `onnxruntime` with `onnxruntime-gpu`.

## Usage

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

### Configuration

You can configure models and runtime behaviour using environment variables or CLI flags. Values are resolved in this order: **environment variables → CLI flags → presets**.

- `PRESET` – name from `apps/face_api/presets.py` (e.g. `photo`)
- `DETECTOR_MODEL` – HuggingFace repository for the face detector
- `DETECTOR_FILE` – path to the detector ONNX file
- `EMBEDDER_MODEL_PATH` – repository for the embedding model
- `EMBEDDER_FILE` – embedding ONNX file
- `DETECTOR_THRESHOLD` – similarity threshold override
- `FACE_MODEL_PROVIDERS` – comma separated ONNX providers (default: `CUDAExecutionProvider,CPUExecutionProvider`)
- `FACE_MODEL_NAME` – InsightFace model name (default: `buffalo_l`)
- `PRELOAD_MODELS` – set to `1` to load models at startup
- `FACE_API_LOG_LEVEL` / `LOG_LEVEL` – logging level

### Built-in Presets

Name | Detector | Embedder | Threshold | Use case
---- | -------- | -------- | --------- | --------
**photo** | `deepghs/real_face_detection` + `face_detect_v1.4_s` | `openailab/onnx-arcface-resnet100-ms1m` | `0.446` | Real photographs (default)
**anime** | `deepghs/anime_face_detection` + `face_detect_v1.4_s` | `Xenova/clip-vit-base-patch32` | `0.307` | Anime-style images
**cg** | `deepghs/real_face_detection` + `face_detect_v1.4_n` | `Xenova/clip-vit-base-patch32` | `0.278` | CG/digital characters

The **photo** preset is recommended for most workflows and is used in the documentation examples above.

### Examples

Zero‑config preset:

```bash
PRESET=photo uvicorn apps.face_api.main:app
```

Custom models:

```bash
DETECTOR_MODEL=deepghs/real_face_detection \
DETECTOR_FILE=face_detect_v1.4_s/model.onnx \
EMBEDDER_MODEL_PATH=openailab/onnx-arcface-resnet100-ms1m \
EMBEDDER_FILE=model.onnx \
uvicorn apps.face_api.main:app
```

Add or remove presets by editing `apps/face_api/presets.py`.
