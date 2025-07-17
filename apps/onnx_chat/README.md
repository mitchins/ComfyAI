# ONNX Chat Completion Server

A minimal OpenAI-compatible endpoint that can run an ONNX model for text classification or fall back to simple rules.

## Installation

Install the optional extras or the requirements file:

```bash
pip install -r apps/onnx_chat/requirements.txt
```

## Usage

```bash
python -m apps.onnx_chat.main
```

The server can download ONNX models from Hugging Face on demand. Pass the
`model` field as `"repo_id[:filename]"` in your request. If `filename` is omitted,
`model.onnx` is assumed. Files are cached under `~/.cache/onnx_chat/`.

### Configuration

- `ONNX_MODEL_PATH` – path to an ONNX model to load (optional)
- `ONNX_FILE_NAME` – default filename when downloading from Hugging Face
- `COMFYAI_ONNX_PORT` – listening port (default `8000`)
- `ONNX_LOG_LEVEL` – uvicorn log level (default `info`)
- `LOG_LEVEL` – Python logging level

Environment variables configure the server. You can override the port and log level with CLI flags when launching via `python -m`.
