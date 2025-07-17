# ONNX Chat Completion Server

A minimal OpenAI-compatible endpoint that can run an ONNX model for text classification or fall back to simple rules. Models
can be referenced directly from Hugging Face by passing ``repo_id/path/to/file.onnx`` in the ``model`` field.

## Installation

Install the optional extras or the requirements file:

```bash
pip install -r apps/onnx_chat/requirements.txt
```

## Usage

```bash
python -m apps.onnx_chat.main
```

### Configuration

- `ONNX_MODEL_PATH` – optional path to preload a local ONNX model
- `COMFYAI_ONNX_PORT` – listening port (default `8000`)
- `ONNX_LOG_LEVEL` – uvicorn log level (default `info`)
- `LOG_LEVEL` – Python logging level

Environment variables configure the server. You can override the port and log level with CLI flags when launching via `python -m`.
