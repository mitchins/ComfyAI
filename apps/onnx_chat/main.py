from __future__ import annotations
import os
from typing import List, Dict, Any
import argparse
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import load_config, setup_logging
from apps.shared.hf_utils import download_model

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

setup_logging()
config = load_config()

sessions: dict[str, object | None] = {}
MODEL_PATH = config.model_path
DEFAULT_FILE = os.getenv("ONNX_FILE_NAME", "model.onnx")

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int | None = None


@app.get("/health")
async def health_check():
    model_name = os.path.basename(MODEL_PATH) if MODEL_PATH else "none"
    return {"status": "ok", "model": model_name}


def _ensure_model_path(name: str) -> str | None:
    if os.path.exists(name):
        return name
    repo_id, filename = (name.split(":", 1) + [DEFAULT_FILE])[:2]
    cache_dir = os.path.expanduser("~/.cache/onnx_chat")
    try:
        return download_model(repo_id, filename, cache_dir)
    except Exception:
        logging.getLogger(__name__).exception("Failed to download %s", name)
        return None


def load_session(model_name: str) -> object | None:
    if model_name not in sessions:
        if ort is None:
            sessions[model_name] = None
        else:
            path = _ensure_model_path(model_name) or MODEL_PATH
            if path and os.path.exists(path):
                try:
                    sessions[model_name] = ort.InferenceSession(path)
                except Exception:
                    logging.getLogger(__name__).exception("Failed to load model %s", path)
                    sessions[model_name] = None
            else:
                sessions[model_name] = None
    return sessions[model_name]


def classify(text: str, model_name: str) -> str:
    session = load_session(model_name)
    if session is None:
        # Fallback simple rule when ONNX model is unavailable
        return "positive" if "good" in text.lower() else "negative"
    inputs = {session.get_inputs()[0].name: [[ord(c) for c in text]]}
    outputs = session.run(None, inputs)[0]
    return str(outputs[0])


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        req = ChatRequest(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    text = ""
    if req.messages:
        content = req.messages[-1].get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
        else:
            text = str(content)
    result = classify(text, req.model or (MODEL_PATH or ""))
    return {
        "id": "cmpl-001",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}
        ],
        "model": req.model,
    }


def main() -> None:
    """CLI entry point for running the server via ``python -m``."""
    import uvicorn

    parser = argparse.ArgumentParser(description="ONNX chat completion server")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "apps.onnx_chat.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
