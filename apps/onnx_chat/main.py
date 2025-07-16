from __future__ import annotations
import os
from typing import List, Dict, Any
import argparse
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import load_config, setup_logging

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

setup_logging()
config = load_config()

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int | None = None


@app.get("/health")
async def health_check():
    model_name = os.path.basename(MODEL_PATH) if MODEL_PATH else "none"
    return {"status": "ok", "model": model_name}

session = None
MODEL_PATH = config.model_path


def load_session():
    global session
    if session is None and ort and MODEL_PATH and os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH)


def classify(text: str) -> str:
    load_session()
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
    result = classify(text)
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
