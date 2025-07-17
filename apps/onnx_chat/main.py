from __future__ import annotations
import os
from typing import List, Dict, Any

from apps.hf_cache import download_repo
import argparse
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import load_config, setup_logging

try:  # optional heavy deps
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

setup_logging()
config = load_config()

HF_CACHE_DIR = config.hf_cache_dir
PIPELINES: dict[str, Any] = {}

app = FastAPI()


def load_pipeline(model_id: str):
    """Return a cached text generation pipeline for the given model."""
    if pipeline is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        return None
    if model_id not in PIPELINES:
        cache_dir = os.path.join(HF_CACHE_DIR, model_id.replace("/", "_"))
        local_repo = download_repo(model_id, cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_repo)
        model = AutoModelForCausalLM.from_pretrained(local_repo)
        PIPELINES[model_id] = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return PIPELINES[model_id]

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int | None = None


@app.get("/health")
async def health_check():
    model_name = os.path.basename(MODEL_PATH) if MODEL_PATH else "none"
    return {
        "status": "ok",
        "model": model_name,
        "loaded_pipelines": list(PIPELINES.keys()),
    }

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
    pipe = load_pipeline(req.model)
    if pipe is not None:
        try:
            out = pipe(text, max_new_tokens=req.max_tokens or 16)
            result = out[0].get("generated_text", "")
        except Exception:
            logging.getLogger(__name__).exception("Pipeline inference failed; falling back")
            result = classify(text)
    else:
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
