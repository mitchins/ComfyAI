import logging
import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import load_config, Config

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

config: Config = load_config()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, config.log_level.upper(), logging.INFO),
)
logger = logging.getLogger(__name__)

app = FastAPI()


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int | None = None


session = None


def load_session():
    global session
    if session is None and ort and config.model_path and os.path.exists(config.model_path):
        session = ort.InferenceSession(config.model_path)
        logger.info("Loaded ONNX model from %s", config.model_path)


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


@app.get("/health")
async def health() -> Dict[str, str]:
    model_name = os.path.basename(config.model_path) if config.model_path else "builtin"
    return {"status": "ok", "model": model_name}


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ONNX chat server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("apps.onnx_chat.main:app", host=args.host, port=args.port, reload=args.reload, log_level=config.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
