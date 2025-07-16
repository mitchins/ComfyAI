import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from .config import load_config

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

app = FastAPI()

config = load_config()

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(handler)
root_level = getattr(logging, config.log_level.upper(), logging.INFO)
logging.getLogger().setLevel(root_level)
handler.setLevel(root_level)

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int | None = None

session = None
MODEL_PATH = config.onnx_model_path
MODEL_NAME = os.path.basename(MODEL_PATH) if MODEL_PATH else "rules"


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


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ONNX Chat Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=config.log_level)
    args = parser.parse_args()

    uvicorn.run(
        "apps.onnx_chat.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
