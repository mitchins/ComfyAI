"""Minimal OpenAI-compatible chat server using FastAPI."""

from typing import List, Dict, Any
import os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

try:
    import onnxruntime as ort  # optional
except Exception:  # pragma: no cover
    ort = None

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int | None = None

session = None
MODEL_PATH = os.environ.get("ONNX_MODEL_PATH")


def load_session():
    """Load the ONNX model on first use if available."""
    global session
    if session is None and ort and MODEL_PATH and os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH)


def classify(text: str) -> str:
    """Return a simple classification using the model or a rule based fallback."""
    load_session()
    if session is None:
        return "positive" if "good" in text.lower() else "negative"
    inputs = {session.get_inputs()[0].name: [[ord(c) for c in text]]}
    outputs = session.run(None, inputs)[0]
    return str(outputs[0])


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        data = await request.json()
        req = ChatRequest(**data)
    except Exception as e:  # JSON error or validation
        raise HTTPException(status_code=400, detail=str(e))

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


def main():  # pragma: no cover
    import uvicorn

    log_level = os.getenv("ONNX_CHAT_LOG_LEVEL", "info")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level=log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
