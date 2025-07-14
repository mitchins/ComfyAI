import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    images: List[str] | None = None
    max_tokens: int | None = None

session = None
MODEL_PATH = os.environ.get("ONNX_MODEL_PATH")


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

    text = req.messages[-1].get("content", "") if req.messages else ""
    # Currently we simply ignore the images but validate the input
    if req.images:
        _ = len(req.images)
    result = classify(text)
    return {
        "id": "cmpl-001",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}
        ],
        "model": req.model,
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
