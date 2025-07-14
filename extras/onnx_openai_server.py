"""Simple OpenAI compatible server using an ONNX model."""
import os
import base64
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

app = FastAPI()

session = None


def load_model(path: str) -> None:
    global session
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    session = ort.InferenceSession(path)


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if session is None:
        return {"choices": [{"message": {"role": "assistant", "content": "Model not loaded"}}]}
    # Extract text from the last message
    text = " ".join(
        part.get("text", "") for msg in req.messages for part in msg.get("content", []) if part.get("type") == "text"
    )
    # Dummy example: run model expecting length of text as input
    import numpy as np

    inp = np.array([[len(text)]], dtype=np.float32)
    out = session.run(None, {session.get_inputs()[0].name: inp})
    result = str(out[0].flatten()[0])
    return {"choices": [{"message": {"role": "assistant", "content": result}}]}


def main():
    model_path = os.environ.get("ONNX_MODEL_PATH")
    if model_path:
        load_model(model_path)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()

