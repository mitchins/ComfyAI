"""Simple OpenAI compatible server backed by an ONNX classifier.

This optional server demonstrates how a lightweight ONNX model can
be wrapped with an OpenAI style API. It is intended as an example and
is not used by default.
"""

import argparse
import time
import uuid
from typing import List, Dict

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("onnxruntime is required to run this server") from e

app = FastAPI()

LABELS = ["negative", "positive"]


def build_dummy_session() -> ort.InferenceSession:
    """Create a tiny ONNX model for demonstration if none is provided."""
    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    W = np.array([[0.5, -0.5, 0.1], [-0.3, 0.4, 0.2]], dtype=np.float32)
    B = np.array([0.0, 0.0], dtype=np.float32)

    node1 = helper.make_node("MatMul", ["input", "W"], ["XW"])
    node2 = helper.make_node("Add", ["XW", "B"], ["XWB"])
    node3 = helper.make_node("Softmax", ["XWB"], ["output"], axis=1)

    graph = helper.make_graph(
        [node1, node2, node3],
        "DummyClassifier",
        [X],
        [Y],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten()),
            helper.make_tensor("B", TensorProto.FLOAT, B.shape, B.flatten()),
        ],
    )

    model = helper.make_model(graph, producer_name="dummy")
    onnx.checker.check_model(model)
    tmp = "dummy_model.onnx"
    onnx.save(model, tmp)
    return ort.InferenceSession(tmp)


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict]


def text_to_features(text: str) -> np.ndarray:
    length = len(text)
    vowels = sum(c.lower() in "aeiou" for c in text)
    chars = sum(not c.isspace() for c in text)
    return np.array([[float(length), float(vowels), float(chars)]], dtype=np.float32)


def classify(text: str, session: ort.InferenceSession) -> str:
    feats = text_to_features(text)
    probs = session.run(None, {"input": feats})[0]
    label_idx = int(np.argmax(probs))
    return LABELS[label_idx]


@app.post("/v1/chat/completions")
def completions(req: ChatRequest):
    text = req.messages[-1].get("content", "")
    label = classify(text, app.state.session)
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": label},
                "finish_reason": "stop",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX OpenAI compatible server")
    parser.add_argument("--model", help="Path to ONNX model", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.model:
        session = ort.InferenceSession(args.model)
    else:
        session = build_dummy_session()

    app.state.session = session

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
