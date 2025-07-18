import os
import sys
import pytest
from starlette.testclient import TestClient
from fastapi import FastAPI
from apps.manage_api.router import router as manage_router

# Ensure required env vars so face_api can import
os.environ.setdefault("DETECTOR_MODEL", "fake")
os.environ.setdefault("DETECTOR_FILE", "model.onnx")
os.environ.setdefault("EMBEDDER_MODEL_PATH", "fake")
os.environ.setdefault("EMBEDDER_FILE", "model.onnx")

app = FastAPI()
app.include_router(manage_router)


@pytest.mark.integration
def test_get_remote_onnx_files_integration():
    client = TestClient(app)
    resp = client.get("/repos/onnx-community/granite-3.0-2b-instruct/files")
    assert resp.status_code == 200
    files = resp.json()
    paths = [e["path"] for e in files]
    assert any(p.endswith(".onnx") for p in paths)
    assert any("model_q4.onnx" in p for p in paths)
