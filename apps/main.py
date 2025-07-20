from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from apps.chat_server.router import router as chat_router
from apps.face_server.router import router as face_router
from apps.manage_api.router import router as manage_router
from apps.shared.manage_cache import list_cached_entries
from apps.shared.model_types import ModelType
from common.logging import setup_logging

setup_logging()
app = FastAPI(title="ComfyAI Master API")

STATIC_DIR = Path(__file__).resolve().parent / "static" / "manage"

app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(face_router, prefix="/face", tags=["face"])
app.include_router(manage_router, prefix="/manage", tags=["manage"])
app.mount(
    "/manage/ui",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="manage_ui",
)


@app.get("/v1/models")
async def list_models():
    """List available chat-compatible models in OpenAI-compatible format."""
    try:
        cached_entries = list_cached_entries()
        
        # Filter for ONNX models that are LLM-capable and create OpenAI-compatible response
        models = []
        chat_compatible_types = ModelType.chat_compatible_types()
        
        for entry in cached_entries:
            if (entry["path"].endswith(".onnx") and 
                entry["kind"] in [t.value for t in chat_compatible_types]):
                
                # Always create full model ID with repo and complete file path
                # This gives users the exact string they need for API calls
                model_id = f"{entry['repo']}/{entry['path']}"
                
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(entry["last_used"]),
                    "owned_by": entry["repo"].split("/")[0] if "/" in entry["repo"] else "huggingface",
                })
        
        return {
            "object": "list",
            "data": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
