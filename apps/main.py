from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import importlib
import pkgutil
from apps.shared.manage_cache import list_cached_entries
from apps.shared.model_types import ModelType
from common.logging import setup_logging

setup_logging()
app = FastAPI(title="ComfyAI Master API")

STATIC_DIR = Path(__file__).resolve().parent / "static" / "manage"

# Auto-discover and register all routers
def register_routers():
    """Auto-discover routers from apps/* directories."""
    apps_dir = Path(__file__).parent
    
    for app_path in apps_dir.iterdir():
        if (app_path.is_dir() and 
            not app_path.name.startswith('_') and 
            not app_path.name == 'shared' and
            (app_path / 'router.py').exists()):
            
            try:
                # Import the router module
                module_name = f"apps.{app_path.name}.router"
                module = importlib.import_module(module_name)
                
                if hasattr(module, 'router'):
                    router = module.router
                    service_name = app_path.name.replace('_', '-')
                    
                    # Register with service prefix
                    app.include_router(router, prefix=f"/{service_name}", tags=[service_name])
                    print(f"✅ Registered {service_name} router at /{service_name}")
                    
                    # Special case: ONNX chat also gets root-level OpenAI compatibility
                    if app_path.name == 'onnx_chat':
                        app.include_router(router, tags=["openai-compatible"])
                        print(f"✅ Registered onnx-chat router at root level for OpenAI compatibility")
                        
            except Exception as e:
                print(f"❌ Failed to register {app_path.name} router: {e}")

register_routers()
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
