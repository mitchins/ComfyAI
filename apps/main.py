from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
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

# Root redirect to main UI
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with helpful navigation to all UIs."""
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ComfyAI Server</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px; margin: 50px auto; padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: white;
        }}
        .container {{ 
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 16px; padding: 2rem; text-align: center;
        }}
        h1 {{ font-size: 2.5rem; margin-bottom: 1rem; }}
        .links {{ display: grid; gap: 1rem; margin: 2rem 0; }}
        .link {{ 
            display: block; padding: 1rem 2rem; background: rgba(255,255,255,0.2);
            border-radius: 8px; text-decoration: none; color: white;
            transition: all 0.3s ease; font-size: 1.1rem;
        }}
        .link:hover {{ 
            background: rgba(255,255,255,0.3); transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .status {{ 
            margin-top: 2rem; padding: 1rem; background: rgba(0,255,0,0.2);
            border-radius: 8px; border-left: 4px solid #00ff00;
        }}
        .api-endpoints {{ 
            margin-top: 2rem; text-align: left; background: rgba(0,0,0,0.2);
            padding: 1rem; border-radius: 8px; font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ComfyAI Server</h1>
        <p>Unified inference server for vision models and face comparison</p>
        
        <div class="links">
            <a href="/manage/ui/" class="link">
                🌐 <strong>Model Management UI</strong><br>
                <small>Manage and download models</small>
            </a>
            <a href="/manage/ui/vision-test.html" class="link">
                🎯 <strong>Vision & Face Test UI</strong><br>
                <small>Drag & drop testing interface</small>
            </a>
            <a href="/docs" class="link">
                📚 <strong>API Documentation</strong><br>
                <small>Interactive OpenAPI docs</small>
            </a>
        </div>
        
        <div class="status">
            ✅ <strong>Server Status:</strong> Running and healthy
        </div>
        
        <div class="api-endpoints">
            <strong>Key API Endpoints:</strong><br>
            POST /v1/chat/completions (Vision + text)<br>
            POST /v1/image/compare_faces (Face comparison)<br>
            GET /v1/models (List available models)<br>
            GET /health (Server health check)
        </div>
    </div>
</body>
</html>
    """)

# Quick redirect for common paths
@app.get("/ui")
async def ui_redirect():
    """Redirect /ui to the main management UI."""
    return RedirectResponse(url="/manage/ui/")

@app.get("/test")
async def test_redirect():
    """Redirect /test to the vision test UI."""
    return RedirectResponse(url="/manage/ui/vision-test.html")

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
