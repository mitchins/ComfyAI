from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
import importlib
import pkgutil
from apps.shared.manage_cache import list_cached_entries
from apps.shared.model_types import ModelType
from common.logging import setup_logging

setup_logging()
app = FastAPI(
    title="ComfyAI Master API", 
    version="1.0.0",
    openapi_version="3.0.2",
    docs_url=None  # Disable default docs to use custom
)

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
            background: #f8f9fa;
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }}
        
        .main-container {{ 
            max-width: 1000px;
            margin: 2rem auto;
            padding: 0 2rem;
            text-align: center;
        }}
        
        h1 {{ 
            font-size: 2.5rem; 
            margin-bottom: 1rem; 
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{ 
            color: #666; 
            font-size: 1.2rem; 
            margin-bottom: 2rem; 
        }}
        
        .links {{ 
            display: grid; 
            gap: 1.5rem; 
            margin: 2rem 0; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }}
        
        .link {{ 
            display: block; 
            padding: 1.5rem 2rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px; 
            text-decoration: none; 
            color: white;
            transition: all 0.3s ease; 
            font-size: 1.1rem;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }}
        
        .link:hover {{ 
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }}
        
        .link strong {{ 
            display: block; 
            font-size: 1.3rem; 
            margin-bottom: 0.5rem; 
        }}
        
        .link small {{ 
            opacity: 0.9; 
            font-size: 1rem; 
        }}
        
        .status {{ 
            margin-top: 2rem; 
            padding: 1rem 0; 
            color: #155724;
            border-left: 4px solid #28a745;
            padding-left: 1rem;
            text-align: left;
        }}
        
        .api-endpoints {{ 
            margin-top: 2rem; 
            text-align: left; 
            color: #495057;
            padding: 1.5rem 0; 
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            border-top: 1px solid #e9ecef;
            padding-top: 1.5rem;
        }}
        
        .api-endpoints strong {{ 
            color: #333; 
            display: block; 
            margin-bottom: 0.5rem; 
        }}
        
        .endpoint {{ 
            margin: 0.5rem 0; 
            padding: 0.3rem 0; 
            color: #6f42c1; 
        }}
        
        @media (max-width: 768px) {{
            .main-container {{ 
                margin: 1rem; 
                padding: 1.5rem; 
            }}
            .links {{ 
                grid-template-columns: 1fr; 
            }}
        }}
    </style>
</head>
<body>
    <script src="/static/manage/navigation.js"></script>
    
    <div class="main-container">
        <h1>🤖 Server Status Dashboard</h1>
        <p class="subtitle">ComfyAI unified inference server monitoring and quick access</p>
        
        <!-- Server Status Section -->
        <div class="status">
            ✅ <strong>Server Status:</strong> Running and healthy
        </div>
        
        <!-- Quick Actions Section -->
        <h2 style="margin-top: 2rem; margin-bottom: 1rem; color: #333;">Quick Actions</h2>
        <div class="links">
            <a href="/manage/ui/" class="link">
                🌐 <strong>Model Management</strong>
                <small>Download and manage ONNX models</small>
            </a>
            <a href="/manage/ui/vision-test.html" class="link">
                🎯 <strong>Vision & Face Testing</strong>
                <small>Drag & drop testing interface</small>
            </a>
            <a href="/docs" class="link">
                📚 <strong>API Documentation</strong>
                <small>Interactive OpenAPI docs</small>
            </a>
        </div>
        
        <!-- API Information Section -->
        <h2 style="margin-top: 2rem; margin-bottom: 1rem; color: #333;">API Endpoints</h2>
        <div class="api-endpoints">
            <div class="endpoint"><strong>POST</strong> /v1/chat/completions <span style="color: #666;">• Vision + text inference</span></div>
            <div class="endpoint"><strong>POST</strong> /v1/image/compare_faces <span style="color: #666;">• Face comparison</span></div>
            <div class="endpoint"><strong>GET</strong> /v1/models <span style="color: #666;">• List available models</span></div>
            <div class="endpoint"><strong>GET</strong> /health <span style="color: #666;">• Server health check</span></div>
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

@app.get("/docs", response_class=HTMLResponse)
async def custom_swagger_ui_html():
    """Custom docs page with navigation."""
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>ComfyAI API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <style>
        body {{ margin: 0; padding: 0; }}
        .comfyai-nav {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .nav-container {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 1rem;
            height: 60px;
        }}
        .nav-brand .brand-link {{
            color: white;
            text-decoration: none;
            font-size: 1.5rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .nav-brand .brand-link:hover {{ color: rgba(255,255,255,0.9); }}
        .nav-links {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}
        .nav-link {{
            color: rgba(255,255,255,0.9);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.3s ease;
            font-size: 0.9rem;
            white-space: nowrap;
        }}
        .nav-link:hover {{
            background: rgba(255,255,255,0.1);
            color: white;
            transform: translateY(-1px);
        }}
        .nav-link.active {{
            background: rgba(255,255,255,0.2);
            color: white;
        }}
        #swagger-ui {{ padding-top: 0; }}
        .swagger-ui .topbar {{ display: none; }}
    </style>
</head>
<body>
    <nav class="comfyai-nav">
        <div class="nav-container">
            <div class="nav-brand">
                <a href="/" class="brand-link">🤖 ComfyAI</a>
            </div>
            <div class="nav-links">
                <a href="/" class="nav-link" title="Home">🏠 Home</a>
                <a href="/manage/ui/" class="nav-link" title="Model Management">🌐 Models</a>
                <a href="/manage/ui/vision-test.html" class="nav-link" title="Vision & Face Testing">🎯 Test</a>
                <a href="/docs" class="nav-link active" title="API Documentation">📚 Docs</a>
            </div>
        </div>
    </nav>
    
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script>
        const ui = SwaggerUIBundle({{
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.presets.standalone
            ],
            layout: "BaseLayout",
            deepLinking: true,
            showExtensions: true,
            showCommonExtensions: true,
            tryItOutEnabled: true
        }});
    </script>
</body>
</html>
    """)

app.mount(
    "/manage/ui",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="manage_ui",
)

# Mount static files for root access
app.mount(
    "/static/manage",
    StaticFiles(directory=STATIC_DIR),
    name="static_manage",
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
