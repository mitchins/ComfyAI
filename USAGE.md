# ComfyAI Usage Guide

ComfyAI provides different installation options depending on your use case:

## 🖥️ Client-Side: ComfyUI Nodes

For running nodes within ComfyUI (minimal dependencies):

```bash
# Install just what's needed for ComfyUI nodes
pip install -e .[client]

# Or manually:
pip install numpy>=1.23 pillow>=9.0 requests>=2.28
```

**Use case**: You have ComfyUI installed and want to use the ComfyAI nodes for face comparison, OpenAI client calls, etc.

## 🚀 Server Deployment: Inference Server

For deploying the full inference server (the main use case):

```bash
# Option 1: Docker (recommended)
./deploy.sh build && ./deploy.sh run

# Option 2: Direct installation
pip install -r requirements.txt
uvicorn apps.main:app --host 0.0.0.0 --port 8000

# Option 3: With pip extras
pip install -e .[server]
```

**Use case**: You want to deploy a standalone inference server that provides:
- **Face comparison API** (key differentiator - no standard solution exists)
- **Vision-language model inference** (SmolVLM, Qwen2-VL, Gemma-3n, Phi-3.5)
- **Model management UI**
- **Drag & drop testing interface**

## 🛠️ Development: Testing & CI/CD

For development, testing, and future browser automation:

```bash
# Install with development dependencies  
pip install -r requirements-dev.txt

# Or with pip extras
pip install -e .[dev]

# Run tests
pytest tests/ integration_tests/

# Future: Browser automation
# pip install playwright>=1.30.0
```

**Use case**: Contributing to the project, running tests, CI/CD pipelines, future E2E testing with Playwright.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client-Side   │    │ Server Deploy   │    │  Development    │
│  (ComfyUI)      │    │ (Docker/Prod)   │    │  (Testing)      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • numpy         │    │ • fastapi       │    │ • pytest       │
│ • pillow        │    │ • uvicorn       │    │ • httpx         │
│ • requests      │    │ • onnxruntime   │    │ • [playwright]  │
│                 │    │ • transformers  │    │ • [black/ruff]  │
│                 │    │ • insightface   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                        │                        │
       │                        │                        │
       v                        v                        v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ComfyUI Nodes   │    │ Inference Server│    │  CI/CD & Tests  │
│ • compare_faces │    │ • Face API      │    │ • Unit tests    │
│ • openai_client │    │ • Vision models │    │ • Integration   │
│ • string_utils  │    │ • Model mgmt    │    │ • [E2E tests]   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 💡 Key Insights

1. **Face comparison is the killer feature** - no standard solution exists, making this a unique value proposition

2. **ONNX vision models are value-add** - they provide additional inference capabilities on top of the core face comparison

3. **Server deployment is the main use case** - most users will want the full inference server rather than just client nodes

4. **Three clear dependency tiers** match the three main usage patterns: client/server/dev

This structure makes it easy to:
- Install minimal deps for ComfyUI integration
- Deploy a full-featured inference server 
- Set up development environment with testing tools
- Future-proof for browser automation and additional dev tools