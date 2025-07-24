# ComfyAI Server Stack

The `apps/` directory contains optional HTTP services that power ComfyAI's local inference capabilities. These servers provide privacy-focused, offline alternatives to cloud-based AI services.

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r onnx_chat/requirements.txt
pip install -r face_api/requirements.txt
```

### Run Individual Services
```bash
# ONNX Chat Server (vision-language models)
python -m apps.onnx_chat.main

# Face Comparison API  
PRESET=photo uvicorn apps.face_api.main:app --port 7860

# Unified Server (all services + model management)
uvicorn apps.main:app --host 0.0.0.0 --port 8000
```

---

## 🧠 ONNX Chat Server

**Local vision-language model inference** with OpenAI-compatible API.

### Features
- **OpenAI API compatibility** – drop-in replacement for remote endpoints
- **Multimodal support** – text, vision, and multi-image analysis
- **Multiple quantizations** – FP16, Q4, INT8 for performance tuning
- **Dynamic model loading** – automatic model discovery and caching

### Supported Models
- **Qwen2-VL-2B-Instruct** – 8 quantization variants
- **Gemma-3n-E2B-it-ONNX** – 3 quantization variants  
- **Phi-3.5-vision-instruct** – 2 quantization variants

### Configuration
```bash
# Basic usage
python -m apps.onnx_chat.main

# Custom configuration
MODEL_NAME="Qwen2-VL-2B-Instruct/Q4" python -m apps.onnx_chat.main
ONNX_CHAT_HOST=0.0.0.0 ONNX_CHAT_PORT=8080 python -m apps.onnx_chat.main
```

**📖 Detailed Documentation:** [onnx_chat/README.md](onnx_chat/README.md)

---

## 👤 Face Comparison API

**High-accuracy face recognition and similarity scoring** across different image styles.

### Features
- **Multi-style support** – real photos, anime, and CG characters
- **Preset configurations** – optimized for different use cases
- **ONNX-powered inference** – fast, efficient processing
- **Batch processing** – handle multiple comparisons

### Presets
- **`photo`** – Real photographs (default preset)
- **`anime`** – Anime-style characters and artwork
- **`cg`** – Computer-generated and digital art

### Supported Models
- **Face Detection**: InsightFace v1.2-v1.4 (multiple variants)
- **Face Embedding**: ArcFace ResNet100, CLIP ViT Base
- **Specialized**: Real face detection, anime face detection

### Usage
```bash
# Photo preset (default)
PRESET=photo uvicorn apps.face_api.main:app

# Anime preset  
PRESET=anime uvicorn apps.face_api.main:app --port 7860

# Custom configuration
DETECTOR_MODEL=deepghs/anime_face_detection \
EMBEDDER_MODEL_PATH=Xenova/clip-vit-base-patch32 \
uvicorn apps.face_api.main:app
```

**📖 Detailed Documentation:** [face_api/README.md](face_api/README.md)

---

## 🔧 Unified Server & Model Management

**All-in-one server** combining chat, face, and model management APIs.

### Features
- **Unified endpoint** – single server for all ComfyAI services
- **Model management UI** – web-based interface at `/manage/ui/`
- **Curated model catalog** – pre-configured ONNX models
- **Quantization control** – download specific model variants
- **Status tracking** – real-time download progress and file verification

### Model Management Features
- **13 total models** (3 chat + 10 face models)  
- **Companion file enforcement** – automatic detection of required data files
- **Quantization-level granularity** – "X/Y SIZES" status indicators
- **Organized interface** – collapsible sections by server type
- **Bulk operations** – download/clear entire model configurations

### Usage
```bash
# Start unified server
uvicorn apps.main:app --host 0.0.0.0 --port 8000

# Access services:
# - Chat API: http://localhost:8000/onnx-chat/
# - Face API: http://localhost:8000/face-server/  
# - Model Management: http://localhost:8000/manage/ui/
# - API Documentation: http://localhost:8000/docs
```

**📖 Model Management Guide:** [../docs/UNIFIED_MODEL_MANAGEMENT.md](../docs/UNIFIED_MODEL_MANAGEMENT.md)

---

## 🏗️ Architecture

### Service Structure
```
apps/
├── main.py                 # Unified server entry point
├── onnx_chat/             # Vision-language model server
│   ├── main.py            # Chat server entry point  
│   ├── requirements.txt   # Chat dependencies
│   └── README.md          # Detailed chat documentation
├── face_api/              # Face comparison service
│   ├── main.py            # Face server entry point
│   ├── requirements.txt   # Face dependencies  
│   ├── presets.py         # Model presets configuration
│   └── README.md          # Detailed face documentation
├── manage_api/            # Model management API
│   └── router.py          # Management endpoints
├── shared/                # Shared utilities
│   ├── unified_model_registry.py  # Model discovery and tracking
│   ├── model_types.py     # Model configurations
│   └── manage_cache.py    # HuggingFace cache utilities
└── static/                # Web interface assets
    └── manage/            # Model management UI
```

### Communication Flow
```
ComfyUI Nodes → HTTP API → Local ONNX Server → Models
     ↓              ↓            ↓
Web Interface → Management API → Model Registry
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# ONNX Chat Server
MODEL_NAME="Qwen2-VL-2B-Instruct/Q4"    # Default model
ONNX_CHAT_HOST="0.0.0.0"                # Server host
ONNX_CHAT_PORT="8080"                   # Server port

# Face API Server  
PRESET="photo"                          # Model preset
DETECTOR_MODEL="deepghs/real_face_detection"
EMBEDDER_MODEL_PATH="openailab/onnx-arcface-resnet100-ms1m"

# Model Management
PRELOAD_MODELS="1"                      # Load models at startup
FACE_MODEL_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
```

### Model Storage
- **HuggingFace Cache**: `~/.cache/huggingface/hub/`
- **Auto-discovery**: Models automatically detected from cache
- **Quantization support**: Multiple variants per model family

---

## 🔒 Privacy & Security

### Local-First Design
- **No cloud dependency** for core functionality
- **All inference runs locally** on your hardware
- **Model data stays local** with HuggingFace cache

### Security Features  
- **Input validation** on all API endpoints
- **Path traversal protection** for file operations
- **Resource limits** to prevent abuse
- **Error handling** without exposing internal details

---

## 🚀 Performance

### Optimization Features
- **ONNX runtime** for efficient inference
- **Multiple quantization levels** (FP32 → FP16 → Q4 → INT8)
- **GPU acceleration** when available (CUDA, CoreML)
- **Model caching** for faster subsequent loads
- **Companion file management** for large models

### Resource Requirements
- **Minimum**: 8GB RAM, CPU inference
- **Recommended**: 16GB+ RAM, dedicated GPU
- **Storage**: 2-10GB per model (varies by quantization)

---

**Ready to deploy your own AI infrastructure? Start with the unified server and scale as needed!** 🚀