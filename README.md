# **ComfyAI – LLM-Powered Vision & Text Query Nodes for ComfyUI**  

🚀 **ComfyAI** brings **multimodal AI capabilities** directly into your **ComfyUI workflows** with powerful custom nodes for text and vision inference using models like **Qwen-VL**, **Llava**, and **face recognition**.

Transform your ComfyUI experience with intelligent image analysis, face comparison, and conditional workflow control - all through simple, drag-and-drop nodes.

---

## ✨ **Custom Nodes**

### **🧠 AI Query Nodes**
- **VLLMTextQuery** – Send text prompts to any LLM endpoint
- **VLLMImageQuery** – Analyze single images with vision-language models  
- **VLLMDualImageQuery** – Compare and analyze two images simultaneously

### **👤 Face Analysis Nodes**
- **CompareFacesNode** – High-accuracy face comparison and similarity scoring
- Supports real photos, anime, and CG character styles

### **🎯 Workflow Control Nodes**  
- **ConditionalSaveImage** – Smart image saving based on boolean conditions
- Perfect for quality filtering and automated workflows

**All nodes work with any OpenAI-compatible API endpoint** – use local models, cloud services, or the included lightweight ONNX server.

---

## 🚀 **Quick Start**

### **Install Nodes**
```bash
cd your-comfyui-folder/custom_nodes
git clone https://github.com/mitchins/ComfyAI.git
pip install -r ComfyAI/requirements.txt
```

### **Basic Usage**
1. **Restart ComfyUI** to load the new nodes
2. **Add nodes** from the ComfyAI category in your workflow
3. **Configure endpoint** (OpenAI API, local server, or included ONNX server)
4. **Connect your images/text** and start creating!

### **Optional: Local ONNX Server**
For privacy and offline use, run the included lightweight server:
```bash
# Install server dependencies
pip install -r ComfyAI/apps/onnx_chat/requirements.txt
pip install -r ComfyAI/apps/face_api/requirements.txt

# Start servers
python -m ComfyAI.apps.onnx_chat.main    # Chat/vision models
PRESET=photo uvicorn ComfyAI.apps.face_api.main:app  # Face comparison
```

---

## 🎯 **Key Features**

### **🔗 Universal Compatibility**
- **Any OpenAI-compatible endpoint** (OpenAI, Anthropic, local servers)
- **Flexible model support** with automatic format detection
- **HTTP-based communication** for scalable, distributed inference

### **🎨 Vision-Language Intelligence**  
- **Image understanding** with natural language queries
- **Multi-image comparison** and analysis capabilities
- **Context-aware responses** based on visual content

### **👁️ Advanced Face Recognition**
- **High-precision face matching** across different styles
- **Specialized presets** for photos, anime, and CG characters  
- **Similarity scoring** for automated face-based workflows

### **⚡ Performance & Privacy**
- **Lightweight nodes** with minimal ComfyUI impact
- **Optional local inference** with included ONNX server
- **Efficient model management** with curated model catalog
- **GPU acceleration support** for fast inference

---

## 📖 **Documentation**

- **[Node Reference](nodes/README.md)** – Complete guide to all available nodes
- **[Server Setup](apps/README.md)** – Detailed server installation and configuration  
- **[Model Management](docs/UNIFIED_MODEL_MANAGEMENT.md)** – Managing local ONNX models

---

## 🛠️ **Supported Models**

### **Vision-Language Models**
- **Qwen2-VL** (2B parameters) – Strong multimodal understanding
- **Gemma-3n** – Advanced multimodal chat capabilities  
- **Phi-3.5-Vision** (4B parameters) – Efficient vision-text processing

### **Face Recognition Models**
- **ArcFace ResNet100** – High-accuracy face embedding
- **CLIP ViT** – Versatile vision model for anime/CG styles
- **InsightFace Detection** – Multiple detection model versions

All models support **multiple quantization levels** (FP16, Q4, INT8) for optimal performance vs. accuracy trade-offs.

---

## 🌟 **Why ComfyAI?**

### **🎯 Built for ComfyUI**
- **Native integration** with ComfyUI's node-based workflow system
- **Familiar interface** using standard ComfyUI patterns and conventions
- **Seamless workflows** connecting AI analysis with image processing

### **🔒 Privacy-First**
- **Local inference option** with included ONNX server
- **No cloud dependency** required for core functionality  
- **Your data stays local** when using the optional server components

### **⚡ Production Ready**
- **Comprehensive testing** with full test coverage
- **Error handling** and graceful degradation
- **Performance optimized** for real-world ComfyUI workflows
- **Active development** with regular updates and improvements

---

## 📦 **What's Included**

### **ComfyUI Nodes** (`nodes/`)
Ready-to-use custom nodes for your ComfyUI installation

### **Optional Server Stack** (`apps/`)
- **ONNX Chat Server** – Local vision-language model inference
- **Face API Server** – High-performance face comparison service
- **Unified Model Manager** – Web-based interface for managing local models

### **Documentation & Examples**
- Complete setup guides and API documentation
- Example workflows and use cases
- Troubleshooting and optimization tips

---

**Get started today and bring the power of modern AI directly into your ComfyUI workflows!** 🚀