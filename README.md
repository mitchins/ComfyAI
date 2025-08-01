# **ComfyAI – LLM-Powered Vision & Text Query Nodes for ComfyUI**  

🚀 **ComfyAI** brings **multimodal AI capabilities** directly into your **ComfyUI workflows** with powerful custom nodes for text and vision inference using state-of-the-art models like **Gemma-3n** and advanced **face recognition**.<br>
<br>
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

**All nodes work with any OpenAI-compatible API endpoint** – use local models, cloud services, or the **separate ImageAIServer** for local inference.

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
3. **Configure endpoint** (OpenAI API, local server, or the ImageAIServer)
4. **Connect your images/text** and start creating!

### **Optional: Local Inference with ImageAIServer**
For privacy and offline use, you can run the **ImageAIServer** as a separate service. This server provides the backend for VLM and face detection features.

1.  **Clone the ImageAIServer repository:**
    ```bash
    git clone https://github.com/mitchins/ImageAIServer.git
    cd ImageAIServer
    ```
2.  **Install server dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the server (e.g., using Docker or directly):**
    ```bash
    # Using Docker (recommended for ease of setup)
    docker-compose up --build -d

    # Or directly (requires manual setup of models)
    # uvicorn apps.main:app --host 0.0.0.0 --port 8000
    ```
4.  **Access the server:**
    *   **API Endpoint:** `http://localhost:8000/`
    *   **Management UI:** `http://localhost:8000/ui`  
    *   **Vision Testing:** `http://localhost:8000/test`

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
- **Optional local inference** with ImageAIServer
- **Efficient model management** with curated model catalog
- **GPU acceleration support** for fast inference

---

## 📖 **Documentation**

- **[Node Reference](nodes/README.md)** – Complete guide to all available nodes
- **[ImageAIServer Documentation](https://github.com/mitchins/ImageAIServer/blob/main/README.md)** – Detailed server installation and configuration  
- **[Model Management](docs/UNIFIED_MODEL_MANAGEMENT.md)** – Managing local ONNX models

---

## 🛠️ **Supported Models**

### **Vision-Language Models**
- **Gemma-3n** (2B parameters) – State-of-the-art multimodal model with vision, audio, and text capabilities
- **SmolVLM** (256M parameters) – Ultra-lightweight vision model (experimental)

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
- **Local inference option** with ImageAIServer
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

### **ImageAIServer (Separate Project)**
- **ONNX Chat Server** – Local vision-language model inference
- **Face API Server** – High-performance face comparison service
- **Unified Model Manager** – Web-based interface for managing local models

### **Documentation & Examples**
- Complete setup guides and API documentation
- Example workflows and use cases
- Troubleshooting and optimization tips

---

**Get started today and bring the power of modern AI directly into your ComfyUI workflows!** 🚀