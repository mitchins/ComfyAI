# ComfyAI Custom Nodes

This folder contains all custom nodes shipped with ComfyAI. These nodes bring multimodal AI capabilities directly into your ComfyUI workflows through simple, drag-and-drop interfaces.

## 🧠 AI Query Nodes

| Node | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| **`VLLMTextQuery`** | Send text prompts to any OpenAI-compatible endpoint | `prompt` (string), `endpoint` (string), `api_key` (string) | `response` (string), `boolean` (bool) |
| **`VLLMImageQuery`** | Analyze single images with vision-language models | `prompt` (string), `image` (IMAGE), `endpoint` (string), `api_key` (string) | `response` (string), `boolean` (bool) |
| **`VLLMDualImageQuery`** | Compare and analyze two images simultaneously | `prompt` (string), `image1` (IMAGE), `image2` (IMAGE), `endpoint` (string), `api_key` (string) | `response` (string), `boolean` (bool) |

## 👤 Face Analysis Nodes

| Node | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| **`CompareFacesNode`** | High-accuracy face comparison and similarity scoring | `image1` (IMAGE), `image2` (IMAGE), `endpoint` (string) | `similarity` (float), `is_same_person` (bool) |

## 🎯 Workflow Control Nodes

| Node | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| **`ConditionalSaveImage`** | Smart image saving based on boolean conditions | `images` (IMAGE), `condition` (bool), `filename_prefix` (string) | `images` (IMAGE) |

## 🔌 Compatibility

### API Endpoints
All AI query nodes work with **any OpenAI-compatible endpoint**:
- **OpenAI API** (`https://api.openai.com/v1`)
- **Local ONNX server** (`http://localhost:8080`) 
- **Anthropic Claude** via compatible proxy
- **Self-hosted models** (Ollama, vLLM, etc.)

### ComfyUI Integration
- **Standard interfaces** using ComfyUI's `INPUT_TYPES` and `RETURN_TYPES`
- **Automatic registration** when loaded into `custom_nodes/`
- **Error handling** with graceful fallbacks
- **Performance optimized** for real-time workflows

## 🚀 Quick Setup

1. **Copy nodes** to your ComfyUI installation:
   ```bash
   cp -r nodes/ /path/to/ComfyUI/custom_nodes/ComfyAI/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI** and find the nodes in the **ComfyAI** category

4. **Configure endpoints** in your workflow and start creating!

## 💡 Usage Examples

### Text Analysis
Connect `VLLMTextQuery` to analyze text content and get both the AI response and a boolean interpretation for workflow branching.

### Image Understanding  
Use `VLLMImageQuery` to describe images, detect objects, or answer questions about visual content.

### Image Comparison
Leverage `VLLMDualImageQuery` for before/after analysis, similarity detection, or multi-image reasoning tasks.

### Face Recognition
Implement `CompareFacesNode` for identity verification, duplicate detection, or character consistency checking.

### Conditional Workflows
Use `ConditionalSaveImage` to only save results that meet specific criteria, perfect for quality filtering pipelines.

---

**Ready to enhance your ComfyUI workflows with AI? These nodes make it simple!** 🎨
