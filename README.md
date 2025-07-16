# **ComfyAI – LLM-Powered Vision & Text Query Node for ComfyUI**  

🚀 **ComfyAI** is an advanced **LLM-powered query node** for **ComfyUI**, enabling both **text-based and vision-based inference** using multimodal models like **Qwen-VL** and **Llava**.  

This project exposes a lightweight HTTP API for text or vision models and can use any OpenAI-compatible endpoint, including the optional ONNX server. The ComfyUI node is fully decoupled from the LLM and communicates purely over HTTP.

---

## **✨ Features**  

ComfyAI ships with a set of custom nodes for ComfyUI:

- **VLLMTextQuery**, **VLLMImageQuery** and **VLLMDualImageQuery** – send text, one image or two images to your favourite LLM endpoint.
- **CompareFacesNode** – compare two face images using the optional face API.
- **ConditionalSaveImage** – only save results when a connected boolean evaluates to `True`.

All nodes communicate with any OpenAI‑compatible HTTP endpoint. Heavy model inference can run remotely, including via the lightweight ONNX server that comes with this repository.
See [nodes/README.md](nodes/README.md) for a full list of inputs and outputs.

---

## **📌 Supported Models**  

Currently supported models:  
- **Qwen-2.5VL** (`qwen2_5_vl`) – Strong multimodal (text+vision) model.  
- **Llava** (`llava`) – Vision-language AI for image understanding.  

✅ **Recommended Model:**  
- **Llava-7B (bnb4) from Unsloth** – **Tested & performs exceptionally well!**  
- **Qwen2.5-VL-3B-Instruct (bnb4)** – Good, but **Llava-7B handles instructions significantly better.**  
- Supports **BitsAndBytes 4bit/8bit quantization** for efficiency.  

🚀 **Planned Support:**  
- **mLLaMA & Pixtral** – Requires additional integration (not yet implemented).  

---

## **📥 Installation**  

### **🔧 Prerequisites**  
Ensure you have the following installed:  
- **Python 3.10+**  
- **PyTorch with CUDA** (`torch + torchvision`)  
- **Hugging Face Transformers** (`transformers`)  
- **ComfyUI** (installed separately)  

### **📌 Install ComfyAI (from your ComfyUI installation folder)**

Clone this repo inside ComfyUI's `custom_nodes/` folder (or copy just the
`nodes/` subfolder) and install the package:

```bash
cd custom_nodes
git clone https://github.com/mitchins/ComfyAI.git

# The plugin nodes live in the `nodes/` directory. You can clone the
# whole repository (as above) or simply copy that folder into a
# subdirectory, e.g. `custom_nodes/ComfyAI/`.
# Only the contents of `nodes/` are required for the ComfyUI plugin.

# Nodes only
pip install comfyai

# Optional ONNX server
pip install comfyai[onnx]
```

---

## **🚀 Usage**  

### **📌 Using the Query Node in ComfyUI**  

1. **Start ComfyUI** (ensure it’s installed and running).  
2. **Load the custom node from ComfyAI**.  
3. **Connect image/text inputs** and send queries.  
4. **Requests are sent to your configured API endpoint**.

---

### **📌 Use Case 1 - Single Image → Text Output**  

To **describe an image**, pass it as `sample`. The `reference` input is only used for comparisons.
If you provide **both** `sample` and `reference`, the node will send **two images at once** for vision models that support comparisons.

**Example Workflow:**  
![Single Image Example](Example01.png)  

📝 **Example Prompt:**  
> *"You are an interface for stable diffusion. Provide a prompt to generate an image like this one."*  

---

### **📌 Use Case 2 - Comparing Two Images (Boolean Output)**  

The **Vision LLM** can compare **two images** and **output a True/False result**.  

**Example Workflow:**  
![Image Comparison Example](Example02.png)  

📝 **Example Prompt:**  
> *"Answer yes or no, are the following two images similarly themed?"*  

💡 **Tip:** This library includes a **`ConditionalSaveImage` node**, which saves images **only when a connected boolean input is `True`**.

---

### **📌 Use Case 3 - AI-Generated Prompt from an Image**  

The **Vision LLM** can generate text prompts **based on an input image**, making it useful for **Stable Diffusion automation**.  

**Example Workflow:**  
![AI Generating Prompts](Example03.png)  

📝 **Example Prompt:**  
> *"Describe this image as a Stable Diffusion prompt."*  

**ComfyAI automatically writes a prompt**, which is then used to generate a similar image!  

---

### **📌 Use Case 4 - Combined Image Comparison + AI-Generated Prompt**  

This setup **first compares two images for similarity**, then **generates a Stable Diffusion prompt to recreate it**.  

**Example Workflow:**  
![AI Prompting AI](Example04.png)  

📝 **Example Prompt:**  
> *"Given the image provided, output the prompt for a Stable Diffusion image service to create one exactly like it. Ensure the style is the same. Be direct but ensure details are well-defined."*  

💡 **This is useful for**:  
- **Style transfer**  
- **Recreating an image in a different medium**  
- **Refining AI-generated art iteratively**  

---

## **🛠️ Configuration**  

### **🔍 Changing the Model**  
To use a different model, **select it inside the node in your ComfyUI workflow**.  

💡 **Example:**  
If you want to use a **Llava-7B model**, make sure it’s downloaded:  

```bash
huggingface-cli download unsloth/llava-1.5-7b-hf-bnb-4bit --all
```

Then, **select it inside the ComfyUI node settings**.

### 🛰️ Running the optional ONNX server
Use the built-in script if you want a lightweight OpenAI compatible endpoint.
Install the optional extra and run the server:

```bash
pip install comfyai[onnx]
comfyai-onnx-server  # or: python -m apps.onnx_chat.main
```

The client node accepts any OpenAI compatible endpoint URL, so you can point it
to this server, Ollama, or the official OpenAI API.

See [apps/README.md](apps/README.md) for details on the bundled servers.

---

## Installation & Runners

**Default** (remote OpenAI):

```bash
pip install comfyai
```

**Optional ONNX VLLM server**:

```bash
pip install comfyai[onnx]
```


### Environment Variables

- `COMFYAI_ENDPOINT` – base URL for API calls (default: `https://api.openai.com/v1`)
- `COMFYAI_ONNX_PORT` – local ONNX server port (default: `8000`)
- `ONNX_LOG_LEVEL` – uvicorn log level for the chat server (default: `info`)
- `LOG_LEVEL` – format-consistent Python logging level (default: `INFO`)
- `FACE_MODEL_PROVIDERS` – comma-separated ONNX providers (default: `CUDAExecutionProvider,CPUExecutionProvider`)
- `FACE_MODEL_NAME` – InsightFace model name (default: `buffalo_l`)
- `DETECTOR_MODEL` – HuggingFace repository for the face detector
- `DETECTOR_FILE` – path to the ONNX detector model file within the repo
- `EMBEDDER_MODEL_PATH` – repository for the embedding model
- `EMBEDDER_FILE` – path to the embedder ONNX file
- `DETECTOR_THRESHOLD` – optional threshold override (default per model)

### Face Comparison API Installation

```bash
pip install fastapi uvicorn insightface onnxruntime[-gpu] python-multipart
```

### Running the Face Comparison API (optional)

```bash
uvicorn apps.face_api.main:app --host 0.0.0.0 --port 7860
```

Set the environment variables above to tweak model selection or provider order.

### Example Usage

```python
from comfyai import openai_client

# Remote:
client = openai_client(api_key="…")

# Local ONNX:
client = openai_client(endpoint="http://localhost:8000/v1/chat/completions")

response = client.create(
model="qwen2.5",
messages=[{"role":"user","content":"Hello"}]
)
print(response)
```

### Testing

```bash
pip install -r requirements-dev.txt
pytest tests -q
pytest integration_tests -q
```

Integration tests expect a clone of the ComfyUI project under
`ComfyUI_repo/` (or otherwise available on `PYTHONPATH`). The CI workflow
runs both test suites sequentially. ONNX/Server tests are auto-skipped
unless you've installed the `onnx` extra.

---


## **📅 Roadmap**  

🚀 **Planned improvements:**  
- ✅ **Expanding model support** (mLLaMA, Pixtral, ONNX models like Phi-3.5 Vision).  
- ✅ **Adding API-based inference** (Ollama, OpenAI endpoints).  
- ✅ **Performance optimizations** to further reduce memory usage.  

---

## **📜 License**  

This project is licensed under the **AGPL-3.0 license**. See `LICENSE` for details.  

---

## **🚀 Stay Updated**  

⭐ **Star this repo** if you find it useful!  
📣 **Issues, feedback, and contributions are welcome.**  

Happy coding! 🎨🤖  

