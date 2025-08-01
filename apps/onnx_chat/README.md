# ONNX Chat Reference Server

A high-performance ONNX inference server for multimodal language models, providing OpenAI-compatible API endpoints with support for vision, text, and audio modalities.

## 🎯 Key Features

- **State-of-the-art multimodal inference** with Gemma-3n
- **OpenAI-compatible API** for easy integration
- **Transformers + ONNXRuntime** architecture (no PyTorch dependency)
- **Multiple image support** for advanced vision workflows
- **Efficient model management** with automatic component discovery

## 📋 Supported Models

### Production Ready

| Model | Repository | Parameters | Features | Status |
|-------|------------|------------|----------|--------|
| **Gemma-3n** | `onnx-community/gemma-3n-E2B-it-ONNX` | 2B | Vision + Audio + Text | ✅ **SOTA** |

### Experimental

| Model | Repository | Parameters | Features | Status |
|-------|------------|------------|----------|--------|
| **SmolVLM** | `HuggingFaceTB/SmolVLM-256M-Instruct` | 256M | Vision + Text | 🧪 Future |
| **Granite Vision** | `ibm-granite/granite-vision-3.2-2b` | 3.2B | Vision + Text | 🧪 Testing |

## 🚀 Quick Start

### Installation

```bash
pip install -r apps/onnx_chat/requirements.txt
```

### Running the Server

```bash
# Start the server
python -m apps.onnx_chat.main

# With custom port
COMFYAI_ONNX_PORT=8002 python -m apps.onnx_chat.main

# Access at: http://localhost:8000/
```

## 📡 API Usage

### Basic Text Generation

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 150
  }'
```

### Vision Analysis (Single Image)

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
    "messages": [{"role": "user", "content": "What is shown in the provided image?"}],
    "images": ["'$(base64 -i image.jpg)'"],
    "max_tokens": 200
  }'
```

### Multiple Image Analysis

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
    "messages": [{"role": "user", "content": "What do you see in these images? Describe each one."}],
    "images": ["'$(base64 -i image1.jpg)'", "'$(base64 -i image2.jpg)'"],
    "max_tokens": 300
  }'
```

## 🔧 Model Specification

### Gemma-3n Quantization Options

| Quantization | Description | Quality | Size | Performance |
|--------------|-------------|---------|------|-------------|
| `Q4_MIXED` | Mixed 4-bit | Excellent | ~1GB | Fast (Recommended) |
| `QUANTIZED` | Default quantized | Good | ~1.2GB | Good |
| `FP16` | Half precision | Better | ~2GB | Medium |
| `FP32` | Full precision | Best | ~4GB | Slower |

### Model Format Examples

```bash
# With specific quantization
"Gemma-3n-E2B-it-ONNX/Q4_MIXED"

# Default quantization
"Gemma-3n-E2B-it-ONNX"

# Full repository path
"onnx-community/gemma-3n-E2B-it-ONNX"
```

## 🏗️ Architecture

### Gemma-3n Components
- **embed_tokens**: Token embeddings with per-layer inputs
- **vision_encoder**: Processes visual inputs (768x768 images)
- **audio_encoder**: Processes audio inputs (future)
- **decoder_model_merged**: Main transformer decoder with KV cache

### Technical Details
- **AutoProcessor Integration**: Uses HuggingFace transformers for proper multimodal preprocessing
- **Vision Feature Injection**: 832 image tokens replaced with vision encoder outputs
- **KV Cache Management**: Efficient incremental generation
- **Position Embeddings**: Cumulative attention-based positioning

## 📊 Performance

### Gemma-3n Benchmarks
- **First token latency**: ~2-3s (including image processing)
- **Generation speed**: ~15-20 tokens/second on CPU
- **Memory usage**: ~4-5GB with Q4_MIXED quantization
- **Multiple images**: Linear scaling with image count

## 🔍 Troubleshooting

### Common Issues

1. **Model download fails**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/hub/models--onnx-community--gemma-3n-E2B-it-ONNX
   ```

2. **Out of memory**
   - Use Q4_MIXED quantization (recommended)
   - Reduce max_tokens parameter
   - Close other applications

3. **Slow generation**
   - Ensure you're using quantized models
   - Check CPU usage - the server uses all available cores
   - Consider GPU acceleration (future feature)

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_onnx_inference.py -v

# Test Gemma-3n specifically
pytest tests/test_onnx_inference.py -k "gemma" -v

# Integration tests
python integration_tests/test_gemma3n_integration.py
```

## 🚧 Limitations

### Current Limitations
- **Batch size**: Fixed at 1 (single request processing)
- **Audio support**: Not yet implemented for Gemma-3n
- **Streaming**: Not supported (full generation only)
- **GPU acceleration**: CPU-only for now

### Model-Specific Notes

#### Gemma-3n
- ✅ Text generation
- ✅ Single image analysis
- ✅ Multiple image analysis
- ✅ Long-form responses
- ⏳ Audio support (model capable, implementation pending)

#### SmolVLM (Experimental)
- 🧪 Ultra-lightweight (256MB)
- 🧪 Basic vision support
- ❌ Currently has generation issues

## 📈 Roadmap

1. **Audio support** for Gemma-3n multimodal capabilities
2. **GPU acceleration** with ONNX Runtime GPU providers
3. **Streaming responses** for real-time generation
4. **SmolVLM fixes** for edge deployment scenarios
5. **Batch processing** for multiple concurrent requests

## 🤝 Contributing

When adding new models:
1. Add to `model_types.py` with proper configuration
2. Implement any model-specific generation logic
3. Add comprehensive tests
4. Update this documentation

## 📝 License

Part of the ComfyAI project. See main repository for license details.