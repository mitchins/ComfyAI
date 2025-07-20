# ONNX Chat Reference Server

A reference implementation of ONNX inference for language models, providing tested configurations for known working models.

## 🎯 Reference Server Philosophy

This is a **reference server** that provides verified, working configurations rather than attempting universal ONNX support. We focus on:

- ✅ **Tested models** with verified quantizations
- ✅ **Known working setups** to avoid compatibility issues  
- ✅ **Clear documentation** of what works and what doesn't
- ✅ **Simple, reliable** model management

## 📋 Supported Reference Models

| Model | Repository | Architecture | Features | Recommended |
|-------|------------|--------------|----------|-------------|
| **Qwen2-VL-2B** | `onnx-community/Qwen2-VL-2B-Instruct` | Multi-component | Vision + Text | ✅ `_q4` quant |
| **Qwen2-VL-7B** | `onnx-community/Qwen2-VL-7B-Instruct` | Multi-component | Vision + Text | ✅ `_q4` quant |
| **Granite 3.0 2B** | `onnx-community/granite-3.0-2b-instruct` | Single file | Text only | ✅ `_q4` quant |

### 🚧 Future Roadmap
| **Gemma 3n 2B** | `onnx-community/gemma-3n-E2B-it-ONNX` | Multi-component | Vision + Audio + Text | 🔨 Under development |

### Architecture Support

#### Multi-Component Models (Qwen2-VL)
- **Components**: `embed_tokens.onnx`, `decoder_model_merged.onnx`, `vision_encoder.onnx`
- **Features**: KV caching, 3D position embeddings, incremental generation
- **Quantization**: Supports `_q4`, `_fp16`, `_int8`, `_uint8`, `_bnb4`, `_quantized` variants

#### Single Model Files (Granite)
- **Components**: `model.onnx` (with optional `.onnx_data`)
- **Features**: Direct inference, standard generation
- **Quantization**: Supports various precision variants

### API Compatibility

#### OpenAI-Compatible Endpoints
- `POST /v1/chat/completions` - Chat completion endpoint
- `GET /health` - Health check

#### Request Format
```json
{
  "model": "onnx-community/Qwen2-VL-2B-Instruct:decoder_model_merged_q4.onnx",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_tokens": 100,
  "images": ["base64_encoded_image"]  // Optional, vision models only
}
```

## Installation

```bash
pip install -r apps/onnx_chat/requirements.txt
```

## Usage

```bash
python -m apps.onnx_chat.main
```

### Model Specification

Models can be specified in several ways:

1. **Repository only**: `onnx-community/Qwen2-VL-2B-Instruct` (uses default components)
2. **With quantization**: `onnx-community/Qwen2-VL-2B-Instruct:decoder_model_merged_q4.onnx`
3. **Full path**: `onnx-community/Qwen2-VL-2B-Instruct/onnx/decoder_model_merged_fp16.onnx`

### Quantization Options

| Suffix | Description | Quality | Size | Speed |
|--------|-------------|---------|------|-------|
| (none) | FP32 | Highest | Largest | Slowest |
| `_fp16` | FP16 | High | Medium | Medium |
| `_q4` | 4-bit quantized | Good | Small | Fast |
| `_q4f16` | Mixed 4-bit/FP16 | Good | Small | Fast |
| `_int8` | 8-bit integer | Medium | Small | Fast |
| `_uint8` | 8-bit unsigned | Medium | Small | Fast |
| `_bnb4` | BitsAndBytes 4-bit | Good | Smallest | Fastest |

### Examples

#### Basic Text Generation
```bash
curl -X POST "http://localhost:7860/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onnx-community/Qwen2-VL-2B-Instruct",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 150
  }'
```

#### Vision + Text (Qwen2-VL only)
```bash
curl -X POST "http://localhost:7860/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onnx-community/Qwen2-VL-2B-Instruct:decoder_model_merged_q4.onnx",
    "messages": [{"role": "user", "content": "What do you see in this image?"}],
    "images": ["'$(base64 -i image.jpg)'"],
    "max_tokens": 100
  }'
```

## Configuration

- `ONNX_MODEL_PATH` – path to an ONNX model to load (optional)
- `ONNX_FILE_NAME` – default filename when downloading from Hugging Face
- `COMFYAI_ONNX_PORT` – listening port (default `8000`)
- `ONNX_LOG_LEVEL` – uvicorn log level (default `info`)
- `LOG_LEVEL` – Python logging level

## Limitations

### Current Limitations

1. **Vision Support**: Only implemented for Qwen2-VL models
2. **Batch Size**: Fixed at 1 (single request processing)
3. **Model Discovery**: Requires explicit configuration for new models
4. **Audio**: Gemma 3n audio components not yet implemented

### Model-Specific Limitations

#### Qwen2-VL
- ✅ Text generation
- ✅ Vision + text
- ❌ Streaming responses
- ❌ Function calling

#### Granite 3.0 2B
- ✅ Basic text generation
- ❌ Advanced features (position embeddings, KV cache)
- ❌ Vision support

#### Gemma 3n 2B  
- 🔧 Text generation (basic)
- 🔧 Vision support (placeholder)
- ❌ Audio support
- ❌ Multimodal integration

## Testing

```bash
# Unit tests
pytest tests/test_onnx_inference.py -v

# Integration tests  
pytest tests/test_onnx_inference.py::TestONNXChatIntegration -v
```
