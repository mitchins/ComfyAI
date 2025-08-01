# Unified Model Management System

The ComfyAI project includes a comprehensive unified model management system that provides a centralized interface for downloading, managing, and monitoring all ONNX models used across both chat and face servers.

## Overview

The unified model management system provides:

- **Centralized model discovery** - Automatically enumerates all supported ONNX models
- **Curated model options** - Pre-configured models rather than general repository browsing
- **Status tracking** - Real-time download status with quantization-level granularity  
- **Companion file enforcement** - Automatically includes required data files
- **Web-based management UI** - Clean, organized interface for model operations

## Architecture

### Components

1. **Unified Model Registry** (`apps/shared/unified_model_registry.py`)
   - Discovers and catalogs all supported ONNX models
   - Tracks download status and file associations
   - Handles companion file detection and enforcement

2. **Management API** (`apps/manage_api/router.py`)
   - RESTful endpoints for model operations
   - Supports quantization-specific downloads/deletions
   - Integrates with HuggingFace Hub for file management

3. **Web Interface** (`apps/static/manage/unified-manage.html`)
   - Collapsible model sections organized by server
   - Quantization status indicators (Available/Downloaded/X/Y Sizes)
   - File-level tracking for multi-component models

## Supported Models

### Chat Server Models
- **Gemma-3n-E2B-it-ONNX** - State-of-the-art multimodal model (vision+audio+text) with 4 quantization options
- **SmolVLM-256M-Instruct** - Ultra-lightweight vision+text model with 8 quantization options (experimental)

### Face Server Models
- **Face Detection Models** - Multiple versions (v1.2-v1.4) for real photos and anime
- **Face Embedding Models** - ArcFace ResNet100 and CLIP ViT Base
- **Specialized Models** - Real face detection, anime face detection, CG detection

## Usage

### Web Interface

1. Navigate to `http://localhost:8000/manage/ui/` 
2. View models organized by server (Chat/Face)
3. Click model headers to expand and see quantization options
4. Use Download/Clear buttons for individual quantizations or entire models
5. Monitor status indicators:
   - **NOT DOWNLOADED** (red) - No files downloaded
   - **X/Y SIZES** (blue) - X out of Y quantizations downloaded  
   - **DOWNLOADED** (green) - All quantizations downloaded

### API Endpoints

#### Get All Models
```http
GET /manage-api/unified-models
```

Returns array of all supported models with status, files, and quantizations.

#### Get Specific Model  
```http
GET /manage-api/unified-models/{model_id}
```

Returns detailed information for a specific model.

#### Download Model
```http
POST /manage-api/unified-models/download
Content-Type: application/json

{
  "model_id": "chat-gemma-3n-e2b-it-onnx",
  "quantization": "FP16"  // Optional: download specific quantization
}
```

Downloads all files for a model or specific quantization.

#### Delete Model
```http
DELETE /manage-api/unified-models/{model_id}?quantization=FP16
```

Deletes all files for a model or specific quantization.

## Model Status Logic

### Quantization Counting
Models with multiple quantizations show status as:
- **"X/Y SIZES"** where X = downloaded quantizations, Y = total available
- A quantization is "downloaded" only when ALL required files are present
- Includes main ONNX files and companion data files

### Companion File Enforcement
The system automatically detects and enforces companion files:
- **Pattern 1**: `filename.onnx_data` (e.g., `decoder_model_merged_fp16.onnx_data`)
- **Pattern 2**: `filename_data.onnx` (e.g., `decoder_model_merged_fp16_data.onnx`)  
- **Pattern 3**: `filename_data_N.onnx` (numbered variants)

Only files that actually exist in the repository are included in the requirements.

## Configuration

### Model Discovery
Models are automatically discovered from:
- `MODEL_QUANT_CONFIGS` in `apps/shared/model_types.py` (chat models)
- Face API presets in `apps/face_api/presets.py` (face models)
- InsightFace model configurations (face detection)

### Adding New Models
To add support for new models:

1. **Chat Models**: Add configuration to `MODEL_QUANT_CONFIGS`
2. **Face Models**: Add to face detection/embedding dictionaries in registry
3. **Repository Mapping**: Update `_get_repo_id_for_model()` method

### Quantization Mapping
Quantizations are mapped to files using patterns:
- `FP16`: Files containing `_fp16` or `_16`
- `Q4`: Files containing `_q4`  
- `FP32`: Files without specific quantization suffixes
- `INT8`: Files containing `_int8`
- etc.

## Testing

Comprehensive test coverage includes:
- Model discovery and registration (`tests/test_unified_model_registry.py`)
- API endpoint functionality (`tests/test_unified_manage_api.py`)
- Companion file detection and status logic
- Error handling and edge cases

Run tests with:
```bash
python -m pytest tests/test_unified_model_registry.py -v
python -m pytest tests/test_unified_manage_api.py -v
```

## Migration from Legacy System

The unified system replaces the previous general cache manager:
- `/manage/ui/` now redirects to the new unified interface
- Old management JavaScript has been removed
- Legacy API endpoints remain for backward compatibility
- All models are now pre-curated rather than general repository browsing

## Benefits

1. **User Experience**: Clear, organized interface showing exactly what's available
2. **Reliability**: Enforces all required files for complete model installations  
3. **Efficiency**: Quantization-level granularity reduces unnecessary downloads
4. **Maintainability**: Centralized model definitions and automatic discovery
5. **Extensibility**: Easy to add new models and quantization options