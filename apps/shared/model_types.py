from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np


class ModelType(str, Enum):
    """Enum for different model types based on their capabilities."""
    
    TEXT_LLM = "text-llm"
    VISION_LLM = "vision-llm"  
    VISION_EMBEDDER = "vision-embedder"
    TEXT_EMBEDDER = "text-embedder"
    UNKNOWN = "unknown"

    @classmethod
    def chat_compatible_types(cls):
        """Return model types that are compatible with chat endpoints."""
        return [cls.TEXT_LLM, cls.VISION_LLM]

    @classmethod
    def vision_compatible_types(cls):
        """Return model types that are compatible with vision endpoints."""
        return [cls.VISION_LLM, cls.VISION_EMBEDDER]


class ReferenceModel(str, Enum):
    """Curated list of supported ONNX models."""
    
    # Gemma-3n models - Vision + Text + Audio (SOTA)
    GEMMA_3N_E2B = "gemma-3n-e2b"
    
    # SmolVLM models - Ultra-lightweight vision + text (Future)
    SMOLVLM_256M = "smolvlm-256m"
    
    # Granite models - Vision + Text (NOT just text-only granite)
    GRANITE_VISION_3_2B = "granite-vision-3-2b"


class Quantization(str, Enum):
    """Supported quantization levels."""
    
    Q4 = "q4"           # 4-bit quantized
    Q4_F16 = "q4f16"    # 4-bit quantized with FP16 weights
    BNB4 = "bnb4"       # BitsAndBytes 4-bit
    FP16 = "fp16"       # 16-bit floating point
    INT8 = "int8"       # 8-bit integer
    UINT8 = "uint8"     # 8-bit unsigned
    QUANTIZED = "quantized"  # Generic quantized
    FULL = "full"       # Full precision FP32


@dataclass
class ONNXModelConfig:
    """Configuration for an ONNX model."""
    num_layers: int
    num_heads: int
    head_dim: int
    has_vision: bool = False
    position_dims: int = 1  # 1 for standard, 3 for vision models like Qwen2-VL
    subfolder: str = "onnx"
    components: Dict[str, str] = None  # Maps component name to filename pattern
    num_kv_heads: Optional[int] = None  # For grouped query attention, None means same as num_heads
    
    def __post_init__(self):
        if self.components is None:
            self.components = {
                'embed': 'embed_tokens{suffix}.onnx',
                'decoder': 'decoder_model_merged{suffix}.onnx'
            }
            if self.has_vision:
                self.components['vision'] = 'vision_encoder{suffix}.onnx'


# Curated reference model specifications with verified working quantizations
@dataclass
class ModelSpec:
    """Specification for a curated reference model."""
    repo_id: str
    config: ONNXModelConfig
    supported_quants: List[Quantization]
    default_quant: Quantization
    description: str


# Optimal "Model/Quant" → component mappings (only quants fully supported by all submodels)
MODEL_QUANT_CONFIGS = {
    # Qwen2-VL-2B-Instruct supports all eight quant types across decoder, embed & vision
    "Qwen2-VL-2B-Instruct/FP32": {
        "decoder":         "onnx/decoder_model_merged.onnx",
        "embed_tokens":    "onnx/embed_tokens.onnx",
        "vision_encoder":  "onnx/vision_encoder.onnx",
    },
    "Qwen2-VL-2B-Instruct/FP16": {
        "decoder":         "onnx/decoder_model_merged_fp16.onnx",
        "embed_tokens":    "onnx/embed_tokens_fp16.onnx",
        "vision_encoder":  "onnx/vision_encoder_fp16.onnx",
    },
    "Qwen2-VL-2B-Instruct/INT8": {
        "decoder":         "onnx/decoder_model_merged_int8.onnx",
        "embed_tokens":    "onnx/embed_tokens_int8.onnx",
        "vision_encoder":  "onnx/vision_encoder_int8.onnx",
    },
    "Qwen2-VL-2B-Instruct/Q4": {
        "decoder":         "onnx/decoder_model_merged_q4.onnx",
        "embed_tokens":    "onnx/embed_tokens_q4.onnx",
        "vision_encoder":  "onnx/vision_encoder_q4.onnx",
    },
    "Qwen2-VL-2B-Instruct/Q4_F16": {
        "decoder":         "onnx/decoder_model_merged_q4f16.onnx",
        "embed_tokens":    "onnx/embed_tokens_q4f16.onnx",
        "vision_encoder":  "onnx/vision_encoder_q4f16.onnx",
    },
    "Qwen2-VL-2B-Instruct/BNB4": {
        "decoder":         "onnx/decoder_model_merged_bnb4.onnx",
        "embed_tokens":    "onnx/embed_tokens_bnb4.onnx",
        "vision_encoder":  "onnx/vision_encoder_bnb4.onnx",
    },
    "Qwen2-VL-2B-Instruct/QUANTIZED": {
        "decoder":         "onnx/decoder_model_merged_quantized.onnx",
        "embed_tokens":    "onnx/embed_tokens_quantized.onnx",
        "vision_encoder":  "onnx/vision_encoder_quantized.onnx",
    },
    "Qwen2-VL-2B-Instruct/UINT8": {
        "decoder":         "onnx/decoder_model_merged_uint8.onnx",
        "embed_tokens":    "onnx/embed_tokens_uint8.onnx",
        "vision_encoder":  "onnx/vision_encoder_uint8.onnx",
    },

    # Gemma-3n-E2B-it-ONNX configurations (matching test_gemma3n.py)
    "Gemma-3n-E2B-it-ONNX/Q4_MIXED": {
        "audio_encoder":   "onnx/audio_encoder_q4.onnx",
        "decoder":         "onnx/decoder_model_merged_q4.onnx",
        "embed_tokens":    "onnx/embed_tokens_quantized.onnx",
        "vision_encoder":  "onnx/vision_encoder_quantized.onnx",
    },
    "Gemma-3n-E2B-it-ONNX/FP16": {
        "audio_encoder":   "onnx/audio_encoder_fp16.onnx",
        "decoder":         "onnx/decoder_model_merged_fp16.onnx",
        "embed_tokens":    "onnx/embed_tokens_fp16.onnx",
        "vision_encoder":  "onnx/vision_encoder_fp16.onnx",
    },
    "Gemma-3n-E2B-it-ONNX/FP32": {
        "audio_encoder":   "onnx/audio_encoder.onnx",
        "decoder":         "onnx/decoder_model_merged.onnx",
        "embed_tokens":    "onnx/embed_tokens.onnx",
        "vision_encoder":  "onnx/vision_encoder.onnx",
    },

    # Phi-3.5-vision-instruct only Q4 & Q4_F16
    "Phi-3.5-vision-instruct/Q4": {
        "prepare_inputs_embeds": "onnx/prepare_inputs_embeds_q4.onnx",
        "decoder":               "onnx/model_q4.onnx",
        "vision_encoder":        "onnx/vision_encoder_q4.onnx",
    },
    "Phi-3.5-vision-instruct/Q4_F16": {
        "prepare_inputs_embeds": "onnx/prepare_inputs_embeds_q4f16.onnx",
        "decoder":               "onnx/model_q4f16.onnx",
        "vision_encoder":        "onnx/vision_encoder_q4f16.onnx",
    },
    
    # SmolVLM-256M-Instruct - Ultra-lightweight with all quantization options
    "SmolVLM-256M-Instruct/FP32": {
        "embed_tokens":    "onnx/embed_tokens.onnx",
        "decoder":         "onnx/decoder_model_merged.onnx",
        "vision_encoder":  "onnx/vision_encoder.onnx",
    },
    "SmolVLM-256M-Instruct/FP16": {
        "embed_tokens":    "onnx/embed_tokens_fp16.onnx",
        "decoder":         "onnx/decoder_model_merged_fp16.onnx",
        "vision_encoder":  "onnx/vision_encoder_fp16.onnx",
    },
    "SmolVLM-256M-Instruct/INT8": {
        "embed_tokens":    "onnx/embed_tokens_int8.onnx",
        "decoder":         "onnx/decoder_model_merged_int8.onnx",
        "vision_encoder":  "onnx/vision_encoder_int8.onnx",
    },
    "SmolVLM-256M-Instruct/UINT8": {
        "embed_tokens":    "onnx/embed_tokens_uint8.onnx",
        "decoder":         "onnx/decoder_model_merged_uint8.onnx",
        "vision_encoder":  "onnx/vision_encoder_uint8.onnx",
    },
    "SmolVLM-256M-Instruct/Q4": {
        "embed_tokens":    "onnx/embed_tokens_q4.onnx",
        "decoder":         "onnx/decoder_model_merged_q4.onnx",
        "vision_encoder":  "onnx/vision_encoder_q4.onnx",
    },
    "SmolVLM-256M-Instruct/Q4_F16": {
        "embed_tokens":    "onnx/embed_tokens_q4f16.onnx",
        "decoder":         "onnx/decoder_model_merged_q4f16.onnx",
        "vision_encoder":  "onnx/vision_encoder_q4f16.onnx",
    },
    "SmolVLM-256M-Instruct/BNB4": {
        "embed_tokens":    "onnx/embed_tokens_bnb4.onnx",
        "decoder":         "onnx/decoder_model_merged_bnb4.onnx",
        "vision_encoder":  "onnx/vision_encoder_bnb4.onnx",
    },
    "SmolVLM-256M-Instruct/QUANTIZED": {
        "embed_tokens":    "onnx/embed_tokens_quantized.onnx",
        "decoder":         "onnx/decoder_model_merged_quantized.onnx",
        "vision_encoder":  "onnx/vision_encoder_quantized.onnx",
    },
}

# Reference model catalog - curated and tested
REFERENCE_MODELS = {
    ReferenceModel.GEMMA_3N_E2B: ModelSpec(
        repo_id="onnx-community/gemma-3n-E2B-it-ONNX",
        config=ONNXModelConfig(
            num_layers=30,  # Gemma3n text_config has 30 hidden layers
            num_heads=8,    # Gemma3n text_config has 8 attention heads
            head_dim=256,   # hidden_size / num_attention_heads = 2048 / 8 = 256
            has_vision=True,  # Gemma-3n has vision + audio
            position_dims=1,
            subfolder="onnx",
            num_kv_heads=8,  # Gemma3n text_config has 8 key-value heads
            components={
                'audio_encoder': 'audio_encoder{suffix}.onnx',
                'decoder': 'decoder_model_merged{suffix}.onnx',
                'embed_tokens': 'embed_tokens{suffix}.onnx',
                'vision_encoder': 'vision_encoder{suffix}.onnx',
            }
        ),
        supported_quants=[Quantization.Q4, Quantization.QUANTIZED, Quantization.FP16, Quantization.FULL],
        default_quant=Quantization.Q4,
        description="2B multimodal model with vision, text, and audio (SOTA)"
    ),
    
    ReferenceModel.GRANITE_VISION_3_2B: ModelSpec(
        repo_id="ibm-granite/granite-vision-3.2-2b",
        config=ONNXModelConfig(
            num_layers=24,
            num_heads=32,
            head_dim=64,
            has_vision=True,
            position_dims=1,
            subfolder="onnx",
            components={
                'model': 'model{suffix}.onnx',
                'vision': 'vision_encoder{suffix}.onnx'
            }
        ),
        supported_quants=[Quantization.Q4, Quantization.FP16],
        default_quant=Quantization.Q4,
        description="2B vision+text model, Granite Vision architecture"
    ),
    
    ReferenceModel.GEMMA_3N_E2B: ModelSpec(
        repo_id="onnx-community/gemma-3n-E2B-it-ONNX",
        config=ONNXModelConfig(
            num_layers=30,  # Gemma3n text_config has 30 hidden layers
            num_heads=8,    # Gemma3n text_config has 8 attention heads
            head_dim=256,   # hidden_size / num_attention_heads = 2048 / 8 = 256
            has_vision=True,  # Gemma-3n has vision + audio
            position_dims=1,
            subfolder="onnx",
            num_kv_heads=8,  # Gemma3n text_config has 8 key-value heads
            components={
                'audio_encoder': 'audio_encoder{suffix}.onnx',
                'decoder': 'decoder_model_merged{suffix}.onnx',
                'embed_tokens': 'embed_tokens{suffix}.onnx',
                'vision_encoder': 'vision_encoder{suffix}.onnx',
            }
        ),
        supported_quants=[Quantization.Q4, Quantization.QUANTIZED, Quantization.FP16, Quantization.FULL],
        default_quant=Quantization.Q4,
        description="2B multimodal model with vision, text, and audio"
    ),
    
    ReferenceModel.SMOLVLM_256M: ModelSpec(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        config=ONNXModelConfig(
            num_layers=30,
            num_heads=9,
            head_dim=64,  # 576 / 9 = 64
            num_kv_heads=3,
            has_vision=True,
            position_dims=1,
            subfolder="onnx",
            components={
                'embed': 'embed_tokens{suffix}.onnx',
                'decoder': 'decoder_model_merged{suffix}.onnx',
                'vision_encoder': 'vision_encoder{suffix}.onnx',
            }
        ),
        supported_quants=[
            Quantization.UINT8, Quantization.INT8, Quantization.Q4, 
            Quantization.Q4_F16, Quantization.BNB4, Quantization.FP16, 
            Quantization.QUANTIZED, Quantization.FULL
        ],
        default_quant=Quantization.UINT8,  # Smallest available
        description="256M ultra-lightweight vision+text model, excellent for CPU/edge inference"
    )
}


def get_curated_model_config(model_quant_name: str) -> Optional[Dict[str, str]]:
    """Get curated model/quant configuration based on model/quant name."""
    return MODEL_QUANT_CONFIGS.get(model_quant_name)


def get_available_model_quants() -> List[str]:
    """Get list of all available model/quant combinations."""
    return list(MODEL_QUANT_CONFIGS.keys())


def get_smallest_quant_for_model(model_name: str) -> Optional[str]:
    """Get the smallest quantization available for a given model."""
    # Define quant order from smallest to largest
    quant_priority = ["UINT8", "INT8", "Q4", "Q4_MIXED", "Q4_F16", "BNB4", "QUANTIZED", "FP16", "FP32"]
    
    available_configs = [key for key in MODEL_QUANT_CONFIGS.keys() if key.startswith(model_name + "/")]
    
    for quant in quant_priority:
        for config in available_configs:
            if config.endswith("/" + quant):
                return config
    
    # If no specific quant found, return first available
    return available_configs[0] if available_configs else None


def get_onnx_model_config(repo_id: str) -> Optional[ONNXModelConfig]:
    """Get ONNX model configuration based on repository ID."""
    repo_lower = repo_id.lower()
    
    # Direct mapping by repo_id
    for ref_model, spec in REFERENCE_MODELS.items():
        if spec.repo_id.lower() == repo_lower:
            return spec.config
    
    # Pattern matching for partial matches
    for ref_model, spec in REFERENCE_MODELS.items():
        spec_repo_lower = spec.repo_id.lower()
        if spec_repo_lower in repo_lower or repo_lower in spec_repo_lower:
            return spec.config
    
    return None


def create_position_ids(seq_length: int, config: ONNXModelConfig, step: int = 0) -> np.ndarray:
    """Create position_ids tensor based on model configuration."""
    if config.position_dims == 1:
        # Standard 2D position_ids: [batch_size, seq_length]
        return np.arange(seq_length, dtype=np.int64).reshape(1, -1)
    elif config.position_dims == 3:
        # 3D position_ids for vision models: [3, batch_size, seq_length]
        if step == 0:
            # Initial sequence
            text_pos = np.arange(seq_length, dtype=np.int64).reshape(1, seq_length)
            height_pos = np.zeros((1, seq_length), dtype=np.int64)
            width_pos = np.zeros((1, seq_length), dtype=np.int64)
        else:
            # Single token update
            text_pos = np.array([[seq_length + step - 1]], dtype=np.int64)
            height_pos = np.zeros((1, 1), dtype=np.int64)
            width_pos = np.zeros((1, 1), dtype=np.int64)
        
        return np.stack([text_pos, height_pos, width_pos], axis=0)
    else:
        raise ValueError(f"Unsupported position_dims: {config.position_dims}")


def initialize_kv_cache(config: ONNXModelConfig, batch_size: int = 1) -> Dict[str, np.ndarray]:
    """Initialize empty KV cache for the model."""
    kv_cache = {}
    # Use separate kv_heads if available, otherwise default to num_heads
    kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
    
    for i in range(config.num_layers):
        key_name = f'past_key_values.{i}.key'
        value_name = f'past_key_values.{i}.value'
        
        # Shape: [batch_size, num_kv_heads, 0, head_dim] (empty sequence)
        empty_cache = np.zeros((batch_size, kv_heads, 0, config.head_dim), dtype=np.float32)
        kv_cache[key_name] = empty_cache
        kv_cache[value_name] = empty_cache
    
    return kv_cache