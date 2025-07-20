from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
    
    def __post_init__(self):
        if self.components is None:
            self.components = {
                'embed': 'embed_tokens{suffix}.onnx',
                'decoder': 'decoder_model_merged{suffix}.onnx'
            }
            if self.has_vision:
                self.components['vision'] = 'vision_encoder{suffix}.onnx'


# Reference ONNX model configurations - tested and verified working setups
ONNX_MODEL_CONFIGS = {
    # ✅ WORKING: Qwen2-VL 2B with multi-component architecture
    'qwen2-vl-2b': ONNXModelConfig(
        num_layers=28,
        num_heads=2, 
        head_dim=128,
        has_vision=True,
        position_dims=3,  # text, height, width
        subfolder="onnx"
        # Uses default multi-component: embed_tokens, decoder_model_merged, vision_encoder
    ),
    
    # ✅ WORKING: Qwen2-VL 7B with multi-component architecture  
    'qwen2-vl-7b': ONNXModelConfig(
        num_layers=32,
        num_heads=4,
        head_dim=128,
        has_vision=True,
        position_dims=3,
        subfolder="onnx"
    ),
    
    # ✅ WORKING: Granite 3.0 2B with single model architecture
    'granite-3.0-2b': ONNXModelConfig(
        num_layers=26,
        num_heads=32,
        head_dim=64,
        has_vision=False,
        position_dims=1,
        subfolder="onnx",
        components={
            'model': 'model{suffix}.onnx'  # Single model file - tested with _q4
        }
    ),
    
    # 🚧 TODO: Gemma 3n 2B - debugging in progress
    # 'gemma-3n-2b': ONNXModelConfig(
    #     num_layers=26,  
    #     num_heads=8,    
    #     head_dim=256,   
    #     has_vision=True,  
    #     position_dims=1,  
    #     subfolder="onnx",
    #     components={
    #         'embed': 'embed_tokens{suffix}.onnx',
    #         'decoder': 'decoder_model_merged{suffix}.onnx', 
    #         'vision': 'vision_encoder{suffix}.onnx',
    #         'audio': 'audio_encoder{suffix}.onnx'
    #     }
    # )
}

# Known working model specifications for reference server
REFERENCE_MODELS = {
    # Qwen2-VL models - multi-component, vision support
    "onnx-community/Qwen2-VL-2B-Instruct": {
        "config": "qwen2-vl-2b",
        "recommended_quant": "_q4",
        "description": "2B vision-language model, multi-component architecture"
    },
    "onnx-community/Qwen2-VL-7B-Instruct": {
        "config": "qwen2-vl-7b", 
        "recommended_quant": "_q4",
        "description": "7B vision-language model, multi-component architecture"
    },
    
    # Granite models - single file, text-only
    "onnx-community/granite-3.0-2b-instruct": {
        "config": "granite-3.0-2b",
        "recommended_quant": "_q4", 
        "description": "2B text-only model, single file architecture"
    },
    
    # 🚧 TODO: Gemma 3n models - debugging in progress
    # "onnx-community/gemma-3n-E2B-it-ONNX": {
    #     "config": "gemma-3n-2b",
    #     "recommended_quant": "_q4",
    #     "description": "2B vision+audio+text model, multi-component architecture" 
    # }
}


def get_onnx_model_config(repo_id: str) -> Optional[ONNXModelConfig]:
    """Get ONNX model configuration based on repository ID."""
    repo_lower = repo_id.lower()
    
    # Direct mapping
    for key, config in ONNX_MODEL_CONFIGS.items():
        if key in repo_lower:
            return config
    
    # Pattern matching
    if 'qwen2-vl' in repo_lower:
        if '7b' in repo_lower:
            return ONNX_MODEL_CONFIGS['qwen2-vl-7b']
        else:
            return ONNX_MODEL_CONFIGS['qwen2-vl-2b']
    elif 'granite' in repo_lower and '3.0' in repo_lower and '2b' in repo_lower:
        return ONNX_MODEL_CONFIGS['granite-3.0-2b']
    elif 'gemma' in repo_lower and '3n' in repo_lower and '2b' in repo_lower:
        return ONNX_MODEL_CONFIGS['gemma-3n-2b']
    
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
    for i in range(config.num_layers):
        key_name = f'past_key_values.{i}.key'
        value_name = f'past_key_values.{i}.value'
        
        # Shape: [batch_size, num_heads, 0, head_dim] (empty sequence)
        empty_cache = np.zeros((batch_size, config.num_heads, 0, config.head_dim), dtype=np.float32)
        kv_cache[key_name] = empty_cache
        kv_cache[value_name] = empty_cache
    
    return kv_cache