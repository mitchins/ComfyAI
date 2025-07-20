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
    
    # Qwen2-VL models - Vision + Text
    QWEN2_VL_2B = "qwen2-vl-2b"
    QWEN2_VL_7B = "qwen2-vl-7b"
    
    # Granite models - Vision + Text (NOT just text-only granite)
    GRANITE_VISION_3_2B = "granite-vision-3-2b"
    
    # Gemma-3n models - Coming soon
    GEMMA_3N_E2B = "gemma-3n-e2b"


class Quantization(str, Enum):
    """Supported quantization levels."""
    
    Q4 = "q4"           # 4-bit quantized
    FP16 = "fp16"       # 16-bit floating point
    INT8 = "int8"       # 8-bit integer
    UINT8 = "uint8"     # 8-bit unsigned
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


# Reference model catalog - curated and tested
REFERENCE_MODELS = {
    ReferenceModel.QWEN2_VL_2B: ModelSpec(
        repo_id="onnx-community/Qwen2-VL-2B-Instruct",
        config=ONNXModelConfig(
            num_layers=28,
            num_heads=2,
            head_dim=128,
            has_vision=True,
            position_dims=3,  # text, height, width
            subfolder="onnx"
        ),
        supported_quants=[Quantization.Q4, Quantization.FP16],
        default_quant=Quantization.Q4,
        description="2B vision+text model, multi-component architecture"
    ),
    
    ReferenceModel.QWEN2_VL_7B: ModelSpec(
        repo_id="onnx-community/Qwen2-VL-7B-Instruct", 
        config=ONNXModelConfig(
            num_layers=32,
            num_heads=4,
            head_dim=128,
            has_vision=True,
            position_dims=3,
            subfolder="onnx"
        ),
        supported_quants=[Quantization.Q4, Quantization.FP16],
        default_quant=Quantization.Q4,
        description="7B vision+text model, multi-component architecture"
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
            num_layers=24,
            num_heads=16,
            head_dim=64,
            has_vision=False,
            position_dims=1,
            subfolder=".",  # Root of repo
            components={
                'model': 'model{suffix}.onnx'
            }
        ),
        supported_quants=[Quantization.Q4, Quantization.FP16],
        default_quant=Quantization.Q4,
        description="2B text model, Gemma-3n architecture (coming soon)"
    )
}


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
    for i in range(config.num_layers):
        key_name = f'past_key_values.{i}.key'
        value_name = f'past_key_values.{i}.value'
        
        # Shape: [batch_size, num_heads, 0, head_dim] (empty sequence)
        empty_cache = np.zeros((batch_size, config.num_heads, 0, config.head_dim), dtype=np.float32)
        kv_cache[key_name] = empty_cache
        kv_cache[value_name] = empty_cache
    
    return kv_cache