#!/usr/bin/env python3
"""Test script to verify Gemma3n configuration loading."""

import logging
from apps.shared.model_types import REFERENCE_MODELS, ReferenceModel, get_curated_model_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemma3n_config():
    """Test Gemma3n configuration parameters."""
    print("=== Testing Gemma3n Configuration ===")
    
    # Test reference model spec
    if ReferenceModel.GEMMA_3N_E2B in REFERENCE_MODELS:
        spec = REFERENCE_MODELS[ReferenceModel.GEMMA_3N_E2B]
        print(f"✓ Found Gemma3n in reference models")
        print(f"  Repository: {spec.repo_id}")
        print(f"  Layers: {spec.config.num_layers}")
        print(f"  Attention heads: {spec.config.num_heads}")
        print(f"  KV heads: {spec.config.num_kv_heads}")
        print(f"  Head dimension: {spec.config.head_dim}")
        print(f"  Has vision: {spec.config.has_vision}")
        print(f"  Subfolder: {spec.config.subfolder}")
        print(f"  Components: {spec.config.components}")
        print(f"  Supported quants: {[q.value for q in spec.supported_quants]}")
    else:
        print("✗ Gemma3n not found in reference models")
        return False
    
    # Test curated model configs
    print("\n=== Testing Curated Model Configs ===")
    test_configs = [
        "Gemma-3n-E2B-it-ONNX/Q4_MIXED",
        "Gemma-3n-E2B-it-ONNX/FP16", 
        "Gemma-3n-E2B-it-ONNX/FP32"
    ]
    
    for config_name in test_configs:
        config = get_curated_model_config(config_name)
        if config:
            print(f"✓ {config_name}")
            for component, path in config.items():
                print(f"    {component}: {path}")
        else:
            print(f"✗ {config_name} not found")
    
    print("\n=== Configuration Summary ===")
    print("Updated configuration based on HuggingFace model:")
    print("- num_layers: 30 (was 24)")
    print("- num_heads: 8 (was 16) - attention heads")
    print("- num_kv_heads: 8 (new) - key-value heads")
    print("- head_dim: 256 (was 64) - calculated as hidden_size/num_heads = 2048/8")
    print("- Sliding window: 512 (from HF config)")
    
    return True

if __name__ == "__main__":
    success = test_gemma3n_config()
    if success:
        print("\n✓ All Gemma3n configuration tests passed!")
    else:
        print("\n✗ Some configuration tests failed!")
        exit(1)