#!/usr/bin/env python3
"""Debug SmolVLM implementation by testing the official example directly."""

import os
from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image
import onnxruntime
import numpy as np

def test_smolvlm_official():
    """Test the official SmolVLM example to see if it works."""
    print("Testing official SmolVLM example...")
    
    # 1. Load models
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    
    try:
        config = AutoConfig.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"✓ Loaded config and processor for {model_id}")
        
        # Print config details
        print(f"Image token ID: {getattr(config, 'image_token_id', 'Not found')}")
        print(f"EOS token ID: {config.text_config.eos_token_id}")
        print(f"Num layers: {config.text_config.num_hidden_layers}")
        print(f"Num KV heads: {config.text_config.num_key_value_heads}")
        print(f"Head dim: {config.text_config.head_dim}")
        
    except Exception as e:
        print(f"✗ Failed to load config/processor: {e}")
        return False
    
    # Check if ONNX files exist or need to be downloaded
    onnx_files = [
        "vision_encoder.onnx",
        "embed_tokens.onnx", 
        "decoder_model_merged.onnx"
    ]
    
    for onnx_file in onnx_files:
        if not os.path.exists(onnx_file):
            print(f"⚠ ONNX file {onnx_file} not found - would need to download")
            # For now, we can't continue without the ONNX files
            return False
        else:
            print(f"✓ Found {onnx_file}")
    
    # If we had the files, we would continue with the official example
    print("Would continue with official SmolVLM generation if ONNX files were available")
    return True

def test_smolvlm_processor_only():
    """Test just the SmolVLM processor to see message formatting."""
    print("\nTesting SmolVLM processor message formatting...")
    
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Create test messages like official example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe this image?"}
                ]
            },
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        print(f"Generated prompt: {repr(prompt)}")
        
        # Test with a sample image URL
        try:
            image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
            inputs = processor(text=prompt, images=[image], return_tensors="np")
            
            print(f"Input IDs shape: {inputs['input_ids'].shape}")
            print(f"Attention mask shape: {inputs['attention_mask'].shape}")
            if 'pixel_values' in inputs:
                print(f"Pixel values shape: {inputs['pixel_values'].shape}")
            if 'pixel_attention_mask' in inputs:
                print(f"Pixel attention mask shape: {inputs['pixel_attention_mask'].shape}")
            
            # Check for image tokens in the input
            config = AutoConfig.from_pretrained(model_id)
            image_token_id = getattr(config, 'image_token_id', None)
            if image_token_id:
                image_token_count = np.sum(inputs['input_ids'] == image_token_id)
                print(f"Found {image_token_count} image tokens (ID: {image_token_id}) in input")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to process image: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load processor: {e}")
        return False

if __name__ == "__main__":
    print("SmolVLM Debug Test")
    print("="*50)
    
    # Test 1: Official example structure
    test_smolvlm_official()
    
    # Test 2: Processor-only test
    test_smolvlm_processor_only()