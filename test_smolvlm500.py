#!/usr/bin/env python3
"""Test SmolVLM-500M using the official example approach."""

from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import onnxruntime
import numpy as np
import os

# 1. Load models
## Load config and processor
model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
config = AutoConfig.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

## Load sessions
vision_session = onnxruntime.InferenceSession("smolvlm500_onnx/vision_encoder.onnx")
embed_session = onnxruntime.InferenceSession("smolvlm500_onnx/embed_tokens.onnx")
decoder_session = onnxruntime.InferenceSession("smolvlm500_onnx/decoder_model_merged.onnx")

## Set config values
num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
num_hidden_layers = config.text_config.num_hidden_layers
eos_token_id = config.text_config.eos_token_id
image_token_id = config.image_token_id

print(f"SmolVLM-500M Config: num_kv_heads={num_key_value_heads}, head_dim={head_dim}, num_layers={num_hidden_layers}")
print(f"EOS token ID: {eos_token_id}, Image token ID: {image_token_id}")

# 2. Prepare inputs
## Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in the provided image?"}
        ]
    },
]

## Load image and apply processor  
# Test with our pizza image
image = Image.open("tests/pizza.jpg")
if image.mode != 'RGB':
    image = image.convert('RGB')

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="np")

print(f"Prompt: {repr(prompt)}")
print(f"Input shapes: input_ids={inputs['input_ids'].shape}, attention_mask={inputs['attention_mask'].shape}")
if 'pixel_values' in inputs:
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
if 'pixel_attention_mask' in inputs:
    print(f"Pixel attention mask shape: {inputs['pixel_attention_mask'].shape}")

# Check for image tokens
image_token_count = np.sum(inputs['input_ids'] == image_token_id)
print(f"Found {image_token_count} image tokens in input")

## Prepare decoder inputs
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
image_features = None
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
position_ids = np.cumsum(inputs['attention_mask'], axis=-1)

# 3. Generation loop
max_new_tokens = 100  # Try more tokens for better response
generated_tokens = np.array([[]], dtype=np.int64)

print(f"\nStarting generation (max {max_new_tokens} tokens)...")

for i in range(max_new_tokens):
    inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

    if image_features is None:
        ## Only compute vision features if not already computed
        print(f"Step {i+1}: Computing vision features...")
        vision_inputs = {
            'pixel_values': inputs['pixel_values'],
            'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
        }
        image_features = vision_session.run(['image_features'], vision_inputs)[0]
        print(f"  image_features shape: {image_features.shape}")
        
        ## Merge text and vision embeddings
        mask = (inputs['input_ids'] == image_token_id)
        vision_features_flat = image_features.reshape(-1, image_features.shape[-1])
        num_image_tokens = np.sum(mask)
        
        print(f"  Injecting {num_image_tokens} image tokens with vision features")
        inputs_embeds[mask] = vision_features_flat

    decoder_inputs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **past_key_values,
    )
    
    outputs = decoder_session.run(None, decoder_inputs)
    logits = outputs[0]

    ## Update values for next generation loop
    next_token = logits[:, -1].argmax(-1, keepdims=True)
    
    input_ids = next_token
    attention_mask = np.ones_like(input_ids)
    position_ids = position_ids[:, -1:] + 1
    for j, key in enumerate(past_key_values):
        past_key_values[key] = outputs[j+1]  # outputs[0] is logits

    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
    
    # Check for EOS
    if (input_ids == eos_token_id).all():
        print(f"  EOS token reached at step {i+1}")
        break

    ## (Optional) Streaming
    decoded_token = processor.decode(input_ids[0], skip_special_tokens=False)
    print(f"Step {i+1}: {repr(decoded_token)}")

print(f"\nGeneration complete. Generated {generated_tokens.shape[-1]} tokens.")

# 4. Output result
result = processor.batch_decode(generated_tokens, skip_special_tokens=True)
print(f"\nFinal result: {result}")
print(f"Final result clean: {repr(result[0])}")