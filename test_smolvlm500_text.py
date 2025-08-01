#!/usr/bin/env python3
"""Test SmolVLM-500M text-only generation."""

from transformers import AutoConfig, AutoProcessor
import onnxruntime
import numpy as np

# 1. Load models
model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
config = AutoConfig.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

embed_session = onnxruntime.InferenceSession("smolvlm500_onnx/embed_tokens.onnx")
decoder_session = onnxruntime.InferenceSession("smolvlm500_onnx/decoder_model_merged.onnx")

num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
num_hidden_layers = config.text_config.num_hidden_layers
eos_token_id = config.text_config.eos_token_id

print(f"Testing text-only generation with EOS token ID: {eos_token_id}")

# 2. Prepare text-only input
messages = [{"role": "user", "content": [{"type": "text", "text": "Hello, what is 2+2?"}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, return_tensors="np")

print(f"Prompt: {repr(prompt)}")
print(f"Input shape: {inputs['input_ids'].shape}")

# 3. Simple generation loop
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
position_ids = np.cumsum(attention_mask, axis=-1)
generated_tokens = np.array([[]], dtype=np.int64)

print("Starting text-only generation...")

for i in range(20):  # Just 20 tokens for testing
    inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]
    
    decoder_inputs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **past_key_values,
    )
    
    outputs = decoder_session.run(None, decoder_inputs)
    logits = outputs[0]
    next_token = logits[:, -1].argmax(-1, keepdims=True)
    
    if (next_token == eos_token_id).all():
        print(f"EOS reached at step {i+1}")
        break
    
    input_ids = next_token
    attention_mask = np.ones_like(input_ids)
    position_ids = position_ids[:, -1:] + 1
    
    for j, key in enumerate(past_key_values):
        past_key_values[key] = outputs[j+1]
    
    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
    
    decoded_token = processor.decode(input_ids[0], skip_special_tokens=False)
    print(f"Step {i+1}: {repr(decoded_token)}")

result = processor.batch_decode(generated_tokens, skip_special_tokens=True)
print(f"\nFinal result: {repr(result[0])}")