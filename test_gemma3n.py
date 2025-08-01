import onnxruntime
from onnxruntime import SessionOptions
import numpy as np
from transformers import AutoConfig, AutoProcessor
import os
import time
import argparse

# Parse execution provider option
parser = argparse.ArgumentParser()
parser.add_argument(
    "--provider",
    choices=["cpu", "cuda"],
    default="cpu",
    help="Execution provider: 'cpu' or 'cuda' (default: cpu)"
)
args = parser.parse_args()
provider = args.provider

def make_session(model_path):
    so = onnxruntime.SessionOptions()
    if provider == "cuda":
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            try:
                so.enable_cuda_graph = True
            except AttributeError:
                print("CUDA graph not supported; proceeding without it.")
            session = onnxruntime.InferenceSession(
                model_path,
                sess_options=so,
                providers=["CUDAExecutionProvider"]
            )
            print(f"Loaded {model_path} with CUDAExecutionProvider")
            return session
        raise RuntimeError(f"CUDAExecutionProvider unavailable for {model_path}; aborting.")
    # CPU fallback
    session = onnxruntime.InferenceSession(
        model_path,
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )
    print(f"Loaded {model_path} with CPUExecutionProvider")
    return session

# Check which execution providers are available globally
print("Available ONNX providers:", onnxruntime.get_available_providers())

# Utility to convert returned tensors or arrays to numpy arrays
def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else x

# 1. Load models
## Load config and processor
model_id = "google/gemma-3n-E2B-it"
try:
    processor = AutoProcessor.from_pretrained(model_id)
    # Try patching fast image processor to allow fallback if necessary
    if hasattr(processor, "image_processor") and getattr(processor.image_processor, "is_fast", False):
        # Reload slow version of image processor if fast version doesn't support numpy
        processor.image_processor = processor.image_processor.__class__.from_pretrained(model_id, use_fast=False)
except ValueError as e:
    # Fall back entirely to use_fast=False if fast processor fails
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
config = AutoConfig.from_pretrained(model_id)

## Load sessions
model_dir          = "."
embed_model_path   = os.path.join(model_dir, "onnx/embed_tokens_quantized.onnx")
audio_model_path   = os.path.join(model_dir, "onnx/audio_encoder_q4.onnx")
vision_model_path  = os.path.join(model_dir, "onnx/vision_encoder_quantized.onnx")
decoder_model_path = os.path.join(model_dir, "onnx/decoder_model_merged_q4.onnx")
vision_session     = make_session(vision_model_path)
audio_session      = make_session(audio_model_path)
embed_session      = make_session(embed_model_path)
decoder_session    = make_session(decoder_model_path)

# Print active providers for each session
print("Vision session providers:", vision_session.get_providers())
print("Audio session providers:", audio_session.get_providers())
print("Embed session providers:", embed_session.get_providers())
print("Decoder session providers:", decoder_session.get_providers())

## Set config values
num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
num_hidden_layers = config.text_config.num_hidden_layers
eos_token_id = 106 # != config.text_config.eos_token_id
image_token_id = config.image_token_id
audio_token_id = config.audio_token_id


# 2. Prepare inputs
## Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "In detail, describe the following audio and image."},
            {"type": "audio", "audio": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"},
            {"type": "image", "image": "https://www.catsinsinks.com/cats/428cfaa91c029.jpg"},
        ],
    },
]
preprocess_start = time.time()
try:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
    )
except ValueError as e:
    # Fallback to PyTorch tensors if numpy return not supported
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
# Measure preprocessing time
preprocess_time = time.time() - preprocess_start
print(f"Preprocessing took {preprocess_time:.3f} seconds")
input_ids = to_numpy(inputs["input_ids"])
attention_mask = to_numpy(inputs["attention_mask"])
position_ids = np.cumsum(attention_mask, axis=-1) - 1

pixel_values = to_numpy(inputs.get("pixel_values")) if inputs.get("pixel_values") is not None else None
input_features = to_numpy(inputs.get("input_features")).astype(np.float32) if inputs.get("input_features") is not None else None
input_features_mask = to_numpy(inputs.get("input_features_mask")) if inputs.get("input_features_mask") is not None else None

## Prepare decoder inputs
batch_size = input_ids.shape[0]
past_key_values = {
    f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ("key", "value")
}

# Batch generation of multiple tokens per inference call to amortize kernel launches
new_tokens_per_batch = 4

#
# 3. Generation loop
max_new_tokens = 1024
generated_tokens = np.array([[]], dtype=np.int64)
image_features = None
audio_features = None
# Start generation timer
gen_start = time.time()
for i in range(max_new_tokens):
    # TODO: Implement full batching logic for generating new_tokens_per_batch tokens per iteration
    inputs_embeds, per_layer_inputs = embed_session.run(None, {"input_ids": input_ids})
    if image_features is None and pixel_values is not None:
        image_features = vision_session.run(
            ["image_features"],
            {
                "pixel_values": pixel_values,
            }
        )[0]
        mask = (input_ids == image_token_id).reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds[mask] = image_features.reshape(-1, image_features.shape[-1])
        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

    if audio_features is None and input_features is not None and input_features_mask is not None:
        audio_features = audio_session.run(
            ["audio_features"],
            {
                "input_features": input_features,
                "input_features_mask": input_features_mask,
            }
        )[0]
        mask = (input_ids == audio_token_id).reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds[mask] = audio_features.reshape(-1, audio_features.shape[-1])
        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

    logits, *present_key_values = decoder_session.run(None, dict(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        position_ids=position_ids,
        **past_key_values,
    ))

    ## Update values for next generation loop
    input_ids = logits[:, -1].argmax(-1, keepdims=True)
    attention_mask = np.ones_like(input_ids)
    position_ids = position_ids[:, -1:] + 1
    for j, key in enumerate(past_key_values):
        past_key_values[key] = present_key_values[j]

    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
    if (input_ids == eos_token_id).all():
        break

    ## (Optional) Streaming
    print(processor.decode(input_ids[0]), end="", flush=True)
print()

# End generation timer and report
gen_time = time.time() - gen_start
total_tokens = generated_tokens.shape[-1]
print(f"Generated {total_tokens} tokens in {gen_time:.3f} seconds ({total_tokens/gen_time:.2f} tokens/sec)")

# 4. Output result
print(processor.batch_decode(generated_tokens, skip_special_tokens=True)[0])
