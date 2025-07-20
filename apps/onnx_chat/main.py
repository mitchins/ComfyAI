from __future__ import annotations
import os
from typing import List, Dict, Any
import argparse
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .config import load_config, setup_logging
from apps.shared.manage_cache import download_file

from transformers import AutoTokenizer
from transformers import AutoConfig
from optimum.onnxruntime import ORTModelForCausalLM

import onnxruntime as ort_rt
import numpy as np
import base64
import io
from PIL import Image

# Utility to find the nearest ancestor directory containing config.json
def find_config_root(path: str) -> str | None:
    dir_path = os.path.dirname(path)
    while True:
        if os.path.exists(os.path.join(dir_path, "config.json")):
            return dir_path
        parent = os.path.dirname(dir_path)
        if parent == dir_path:
            return None
        dir_path = parent

setup_logging()
config = load_config()

MODEL_PATH = config.model_path
DEFAULT_FILE = os.getenv("ONNX_FILE_NAME", "model.onnx")

# Caches for loaded models and tokenizers
ort_models: dict[str, Any] = {}
ort_sessions: dict[str, ort_rt.InferenceSession] = {}
tokenizers: dict[str, AutoTokenizer] = {}

# Resolve a model identifier to a local ONNX file path, downloading if necessary
import logging
from huggingface_hub import hf_hub_download

def _ensure_model_path(name: str) -> str | None:
    if os.path.exists(name):
        return name
    # parse repo and filename
    if ":" in name:
        repo_id, filename = (name.split(":", 1) + [DEFAULT_FILE])[:2]
    else:
        parts = name.split("/")
        if len(parts) >= 3 and name.endswith(".onnx"):
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
        else:
            repo_id = name
            filename = DEFAULT_FILE
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to load model: {name} (repo={repo_id}, file={filename})")
    # check cache entries
    from apps.shared.manage_cache import list_cached_entries
    from huggingface_hub.utils._cache_manager import scan_cache_dir
    try:
        for entry in list_cached_entries():
            if entry.get("repo") == repo_id and entry.get("path") == filename:
                cache = scan_cache_dir()
                for repo in cache.repos:
                    if repo.repo_id == repo_id:
                        for rev in repo.revisions:
                            for f in rev.files:
                                if f.file_name == filename and os.path.exists(f.file_path):
                                    logger.info(f"Found cached model at {f.file_path}")
                                    return f.file_path
    except Exception:
        logger.warning("Cache scan failed, will attempt download")
    # not in cache, download
    try:
        hf_hub_download(repo_id=repo_id, filename=filename)
        # retry scan for downloaded file
        return _ensure_model_path(name)
    except Exception as e:
        logger.error(f"Failed to download {repo_id}/{filename}: {e}")
        return None

# Helper to parse model name and download ONNX file
def parse_model_name(name: str) -> tuple[str, str]:
    """Split model identifier into HF repo and filename."""
    if ":" in name:
        repo_id, filename = (name.split(":", 1) + [DEFAULT_FILE])[:2]
    else:
        parts = name.split("/")
        if name.endswith(".onnx") and len(parts) >= 3:
            repo_id = "/".join(parts[:-1])
            filename = parts[-1]
        else:
            repo_id = name
            filename = DEFAULT_FILE
    return repo_id, filename

def download_model_components(repo_id: str, filename: str) -> None:
    # Helper to ensure model files are cached locally
    try:
        hf_hub_download(repo_id=repo_id, filename=filename)
        hf_hub_download(repo_id=repo_id, filename="config.json")
        
        # Download standard tokenizer and processor files
        auxiliary_files = [
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "generation_config.json",
            "added_tokens.json",
            "chat_template.json"
        ]
        
        # Download vision model ONNX components for multimodal models
        # Extract quantization suffix from main filename
        quant_suffix = ""
        if filename != "model.onnx":
            # Extract suffix like "_q4", "_fp16", "_int8" etc.
            base_name = filename.replace(".onnx", "").replace("decoder_model_merged", "").replace("embed_tokens", "").replace("vision_encoder", "")
            if base_name and base_name.startswith("_"):
                quant_suffix = base_name
        
        vision_files = [
            f"onnx/embed_tokens{quant_suffix}.onnx",
            f"onnx/decoder_model_merged{quant_suffix}.onnx", 
            f"onnx/vision_encoder{quant_suffix}.onnx"
        ]
        
        for aux_file in auxiliary_files:
            try:
                hf_hub_download(repo_id=repo_id, filename=aux_file)
            except Exception:
                pass
                
        for vision_file in vision_files:
            try:
                hf_hub_download(repo_id=repo_id, filename=vision_file)
            except Exception:
                pass
    except Exception:
        pass

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    images: List[str] | None = None  # base64-encoded images
    max_tokens: int | None = None


@app.get("/health")
async def health_check():
    model_name = os.path.basename(MODEL_PATH) if MODEL_PATH else "none"
    return {"status": "ok", "model": model_name}


# Helper to load or retrieve ONNX session and tokenizer
async def get_model_and_tokenizer(model_name: str) -> tuple[dict, AutoTokenizer]:
    repo_id, filename = parse_model_name(model_name)
    cache_key = model_name
    
    if cache_key not in ort_sessions:
        # Download all required model components
        download_model_components(repo_id, filename)
        
        # Load tokenizer from repo
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=True)
        
        # Try optimum approach first, fallback to direct ONNX Runtime
        try:
            # Use the specified filename or default decoder file with onnx/ prefix
            if "decoder_model_merged" in filename:
                decoder_file = f"onnx/{filename}"
            else:
                decoder_file = "onnx/decoder_model_merged.onnx"
            
            model = ORTModelForCausalLM.from_pretrained(
                repo_id,
                file_name=decoder_file,
                trust_remote_code=True,
            )
            ort_models[cache_key] = model
            
        except Exception as e:
            # Fallback to direct ONNX Runtime for incompatible models
            logger = logging.getLogger(__name__)
            logger.warning(f"Optimum loading failed: {e}. Falling back to direct ONNX Runtime.")
            
            # Load multi-component ONNX models
            quant_suffix = ""
            if filename != "model.onnx" and "decoder_model_merged" in filename:
                base_name = filename.replace(".onnx", "").replace("decoder_model_merged", "")
                if base_name and base_name.startswith("_"):
                    quant_suffix = base_name
            
            # Download and load embed_tokens model
            embed_file = f"onnx/embed_tokens{quant_suffix}.onnx"
            embed_path = hf_hub_download(repo_id=repo_id, filename=embed_file)
            embed_session = ort_rt.InferenceSession(embed_path)
            
            # Download and load decoder model
            decoder_file = f"onnx/decoder_model_merged{quant_suffix}.onnx"
            decoder_path = hf_hub_download(repo_id=repo_id, filename=decoder_file)
            decoder_session = ort_rt.InferenceSession(decoder_path)
            
            # Store as dict of sessions
            ort_sessions[cache_key] = {
                'embed': embed_session,
                'decoder': decoder_session
            }
            
        tokenizers[cache_key] = tokenizer
        
    # Return either optimum model or session dict
    if cache_key in ort_models:
        return ort_models[cache_key], tokenizers[cache_key]
    else:
        return ort_sessions[cache_key], tokenizers[cache_key]


# Generate text using ONNX Runtime and tokenizer
async def generate_text(text: str, model_name: str, max_tokens: int = 100, images: List[str] | None = None) -> str:
    model, tokenizer = await get_model_and_tokenizer(model_name)
    if images:
        raise HTTPException(status_code=400, detail="Image inputs not supported for this model")
    
    # Check if we have an optimum model or direct ONNX session
    if isinstance(model, ORTModelForCausalLM):
        # Use optimum's generate method
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Direct ONNX Runtime inference using separate embed + decoder models
        inputs = tokenizer(text, return_tensors='np')
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', np.ones_like(input_ids))
        
        embed_model = model['embed']
        decoder_model = model['decoder']
        
        # Get decoder input names
        decoder_input_names = [inp.name for inp in decoder_model.get_inputs()]
        
        # Initialize tracking
        batch_size = 1
        seq_length = input_ids.shape[1]
        
        # Get embeddings from embed model
        embed_outputs = embed_model.run(None, {'input_ids': input_ids})
        inputs_embeds = embed_outputs[0]
        
        # Prepare decoder inputs
        onnx_inputs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask
        }
        
        # Add position_ids (3D tensor for Qwen2-VL: [3, batch, seq_len])
        if 'position_ids' in decoder_input_names:
            # Qwen2-VL uses 3D position embeddings: text, image height, image width
            text_pos = np.arange(seq_length, dtype=np.int64).reshape(1, seq_length)
            height_pos = np.zeros((1, seq_length), dtype=np.int64)  # No image for text-only
            width_pos = np.zeros((1, seq_length), dtype=np.int64)   # No image for text-only
            onnx_inputs['position_ids'] = np.stack([text_pos, height_pos, width_pos], axis=0)
        
        # Initialize past_key_values (KV cache) - 28 layers for Qwen2-VL-2B
        num_layers = 28
        head_dim = 128  # Corrected based on model error
        num_heads = 2   # Corrected based on model error
        
        for i in range(num_layers):
            if f'past_key_values.{i}.key' in decoder_input_names:
                # Initialize empty KV cache
                onnx_inputs[f'past_key_values.{i}.key'] = np.zeros((batch_size, num_heads, 0, head_dim), dtype=np.float32)
                onnx_inputs[f'past_key_values.{i}.value'] = np.zeros((batch_size, num_heads, 0, head_dim), dtype=np.float32)
        
        # Generate tokens
        generated_ids = input_ids.copy()
        
        for step in range(max_tokens):
            # Run decoder inference
            outputs = decoder_model.run(None, onnx_inputs)
            
            # Get logits (first output)
            logits = outputs[0]
            next_token = np.argmax(logits[0, -1, :])
            
            if next_token == tokenizer.eos_token_id:
                break
            
            # Update generated sequence
            generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
            
            # Get embeddings for next token
            next_token_embed = embed_model.run(None, {'input_ids': np.array([[next_token]])})
            next_inputs_embeds = next_token_embed[0]
            
            # Update inputs for next iteration
            onnx_inputs['inputs_embeds'] = next_inputs_embeds
            onnx_inputs['attention_mask'] = np.ones((1, generated_ids.shape[1]), dtype=np.int64)
            if 'position_ids' in decoder_input_names:
                # Update position_ids for new token (3D format)
                new_pos = seq_length + step
                text_pos = np.array([[new_pos]], dtype=np.int64)
                height_pos = np.zeros((1, 1), dtype=np.int64)
                width_pos = np.zeros((1, 1), dtype=np.int64)
                onnx_inputs['position_ids'] = np.stack([text_pos, height_pos, width_pos], axis=0)
            
            # Update KV cache with new outputs
            output_idx = 1  # Skip logits
            for i in range(num_layers):
                if f'past_key_values.{i}.key' in decoder_input_names:
                    onnx_inputs[f'past_key_values.{i}.key'] = outputs[output_idx]
                    onnx_inputs[f'past_key_values.{i}.value'] = outputs[output_idx + 1]
                    output_idx += 2
        
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        req = ChatRequest(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    text = ""
    if req.messages:
        content = req.messages[-1].get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
        else:
            text = str(content)
    result = await generate_text(text, req.model or (MODEL_PATH or ""), req.max_tokens or 100, req.images)
    return {
        "id": "cmpl-001",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}
        ],
        "model": req.model,
    }


def main() -> None:
    """CLI entry point for running the server via ``python -m``."""
    import uvicorn

    parser = argparse.ArgumentParser(description="ONNX chat completion server")
    parser.add_argument("--host", default=config.host)
    parser.add_argument("--port", type=int, default=config.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "apps.onnx_chat.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
