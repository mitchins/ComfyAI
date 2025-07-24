"""ONNX Chat Router for integration with main app."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from typing import List, Dict, Any
import logging

from .main import (
    ChatRequest, 
    get_inference_engine, 
    generate_text,
    MODEL_PATH
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint for ONNX chat service."""
    model_name = MODEL_PATH if MODEL_PATH else "dynamic"
    return {"status": "ok", "service": "onnx-chat", "model": model_name}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        req = ChatRequest(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())

    # Extract text from messages
    text = ""
    if req.messages:
        content = req.messages[-1].get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
        else:
            text = str(content)
    
    try:
        result = await generate_text(
            text, 
            req.model or (MODEL_PATH or ""), 
            req.max_tokens or 100, 
            req.images,
            req.audio
        )
        
        return {
            "id": "cmpl-001",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0, 
                    "message": {"role": "assistant", "content": result}, 
                    "finish_reason": "stop"
                }
            ],
            "model": req.model,
        }
    except Exception as e:
        import traceback
        logger.error(f"Error generating text: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")