import os
import re
from typing import List, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pydantic import ConfigDict

from apps.shared.manage_cache import (
    list_repo_files,
    list_cached_entries,
    download_file,
    delete_cached_file,
)
from apps.shared.unified_model_registry import UnifiedModel, ModelFile

# Import and create fresh registry instance
from apps.shared.unified_model_registry import UnifiedModelRegistry
registry = UnifiedModelRegistry()

router = APIRouter(prefix="", tags=["manage"])


def _normalize_repo_id(raw: str) -> str:
    # If full HF URL, strip protocol and domain
    if "huggingface.co" in raw:
        # Extract part after domain
        parts = raw.split("huggingface.co/", 1)[1]
    else:
        parts = raw
    # Remove any '/tree/' or '/blob/' segments and following path
    parts = re.sub(r"/(?:tree|blob)/.*$", "", parts)
    # Strip leading/trailing slashes
    return parts.strip("/")


class DownloadRequest(BaseModel):
    repo: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")
    path: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")


class DeleteRequest(BaseModel):
    repo: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")
    path: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")


class FileEntry(BaseModel):
    path: str
    size: int

    model_config = ConfigDict(json_schema_extra={"example": {"path": "weights/model.bin", "size": 1234}})


class CacheEntry(BaseModel):
    repo: str
    path: str
    size: int
    last_used: float
    framework: Optional[str] = None
    kind: Optional[str] = None
    inputs: List[Any] = []

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "repo": "myrepo",
                "path": "weights/model.bin",
                "size": 1234,
                "framework": "pytorch",
                "kind": "model",
                "inputs": []
            }
        })


def _validate_paths(repo_id: str, file_path: str | None = None) -> None:
    pattern = re.compile(r"^[A-Za-z0-9_\-./]+$")
    if not pattern.fullmatch(repo_id):
        raise HTTPException(status_code=400, detail="Invalid repo")
    if file_path is not None and not pattern.fullmatch(file_path):
        raise HTTPException(status_code=400, detail="Invalid path")
    root = os.path.realpath(os.path.expanduser("~/.cache/huggingface/hub"))
    if file_path is not None:
        full = os.path.realpath(os.path.join(root, repo_id, file_path))
        if not full.startswith(root):
            raise HTTPException(status_code=400, detail="Path traversal detected")


@router.get("/repos/{repo_id:path}/files", response_model=List[FileEntry], tags=["manage"])
async def get_remote_files(repo_id: str):
    norm_repo = _normalize_repo_id(repo_id)
    _validate_paths(norm_repo)
    try:
        files = list_repo_files(norm_repo)
        # Ensure list_repo_files returns dicts or objects; filter by dict key if needed
        files = [f for f in files if (f.get("path") or getattr(f, "path", "")).lower().endswith(".onnx")]
        return files
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache", response_model=List[CacheEntry], tags=["manage"])
async def get_cache():
    entries = list_cached_entries()
    return entries


@router.post("/cache/download", status_code=202, response_model=None, tags=["manage"])
async def post_download(req: DownloadRequest):
    repo = _normalize_repo_id(req.repo)
    _validate_paths(repo, req.path)
    try:
        download_file(repo_id=repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache", status_code=200, response_model=None, tags=["manage"])
async def delete_cache(req: DeleteRequest):
    repo = _normalize_repo_id(req.repo)
    _validate_paths(repo, req.path)
    try:
        delete_cached_file(repo_id=repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


# === NEW UNIFIED MODEL MANAGEMENT ENDPOINTS ===

class ModelFileResponse(BaseModel):
    """Response model for individual files within a model."""
    path: str
    repo_id: str
    size_mb: Optional[float] = None
    downloaded: bool = False


class UnifiedModelResponse(BaseModel):
    """Response model for unified models."""
    id: str
    name: str
    server: str
    architecture: str
    description: str
    files: List[ModelFileResponse]
    quantizations: List[str] = []
    status: str = "not-downloaded"
    total_size_mb: float = 0.0


class DownloadModelRequest(BaseModel):
    """Request to download a complete model configuration."""
    model_id: str
    quantization: Optional[str] = None  # For multi-component models only


class DeleteModelRequest(BaseModel):
    """Request to delete a complete model."""
    model_id: str
    quantization: Optional[str] = None  # For multi-component models only


@router.get("/unified-models", response_model=List[UnifiedModelResponse], tags=["unified-manage"])
async def get_unified_models():
    """Get all unified models across chat and face servers."""
    try:
        # Update download status from cache
        registry.update_download_status()
        
        models = registry.get_all_models()
        
        # Convert to response format
        response = []
        for model in models:
            files = [
                ModelFileResponse(
                    path=f.path,
                    repo_id=f.repo_id,
                    size_mb=f.size_mb,
                    downloaded=f.downloaded
                )
                for f in model.files
            ]
            
            response.append(UnifiedModelResponse(
                id=model.id,
                name=model.name,
                server=model.server,
                architecture=model.architecture,
                description=model.description,
                files=files,
                quantizations=model.quantizations,
                status=model.status,
                total_size_mb=model.total_size_mb
            ))
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unified-models/{model_id}", response_model=UnifiedModelResponse, tags=["unified-manage"])
async def get_unified_model(model_id: str):
    """Get details for a specific unified model."""
    try:
        registry.update_download_status()
        model = registry.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        files = [
            ModelFileResponse(
                path=f.path,
                repo_id=f.repo_id,
                size_mb=f.size_mb,
                downloaded=f.downloaded
            )
            for f in model.files
        ]
        
        return UnifiedModelResponse(
            id=model.id,
            name=model.name,
            server=model.server,
            architecture=model.architecture,
            description=model.description,
            files=files,
            quantizations=model.quantizations,
            status=model.status,
            total_size_mb=model.total_size_mb
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unified-models/download", status_code=202, tags=["unified-manage"])
async def download_unified_model(req: DownloadModelRequest):
    """Download all files for a unified model configuration."""
    try:
        model = registry.get_model(req.model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found")
        
        # For multi-component models, filter files by quantization if specified
        files_to_download = model.files
        if model.architecture == "multi-component" and req.quantization:
            # Get files for specific quantization from MODEL_QUANT_CONFIGS
            from apps.shared.model_types import get_curated_model_config
            
            # Map model ID back to config name
            model_name = req.model_id.replace("chat-", "")
            
            # Handle special cases for model name formatting
            model_name_mapping = {
                "smolvlm_256m_instruct": "SmolVLM-256M-Instruct",
                "qwen2_vl_2b_instruct": "Qwen2-VL-2B-Instruct", 
                "gemma_3n_e2b_it_onnx": "Gemma-3n-E2B-it-ONNX",
                "phi_3.5_vision_instruct": "Phi-3.5-vision-instruct"
            }
            
            config_model_name = model_name_mapping.get(model_name)
            if not config_model_name:
                # Fallback: title case with underscores to hyphens
                config_model_name = model_name.replace("_", "-").title()
            
            config_key = f"{config_model_name}/{req.quantization}"
            
            quant_config = get_curated_model_config(config_key)
            if not quant_config:
                raise HTTPException(status_code=400, detail=f"Quantization {req.quantization} not available for {model_name}")
            
            # Filter files to only those in this quantization
            quant_files = set(quant_config.values())
            files_to_download = [f for f in model.files if f.path in quant_files]
        
        # Download all files
        download_errors = []
        for file in files_to_download:
            try:
                download_file(repo_id=file.repo_id, file_path=file.path)
            except Exception as e:
                download_errors.append(f"{file.path}: {str(e)}")
        
        if download_errors:
            raise HTTPException(status_code=500, detail=f"Some downloads failed: {'; '.join(download_errors)}")
        
        return {"message": f"Download initiated for {len(files_to_download)} files"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unified-models/{model_id}", status_code=200, tags=["unified-manage"])
async def delete_unified_model(model_id: str, quantization: Optional[str] = None):
    """Delete all files for a unified model (or specific quantization)."""
    try:
        model = registry.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # For multi-component models, filter files by quantization if specified
        files_to_delete = model.files
        if model.architecture == "multi-component" and quantization:
            # Get files for specific quantization from MODEL_QUANT_CONFIGS
            from apps.shared.model_types import get_curated_model_config
            
            # Map model ID back to config name
            model_name = model_id.replace("chat-", "")
            
            # Handle special cases for model name formatting
            model_name_mapping = {
                "smolvlm_256m_instruct": "SmolVLM-256M-Instruct",
                "qwen2_vl_2b_instruct": "Qwen2-VL-2B-Instruct", 
                "gemma_3n_e2b_it_onnx": "Gemma-3n-E2B-it-ONNX",
                "phi_3.5_vision_instruct": "Phi-3.5-vision-instruct"
            }
            
            config_model_name = model_name_mapping.get(model_name)
            if not config_model_name:
                # Fallback: title case with underscores to hyphens
                config_model_name = model_name.replace("_", "-").title()
            
            config_key = f"{config_model_name}/{quantization}"
            
            quant_config = get_curated_model_config(config_key)
            if quant_config:
                # Filter files to only those in this quantization
                quant_files = set(quant_config.values())
                files_to_delete = [f for f in model.files if f.path in quant_files]
        
        # Delete all files
        delete_errors = []
        for file in files_to_delete:
            try:
                delete_cached_file(repo_id=file.repo_id, file_path=file.path)
            except Exception as e:
                delete_errors.append(f"{file.path}: {str(e)}")
        
        if delete_errors:
            raise HTTPException(status_code=500, detail=f"Some deletions failed: {'; '.join(delete_errors)}")
        
        return {"message": f"Deleted {len(files_to_delete)} files"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
