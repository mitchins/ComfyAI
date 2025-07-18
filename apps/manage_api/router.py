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
