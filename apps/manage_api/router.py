import os
import re
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from nodes.manage_cache import (
    list_repo_files,
    list_cached_entries,
    download_file,
    delete_cached_file,
)

router = APIRouter(prefix="", tags=["manage"])


class DownloadRequest(BaseModel):
    repo: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")
    path: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")


class DeleteRequest(BaseModel):
    repo: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")
    path: str = Field(..., pattern=r"^[A-Za-z0-9_\-./]+$")


class FileEntry(BaseModel):
    path: str
    size: int

    class Config:
        json_schema_extra = {"example": {"path": "weights/model.bin", "size": 1234}}


class CacheEntry(BaseModel):
    repo: str
    path: str
    size: int
    last_used: float

    class Config:
        json_schema_extra = {
            "example": {
                "repo": "myrepo",
                "path": "weights/model.bin",
                "size": 1234,
                "last_used": 0.0,
            }
        }


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


@router.get("/repos/{repo_id}/files", response_model=List[FileEntry], tags=["manage"])
async def get_remote_files(repo_id: str):
    try:
        files = list_repo_files(repo_id)
        return files
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache", response_model=List[CacheEntry], tags=["manage"])
async def get_cache():
    entries = list_cached_entries()
    return entries


@router.post(
    "/cache/download", status_code=202, tags=["manage"], response_model=None
)
async def post_download(req: DownloadRequest):
    _validate_paths(req.repo, req.path)
    try:
        download_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/cache", status_code=200, tags=["manage"], response_model=None
)
async def delete_cache(req: DeleteRequest):
    _validate_paths(req.repo, req.path)
    try:
        delete_cached_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))
