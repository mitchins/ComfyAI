import os
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from nodes.manage_cache import (
    list_repo_files,
    list_cached_entries,
    download_file,
    delete_cached_file,
)

router = APIRouter(prefix="", tags=["manage"])


allowed_pattern = re.compile(r"^[A-Za-z0-9_\-./]+$")


class RemoteFile(BaseModel):
    path: str
    size: int


class FileListResponse(BaseModel):
    files: list[RemoteFile]

    class Config:
        schema_extra = {
            "example": {"files": [{"path": "model.bin", "size": 123456}]}
        }


class CacheEntry(BaseModel):
    repo: str
    path: str
    size: int
    last_used: float


class CacheListResponse(BaseModel):
    cache: list[CacheEntry]

    class Config:
        schema_extra = {
            "example": {
                "cache": [
                    {
                        "repo": "user/my-model",
                        "path": "model.bin",
                        "size": 123456,
                        "last_used": 0.0,
                    }
                ]
            }
        }


def _sanitize(repo_id: str, file_path: str) -> None:
    if not allowed_pattern.fullmatch(repo_id) or not allowed_pattern.fullmatch(file_path):
        raise HTTPException(status_code=400, detail="Invalid characters")
    base = os.path.realpath(HUGGINGFACE_HUB_CACHE)
    real = os.path.realpath(os.path.join(base, repo_id, file_path))
    if not real.startswith(base):
        raise HTTPException(status_code=400, detail="Invalid path")


def _sanitize_repo(repo_id: str) -> None:
    if not allowed_pattern.fullmatch(repo_id):
        raise HTTPException(status_code=400, detail="Invalid characters")


class DownloadRequest(BaseModel):
    repo: str
    path: str

    class Config:
        schema_extra = {
            "example": {"repo": "user/my-model", "path": "model.bin"}
        }


class DeleteRequest(BaseModel):
    repo: str
    path: str

    class Config:
        schema_extra = {
            "example": {"repo": "user/my-model", "path": "model.bin"}
        }


@router.get("/repos/{repo_id}/files", response_model=FileListResponse, tags=["manage"])
async def get_remote_files(repo_id: str):
    _sanitize_repo(repo_id)
    try:
        files = list_repo_files(repo_id)
        return {"files": files}
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache", response_model=CacheListResponse, tags=["manage"])
async def get_cache():
    entries = list_cached_entries()
    return {"cache": entries}


@router.post("/cache/download", status_code=202, tags=["manage"], response_model=None)
async def post_download(req: DownloadRequest):
    _sanitize(req.repo, req.path)
    try:
        download_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache", status_code=200, tags=["manage"], response_model=None)
async def delete_cache(req: DeleteRequest):
    _sanitize(req.repo, req.path)
    try:
        delete_cached_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))
