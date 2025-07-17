import os
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from nodes.manage_cache import (
    list_repo_files,
    list_cached_entries,
    download_file,
    delete_cached_file,
)

router = APIRouter(prefix="/manage")

_VALID_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

def _validate(repo: str, path: str | None = None) -> None:
    if not _VALID_RE.fullmatch(repo):
        raise HTTPException(status_code=400, detail="Invalid repo")
    if path is not None:
        if not _VALID_RE.fullmatch(path):
            raise HTTPException(status_code=400, detail="Invalid path")
        base = os.path.realpath(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
        target = os.path.realpath(os.path.join(base, "hub", repo, path))
        if not target.startswith(os.path.join(base, "hub")):
            raise HTTPException(status_code=400, detail="Path outside cache")


class RepoFile(BaseModel):
    path: str
    size: int

    class Config:
        json_schema_extra = {"example": {"path": "f.txt", "size": 1}}


class CacheEntry(BaseModel):
    repo: str
    path: str
    size: int
    last_used: float

    class Config:
        json_schema_extra = {
            "example": {"repo": "r", "path": "f.txt", "size": 1, "last_used": 0.0}
        }


class DownloadRequest(BaseModel):
    repo: str
    path: str


class DeleteRequest(BaseModel):
    repo: str
    path: str


@router.get("/repos/{repo_id}/files", tags=["manage"], response_model=List[RepoFile])
async def get_remote_files(repo_id: str):
    _validate(repo_id)
    try:
        files = list_repo_files(repo_id)
        return files
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache", tags=["manage"], response_model=List[CacheEntry])
async def get_cache():
    entries = list_cached_entries()
    return entries


@router.post("/cache/download", status_code=202, tags=["manage"])
async def post_download(req: DownloadRequest):
    _validate(req.repo, req.path)
    try:
        download_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache", status_code=200, tags=["manage"])
async def delete_cache(req: DeleteRequest):
    _validate(req.repo, req.path)
    try:
        delete_cached_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - pass through
        raise HTTPException(status_code=500, detail=str(e))

