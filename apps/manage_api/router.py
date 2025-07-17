from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from nodes.manage_cache import (
    list_repo_files,
    list_cached_entries,
    download_file,
    delete_cached_file,
)

router = APIRouter(prefix="/api/manage", tags=["manage"])


class DownloadRequest(BaseModel):
    repo: str
    path: str


class DeleteRequest(BaseModel):
    repo: str
    path: str


@router.get("/repos/{repo_id}/files")
async def get_remote_files(repo_id: str):
    try:
        files = list_repo_files(repo_id)
        return {"files": files}
    except Exception as e:  # pragma: no cover - error case
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def get_cache():
    entries = list_cached_entries()
    return {"cache": entries}


@router.post("/cache/download", status_code=202)
async def post_download(req: DownloadRequest):
    try:
        download_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - error case
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache", status_code=200)
async def delete_cache(req: DeleteRequest):
    try:
        delete_cached_file(repo_id=req.repo, file_path=req.path)
    except Exception as e:  # pragma: no cover - error case
        raise HTTPException(status_code=500, detail=str(e))
