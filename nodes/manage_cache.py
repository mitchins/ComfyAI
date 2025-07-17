import os
import logging
from typing import List, Dict

try:
    from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
except Exception:  # pragma: no cover - optional dependency
    HfApi = None  # type: ignore
    hf_hub_download = None  # type: ignore
    scan_cache_dir = None  # type: ignore

logger = logging.getLogger(__name__)


def _require_hf() -> None:
    if HfApi is None or hf_hub_download is None:
        raise RuntimeError("huggingface_hub not available")


def list_repo_files(repo_id: str) -> List[Dict[str, int]]:
    """Return files available in a remote repo."""
    _require_hf()
    info = HfApi().repo_info(repo_id, files_metadata=True)
    files = []
    for file_info in getattr(info, "siblings", []):
        path = getattr(file_info, "rfilename", None) or getattr(file_info, "path", None)
        if not path and hasattr(file_info, "name"):
            path = file_info.name
        size = getattr(file_info, "size", 0) or 0
        files.append({"path": path, "size": int(size)})
    return files


def list_cached_entries() -> List[Dict[str, int]]:
    """List cached files in the local HuggingFace cache."""
    if scan_cache_dir is None:
        raise RuntimeError("huggingface_hub not available")
    cache = scan_cache_dir()
    entries: List[Dict[str, int | float | str]] = []
    for repo in cache.repos:
        for rev in repo.revisions:
            for f in rev.files:
                entries.append(
                    {
                        "repo": repo.repo_id,
                        "path": f.file_name,
                        "size": f.size_on_disk,
                        "last_used": f.blob_last_accessed,
                    }
                )
    return entries


def download_file(repo_id: str, file_path: str) -> str:
    """Download a file into the local cache."""
    _require_hf()
    return hf_hub_download(repo_id=repo_id, filename=file_path)


def delete_cached_file(repo_id: str, file_path: str) -> None:
    """Remove a cached file and its blob if present."""
    _require_hf()
    path = hf_hub_download(repo_id=repo_id, filename=file_path, local_files_only=True)
    if os.path.exists(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    if scan_cache_dir is not None:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id != repo_id:
                continue
            for rev in repo.revisions:
                for f in rev.files:
                    if f.file_name == file_path and f.blob_path.exists():
                        try:
                            os.remove(f.blob_path)
                        except FileNotFoundError:
                            pass
