import os
from typing import List, Dict

try:
    from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir
except Exception:  # pragma: no cover - optional dependency
    HfApi = None
    hf_hub_download = None
    scan_cache_dir = None


def list_repo_files(repo_id: str) -> List[Dict[str, int]]:
    """Return file listing for a repo."""
    if HfApi is None:
        raise RuntimeError("huggingface_hub not available")
    info = HfApi().repo_info(repo_id)
    return [
        {"path": s.rfilename, "size": getattr(s, "size", 0) or 0}
        for s in getattr(info, "siblings", [])
    ]


def list_cached_entries() -> List[Dict[str, object]]:
    """Return details about cached files."""
    if scan_cache_dir is None:
        raise RuntimeError("huggingface_hub not available")
    scan = scan_cache_dir()
    entries: List[Dict[str, object]] = []
    for repo in scan.repos:
        repo_id = repo.repo_id
        for rev in repo.revisions:
            for f in rev.files:
                entries.append(
                    {
                        "repo": repo_id,
                        "path": f.file_name,
                        "size": f.size_on_disk,
                        "last_used": f.blob_last_accessed,
                    }
                )
    return entries


def download_file(repo_id: str, file_path: str) -> str:
    """Download a file into the local Hugging Face cache."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not available")
    return hf_hub_download(repo_id=repo_id, filename=file_path)


def delete_cached_file(repo_id: str, file_path: str) -> None:
    """Remove a cached file and its blob if present."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not available")
    path = hf_hub_download(repo_id=repo_id, filename=file_path, local_files_only=True)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    if scan_cache_dir is None:
        return
    scan = scan_cache_dir()
    for repo in scan.repos:
        if repo.repo_id != repo_id:
            continue
        for rev in repo.revisions:
            for f in rev.files:
                if f.file_name == file_path:
                    try:
                        os.remove(f.blob_path)
                    except Exception:
                        pass
