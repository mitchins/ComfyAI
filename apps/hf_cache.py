from __future__ import annotations
import os
import shutil
from typing import Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:  # pragma: no cover - optional
    hf_hub_download = None  # type: ignore
    snapshot_download = None  # type: ignore


def download_file(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a file from Hugging Face with caching."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required")
    cache_dir = os.path.expanduser(cache_dir)
    local_path = os.path.join(cache_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy(downloaded_path, local_path)
    return local_path


def download_repo(repo_id: str, cache_dir: str) -> str:
    """Download an entire repository snapshot if not cached."""
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required")
    cache_dir = os.path.expanduser(cache_dir)
    if not os.path.exists(cache_dir):
        snapshot_download(repo_id=repo_id, local_dir=cache_dir, local_dir_use_symlinks=False)
    return cache_dir
