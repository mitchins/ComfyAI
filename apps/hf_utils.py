import os
import shutil
import logging
from typing import Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None
    snapshot_download = None

logger = logging.getLogger(__name__)


def download_file(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a single file from Hugging Face with caching."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not available")
    local_path = os.path.join(cache_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy(path, local_path)
        logger.info(f"Downloaded {repo_id}/{filename} to {local_path}")
    return local_path


def download_repo(repo_id: str, cache_dir: str) -> str:
    """Download an entire repository snapshot into ``cache_dir``."""
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is not available")
    local_dir = os.path.join(cache_dir, repo_id.replace("/", "_"))
    if not os.path.exists(local_dir):
        temp_dir = snapshot_download(repo_id=repo_id)
        shutil.copytree(temp_dir, local_dir)
        logger.info(f"Downloaded repo {repo_id} to {local_dir}")
    return local_dir
