import os
import logging
import shutil
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None  # type: ignore

logger = logging.getLogger(__name__)


def download_model(repo_id: str, filename: str, cache_subdir: str) -> str:
    """Download a file from Hugging Face with local caching."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not available")

    cache_dir = os.path.expanduser(cache_subdir)
    local_path = os.path.join(cache_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy(downloaded_path, local_path)
        logger.info(f"Downloaded {repo_id}/{filename} to {local_path}")
    return local_path
