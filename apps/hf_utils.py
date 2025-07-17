from __future__ import annotations
import os
import shutil

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None  # type: ignore


def download_model(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a file from Hugging Face with caching.

    Parameters
    ----------
    repo_id: str
        Repository ID on Hugging Face.
    filename: str
        File path within the repository.
    cache_dir: str
        Local directory to copy the file into.

    Returns
    -------
    str
        Path to the downloaded file within ``cache_dir``.
    """
    local_path = os.path.join(cache_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is required to download models")
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy(downloaded_path, local_path)
    return local_path
