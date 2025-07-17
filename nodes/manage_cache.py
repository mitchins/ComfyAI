"""Helpers for interacting with cached HuggingFace files."""
from __future__ import annotations

from typing import List, Dict


def list_repo_files(repo_id: str) -> List[Dict[str, int]]:
    """Return file listing for ``repo_id``.

    Placeholder implementation used for unit tests."""
    raise NotImplementedError("list_repo_files is not implemented")


def list_cached_entries() -> List[Dict[str, object]]:
    """Return list of cached files."""
    return []


def download_file(repo_id: str, file_path: str) -> None:
    """Download ``file_path`` from ``repo_id`` into cache."""
    raise NotImplementedError("download_file is not implemented")


def delete_cached_file(repo_id: str, file_path: str) -> None:
    """Delete the cached file."""
    raise NotImplementedError("delete_cached_file is not implemented")
