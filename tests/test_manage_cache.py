import os
from unittest.mock import patch, MagicMock
from huggingface_hub.errors import CacheNotFound
from apps.shared.manage_cache import (
    list_repo_files,
    list_cached_entries,
    download_file,
    delete_cached_file,
)

# Mock HuggingFace Hub components
class MockHfApi:
    def repo_info(self, repo_id, files_metadata):
        mock_file = MagicMock()
        mock_file.rfilename = "test_file.txt"
        mock_file.size = 100
        mock_info = MagicMock()
        mock_info.siblings = [mock_file]
        return mock_info

def mock_hf_hub_download(*args, **kwargs):
    return "/mock/download/path"

class MockCacheRepo:
    def __init__(self, repo_id, revisions):
        self.repo_id = repo_id
        self.revisions = revisions

class MockCacheRevision:
    def __init__(self, files):
        self.files = files

class MockCacheFile:
    def __init__(self, file_name, size_on_disk, blob_last_accessed, file_path, blob_path):
        self.file_name = file_name
        self.size_on_disk = size_on_disk
        self.blob_last_accessed = blob_last_accessed
        self.file_path = file_path
        self.blob_path = blob_path

class MockCacheDir:
    def __init__(self, repos):
        self.repos = {repo.repo_id: repo for repo in repos}

# Tests for list_repo_files
def test_list_repo_files_hf_not_available(monkeypatch):
    monkeypatch.setattr("apps.shared.manage_cache.HfApi", None)
    try:
        list_repo_files("test_repo")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert str(e) == "huggingface_hub not available"

def test_list_repo_files_success(monkeypatch):
    monkeypatch.setattr("apps.shared.manage_cache.HfApi", MockHfApi)
    result = list_repo_files("test_repo")
    assert result == [{"path": "test_file.txt", "size": 100}]

# Tests for list_cached_entries
def test_list_cached_entries_cache_not_found(monkeypatch):
    monkeypatch.setattr("apps.shared.manage_cache.scan_cache_dir", MagicMock(side_effect=CacheNotFound("test message", "test_cache_dir")))
    monkeypatch.setattr("apps.shared.manage_cache.HfApi", MagicMock())
    monkeypatch.setattr("apps.shared.manage_cache.hf_hub_download", MagicMock())
    result = list_cached_entries()
    assert result == []

def test_list_cached_entries_success_no_onnx(monkeypatch):
    mock_file = MockCacheFile("test_file.txt", 100, 0.0, "/path/to/file.txt", "/path/to/blob.txt")
    mock_rev = MockCacheRevision([mock_file])
    mock_repo = MockCacheRepo("test_repo", [mock_rev])
    mock_cache_dir = MockCacheDir([mock_repo])
    monkeypatch.setattr("apps.shared.manage_cache.scan_cache_dir", MagicMock(return_value=mock_cache_dir))
    monkeypatch.setattr("apps.shared.manage_cache.HfApi", MagicMock())
    monkeypatch.setattr("apps.shared.manage_cache.hf_hub_download", MagicMock())
    result = list_cached_entries()
    expected = [{'repo': 'test_repo', 'path': 'test_file.txt', 'size': 100, 'last_used': 0.0, 'framework': None, 'kind': None, 'inputs': []}]
    assert result == expected

def test_list_cached_entries_success_with_onnx(monkeypatch):
    mock_onnx_model = MagicMock()
    mock_onnx_model.graph.input = []
    mock_onnx_model.metadata_props = []
    monkeypatch.setattr("onnx.load_model", MagicMock(return_value=mock_onnx_model))
    mock_file = MockCacheFile("test_model.onnx", 200, 1.0, "/path/to/model.onnx", "/path/to/blob.onnx")
    mock_rev = MockCacheRevision([mock_file])
    mock_repo = MockCacheRepo("test_repo", [mock_rev])
    mock_cache_dir = MockCacheDir([mock_repo])
    monkeypatch.setattr("apps.shared.manage_cache.scan_cache_dir", MagicMock(return_value=mock_cache_dir))
    monkeypatch.setattr("apps.shared.manage_cache.HfApi", MagicMock())
    monkeypatch.setattr("apps.shared.manage_cache.hf_hub_download", MagicMock())
    result = list_cached_entries()
    expected = [{'repo': 'test_repo', 'path': 'test_model.onnx', 'size': 200, 'last_used': 1.0, 'framework': None, 'kind': 'unknown', 'inputs': []}]
    assert result == expected

# Tests for download_file
def test_download_file_hf_not_available(monkeypatch):
    monkeypatch.setattr("apps.shared.manage_cache.hf_hub_download", None)
    try:
        download_file("test_repo", "test_file.txt")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert str(e) == "huggingface_hub not available"

def test_download_file_success(monkeypatch):
    with patch("apps.shared.manage_cache.hf_hub_download") as mock_hf_hub_download:
        monkeypatch.setattr("apps.shared.manage_cache.HfApi", MagicMock())
        download_file("test_repo", "test_file.txt")
        mock_hf_hub_download.assert_called_once_with(repo_id="test_repo", filename="test_file.txt")

# Tests for delete_cached_file
def test_delete_cached_file_cache_not_found(monkeypatch):
    monkeypatch.setattr("apps.shared.manage_cache.scan_cache_dir", MagicMock(side_effect=CacheNotFound("test message", "test_cache_dir")))
    monkeypatch.setattr("apps.shared.manage_cache.HfApi", MagicMock())
    monkeypatch.setattr("apps.shared.manage_cache.hf_hub_download", MagicMock())
    delete_cached_file("test_repo", "test_file.txt")
    # No error should be raised

def test_delete_cached_file_success(monkeypatch):
    mock_file = MockCacheFile("test_file.txt", 100, 0.0, "/path/to/file.txt", "/path/to/blob.txt")
    mock_rev = MockCacheRevision([mock_file])
    mock_repo = MockCacheRepo("test_repo", [mock_rev])
    mock_cache_dir = MockCacheDir([mock_repo])
    monkeypatch.setattr("apps.shared.manage_cache.scan_cache_dir", MagicMock(return_value=mock_cache_dir))
    mock_try_delete_path = MagicMock()
    monkeypatch.setattr("apps.shared.manage_cache._try_delete_path", mock_try_delete_path)
    delete_cached_file("test_repo", "test_file.txt")
    mock_try_delete_path.assert_any_call("/path/to/file.txt")
    mock_try_delete_path.assert_any_call("/path/to/blob.txt")
