import os
import shutil
from unittest.mock import patch, MagicMock
from apps.shared.hf_utils import download_model


def test_download_model_exists(monkeypatch):
    with patch('os.path.exists', return_value=True):
        with patch('os.makedirs') as mock_makedirs:
            mock_hf_hub_download = MagicMock()
            monkeypatch.setattr('apps.shared.hf_utils.hf_hub_download', mock_hf_hub_download)
            with patch('shutil.copy') as mock_copy:
                result = download_model('repo', 'file.txt', '~/.cache/test')
                assert result == os.path.expanduser('~/.cache/test/file.txt')
                mock_makedirs.assert_called_once()
                mock_hf_hub_download.assert_not_called()
                mock_copy.assert_not_called()

def test_download_model_new(monkeypatch):
    with patch('os.path.exists', return_value=False):
        with patch('os.makedirs') as mock_makedirs:
            mock_hf_hub_download = MagicMock(return_value='/tmp/downloaded_file')
            monkeypatch.setattr('apps.shared.hf_utils.hf_hub_download', mock_hf_hub_download)
            with patch('shutil.copy') as mock_copy:
                result = download_model('repo', 'file.txt', '~/.cache/test')
                assert result == os.path.expanduser('~/.cache/test/file.txt')
                mock_makedirs.assert_called_once()
                mock_hf_hub_download.assert_called_once_with(repo_id='repo', filename='file.txt')
                mock_copy.assert_called_once_with('/tmp/downloaded_file', os.path.expanduser('~/.cache/test/file.txt'))

def test_download_model_hf_not_available(monkeypatch):
    monkeypatch.setattr('apps.shared.hf_utils.hf_hub_download', None)
    try:
        download_model('repo', 'file.txt', '~/.cache/test')
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert str(e) == "huggingface_hub not available"
