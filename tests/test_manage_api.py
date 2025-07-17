import sys

sys.modules.pop("fastapi", None)
from fastapi.testclient import TestClient

from comfyai.main import app
import apps.manage_api.router as manage_router

client = TestClient(app)


def test_get_remote_files(monkeypatch):
    monkeypatch.setattr(
        manage_router,
        "list_repo_files",
        lambda repo: [{"path": "file.txt", "size": 123}],
    )
    resp = client.get("/manage/api/manage/repos/myrepo/files")
    assert resp.status_code == 200
    assert resp.json() == {"files": [{"path": "file.txt", "size": 123}]}


def test_get_cache(monkeypatch):
    monkeypatch.setattr(
        manage_router,
        "list_cached_entries",
        lambda: [{"repo": "r", "path": "p", "size": 1, "last_used": 0.0}],
    )
    resp = client.get("/manage/api/manage/cache")
    assert resp.status_code == 200
    assert resp.json() == {
        "cache": [{"repo": "r", "path": "p", "size": 1, "last_used": 0.0}]
    }


def test_post_download(monkeypatch):
    calls = {}

    def fake_download(repo_id, file_path):
        calls["repo"] = repo_id
        calls["path"] = file_path

    monkeypatch.setattr(manage_router, "download_file", fake_download)
    resp = client.post(
        "/manage/api/manage/cache/download", json={"repo": "r", "path": "p"}
    )
    assert resp.status_code == 202
    assert calls == {"repo": "r", "path": "p"}


def test_delete_cache(monkeypatch):
    calls = {}

    def fake_delete(repo_id, file_path):
        calls["repo"] = repo_id
        calls["path"] = file_path

    monkeypatch.setattr(manage_router, "delete_cached_file", fake_delete)
    resp = client.request(
        "DELETE", "/manage/api/manage/cache", json={"repo": "r", "path": "p"}
    )
    assert resp.status_code == 200
    assert calls == {"repo": "r", "path": "p"}
