import os, sys
sys.modules.pop("fastapi", None)
from fastapi.testclient import TestClient

os.environ.setdefault("DETECTOR_MODEL", "fake")
os.environ.setdefault("DETECTOR_FILE", "model.onnx")
os.environ.setdefault("EMBEDDER_MODEL_PATH", "fake")
os.environ.setdefault("EMBEDDER_FILE", "model.onnx")

from comfyai.main import app
import apps.manage_api.router as manage_router

client = TestClient(app)


def test_get_remote_files(monkeypatch):
    monkeypatch.setattr(manage_router, "list_repo_files", lambda repo: [{"path": "f", "size": 1}])
    resp = client.get("/manage/api/manage/repos/test/files")
    assert resp.status_code == 200
    assert resp.json() == {"files": [{"path": "f", "size": 1}]}


def test_get_cache(monkeypatch):
    monkeypatch.setattr(manage_router, "list_cached_entries", lambda: [{"repo": "r", "path": "p", "size": 1, "last_used": 0.0}])
    resp = client.get("/manage/api/manage/cache")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cache"][0]["repo"] == "r"


def test_post_download(monkeypatch):
    called = {}

    def fake_dl(repo_id, file_path):
        called["repo"] = repo_id
        called["path"] = file_path

    monkeypatch.setattr(manage_router, "download_file", fake_dl)
    resp = client.post("/manage/api/manage/cache/download", json={"repo": "r", "path": "x"})
    assert resp.status_code == 202
    assert called == {"repo": "r", "path": "x"}


def test_delete_cache(monkeypatch):
    called = {}

    def fake_del(repo_id, file_path):
        called["repo"] = repo_id
        called["path"] = file_path

    monkeypatch.setattr(manage_router, "delete_cached_file", fake_del)
    resp = client.request("DELETE", "/manage/api/manage/cache", json={"repo": "r", "path": "y"})
    assert resp.status_code == 200
    assert called == {"repo": "r", "path": "y"}
