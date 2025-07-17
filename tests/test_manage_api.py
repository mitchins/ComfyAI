import os
import sys

# Ensure required env vars so face_api can import
os.environ.setdefault("DETECTOR_MODEL", "fake")
os.environ.setdefault("DETECTOR_FILE", "model.onnx")
os.environ.setdefault("EMBEDDER_MODEL_PATH", "fake")
os.environ.setdefault("EMBEDDER_FILE", "model.onnx")

sys.modules.pop("fastapi", None)
from fastapi.testclient import TestClient

from comfyai.main import app

client = TestClient(app)


def test_get_remote_files(monkeypatch):
    monkeypatch.setattr("apps.manage_api.router.list_repo_files", lambda repo: [{"path": "f.txt", "size": 123}])
    resp = client.get("/manage/api/manage/repos/test/files")
    assert resp.status_code == 200
    assert resp.json() == {"files": [{"path": "f.txt", "size": 123}]}


def test_get_cache(monkeypatch):
    monkeypatch.setattr("apps.manage_api.router.list_cached_entries", lambda: [{"repo": "r", "path": "f.txt", "size": 1, "last_used": 0.0}])
    resp = client.get("/manage/api/manage/cache")
    assert resp.status_code == 200
    assert resp.json() == {"cache": [{"repo": "r", "path": "f.txt", "size": 1, "last_used": 0.0}]}


def test_download(monkeypatch):
    called = {}
    def fake(repo_id, file_path):
        called["repo"] = repo_id
        called["path"] = file_path
    monkeypatch.setattr("apps.manage_api.router.download_file", fake)
    resp = client.post("/manage/api/manage/cache/download", json={"repo": "r", "path": "p"})
    assert resp.status_code == 202
    assert called == {"repo": "r", "path": "p"}


def test_delete(monkeypatch):
    called = {}
    def fake(repo_id, file_path):
        called["repo"] = repo_id
        called["path"] = file_path
    monkeypatch.setattr("apps.manage_api.router.delete_cached_file", fake)
    resp = client.request("DELETE", "/manage/api/manage/cache", json={"repo": "r", "path": "p"})
    assert resp.status_code == 200
    assert called == {"repo": "r", "path": "p"}
