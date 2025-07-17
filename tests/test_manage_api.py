import sys
import os
sys.modules.pop("fastapi", None)
os.environ.setdefault("DETECTOR_MODEL", "fake")
os.environ.setdefault("DETECTOR_FILE", "model.onnx")
os.environ.setdefault("EMBEDDER_MODEL_PATH", "fake")
os.environ.setdefault("EMBEDDER_FILE", "model.onnx")
from fastapi.testclient import TestClient

from comfyai.main import app

client = TestClient(app)


def test_get_remote_files(monkeypatch):
    monkeypatch.setattr(
        "apps.manage_api.router.list_repo_files",
        lambda repo_id: [{"path": "f.txt", "size": 1}],
    )
    resp = client.get("/manage/api/manage/repos/test-repo/files")
    assert resp.status_code == 200
    assert resp.json() == {"files": [{"path": "f.txt", "size": 1}]}


def test_get_cache(monkeypatch):
    monkeypatch.setattr(
        "apps.manage_api.router.list_cached_entries",
        lambda: [{"repo": "r", "path": "p", "size": 2, "last_used": 0.0}],
    )
    resp = client.get("/manage/api/manage/cache")
    assert resp.status_code == 200
    data = resp.json()["cache"][0]
    assert data["repo"] == "r" and data["path"] == "p"


def test_post_download(monkeypatch):
    called = {}

    def fake_dl(repo_id, file_path):
        called["args"] = (repo_id, file_path)

    monkeypatch.setattr("apps.manage_api.router.download_file", fake_dl)
    resp = client.post(
        "/manage/api/manage/cache/download",
        json={"repo": "r", "path": "p"},
    )
    assert resp.status_code == 202
    assert called["args"] == ("r", "p")


def test_delete_cache(monkeypatch):
    called = {}

    def fake_del(repo_id, file_path):
        called["args"] = (repo_id, file_path)

    monkeypatch.setattr("apps.manage_api.router.delete_cached_file", fake_del)
    resp = client.request("DELETE", "/manage/api/manage/cache", json={"repo": "r", "path": "p"})
    assert resp.status_code == 200
    assert called["args"] == ("r", "p")
