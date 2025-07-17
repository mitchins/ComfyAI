import os
from apps import hf_utils


def test_download_file(monkeypatch, tmp_path):
    called = {}

    def fake_download(repo_id, filename):
        called['args'] = (repo_id, filename)
        p = tmp_path / "src"
        p.write_text("x")
        return str(p)

    monkeypatch.setattr(hf_utils, "hf_hub_download", fake_download)
    dest = hf_utils.download_file("repo", "file.txt", tmp_path)
    assert called['args'] == ("repo", "file.txt")
    assert os.path.exists(dest)


def test_download_repo(monkeypatch, tmp_path):
    called = {}

    def fake_snapshot(repo_id):
        called['repo'] = repo_id
        src = tmp_path / "snap"
        src.mkdir()
        (src / "a.txt").write_text("x")
        return str(src)

    monkeypatch.setattr(hf_utils, "snapshot_download", fake_snapshot)
    path = hf_utils.download_repo("repo", tmp_path)
    assert called['repo'] == "repo"
    assert os.path.isdir(path)
    assert os.path.exists(os.path.join(path, "a.txt"))
