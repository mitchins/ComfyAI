import os
from types import SimpleNamespace
import sys
sys.modules.pop("fastapi", None)
from fastapi.testclient import TestClient

os.environ.setdefault("ONNX_MODEL_PATH", "")

import apps.onnx_chat.main as chat_main
from apps import hf_utils

client = TestClient(chat_main.app)


def test_dynamic_load(monkeypatch, tmp_path):
    called = {}

    def fake_download(repo_id, filename, cache_dir):
        called['repo_id'] = repo_id
        called['filename'] = filename
        called['cache_dir'] = cache_dir
        path = tmp_path / filename
        path.touch()
        return str(path)

    class FakeSession:
        def __init__(self, path):
            called['path'] = path
        def get_inputs(self):
            class Inp: pass
            inp = Inp(); inp.name = 'in'
            return [inp]
        def run(self, *_):
            return [[123]]

    monkeypatch.setattr(hf_utils, 'download_model', fake_download)
    monkeypatch.setattr(chat_main, 'download_model', fake_download)
    monkeypatch.setattr(chat_main, 'ort', SimpleNamespace(InferenceSession=FakeSession))
    chat_main._SESSIONS.clear()

    resp = client.post('/v1/chat/completions', json={
        'model': 'repo/model.onnx',
        'messages': [{'role': 'user', 'content': 'hi'}]
    })
    assert resp.status_code == 200
    assert resp.json()['choices'][0]['message']['content'] == '123'
    assert called['repo_id'] == 'repo'
    assert called['filename'] == 'model.onnx'

    # second request should reuse cached session
    called.clear()
    resp = client.post('/v1/chat/completions', json={
        'model': 'repo/model.onnx',
        'messages': [{'role': 'user', 'content': 'bye'}]
    })
    assert resp.status_code == 200
    assert called == {}
