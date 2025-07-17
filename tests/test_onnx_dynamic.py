import os
import types

import apps.onnx_chat.main as chat


def test_load_session_download(monkeypatch, tmp_path):
    called = {}

    def fake_download(repo_id, filename, cache_dir):
        called['args'] = (repo_id, filename, cache_dir)
        path = tmp_path / filename
        path.write_text('x')
        return str(path)

    class FakeORT:
        def InferenceSession(self, path):
            called['path'] = path
            return f'session:{path}'

    monkeypatch.setattr(chat, 'download_model', fake_download)
    monkeypatch.setattr(chat, 'ort', FakeORT())
    chat.sessions.clear()

    sess = chat.load_session('repo/model:onx.onnx')
    assert called['args'][0] == 'repo/model'
    assert called['args'][1] == 'onx.onnx'
    assert 'onnx_chat' in called['args'][2]
    assert sess == f'session:{tmp_path/"onx.onnx"}'

