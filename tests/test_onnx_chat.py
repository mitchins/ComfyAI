import importlib
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _load_app(monkeypatch):
    sys.modules.pop('apps.onnx_chat.main', None)
    import apps.onnx_chat.main as chat_main
    importlib.reload(chat_main)
    monkeypatch.setattr(chat_main, 'download_repo', lambda r, c: c)
    monkeypatch.setattr(chat_main, 'AutoTokenizer', SimpleNamespace(from_pretrained=lambda p: 'tok'))
    monkeypatch.setattr(chat_main, 'AutoModelForCausalLM', SimpleNamespace(from_pretrained=lambda p: 'model'))
    calls = {}

    class FakePipe:
        def __call__(self, text, max_new_tokens=16):
            calls['text'] = text
            calls['tokens'] = max_new_tokens
            return [{'generated_text': text[::-1]}]

    def fake_pipeline(task, model=None, tokenizer=None):
        calls['task'] = task
        return FakePipe()

    monkeypatch.setattr(chat_main, 'pipeline', fake_pipeline)
    return TestClient(chat_main.app), calls


def test_dynamic_model_loading(monkeypatch):
    client, calls = _load_app(monkeypatch)
    resp = client.post(
        '/v1/chat/completions',
        json={'model': 'hf/test', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 5}
    )
    assert resp.status_code == 200
    assert resp.json()['choices'][0]['message']['content'] == 'ih'
    assert calls['task'] == 'text-generation'
    assert calls['text'] == 'hi'
    assert calls['tokens'] == 5
