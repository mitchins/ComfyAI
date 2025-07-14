import requests
from openai_client import chat_completion


def test_chat_completion(monkeypatch):
    sent = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        sent['url'] = url
        sent['headers'] = headers
        sent['json'] = json
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}
        return Resp()
    monkeypatch.setattr(requests, 'post', fake_post)
    result = chat_completion('http://host', 'model', [
        {'role': 'user', 'content': 'hi'}], api_key='k', max_tokens=5)
    assert sent['url'] == 'http://host/v1/chat/completions'
    assert sent['headers']['Authorization'] == 'Bearer k'
    assert sent['json']['max_tokens'] == 5
    assert result == 'ok'

