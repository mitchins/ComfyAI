import requests
from openai_client import chat_completion

class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_chat_completion_builds_request(monkeypatch):
    recorded = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        recorded['url'] = url
        recorded['headers'] = headers
        recorded['json'] = json
        return DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(requests, 'post', fake_post)
    result = chat_completion(
        "http://server", 'gpt', [{"role": "user", "content": "hi"}],
        api_key='KEY', max_tokens=5, temperature=0.2
    )
    assert recorded['url'].endswith('/v1/chat/completions')
    assert recorded['headers']["Authorization"] == "Bearer KEY"
    assert recorded['json']["max_tokens"] == 5
    assert recorded['json']["temperature"] == 0.2
    assert result == 'ok'
