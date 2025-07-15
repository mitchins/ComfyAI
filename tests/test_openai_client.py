from types import SimpleNamespace
import os

os.environ["UNIT_TEST_MODE"] = "1"
from custom_nodes import openai_client
import requests


def test_chat_completion(monkeypatch):
    called = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        called['url'] = url
        called['json'] = json
        response = SimpleNamespace()
        response.json = lambda: {"choices": [{"message": {"content": "ok"}}]}
        response.raise_for_status = lambda: None
        return response

    monkeypatch.setattr(requests, "post", fake_post)
    result = openai_client.chat_completion(
        "http://host", "gpt", [ {"role": "user", "content": "hi"}],
        api_key="k", max_tokens=5, temperature=0.7)

    assert called['url'] == "http://host/v1/chat/completions"
    assert called['json']['max_tokens'] == 5
    assert called['json']['temperature'] == 0.7
    assert result == "ok"
