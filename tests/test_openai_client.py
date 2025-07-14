from unittest.mock import Mock, patch
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai_client import chat_completion


def test_chat_completion_builds_request():
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "hello"}}]
    }
    with patch("requests.post", return_value=mock_resp) as post:
        result = chat_completion(
            "http://localhost:1234/",
            "gpt",
            [{"role": "user", "content": "hi"}],
            api_key="KEY",
            max_tokens=5,
            temperature=0.7,
        )
        post.assert_called_once()
        url = post.call_args.args[0]
        assert url == "http://localhost:1234/v1/chat/completions"
        json_payload = post.call_args.kwargs["json"]
        assert json_payload["max_tokens"] == 5
        assert json_payload["temperature"] == 0.7
        headers = post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer KEY"
        assert result == "hello"
