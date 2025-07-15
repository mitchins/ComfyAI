import pytest
from custom_nodes.vllm_query import VisionLLMQuery
import custom_nodes.vllm_query as vllm_query


def test_contract():
    cls = VisionLLMQuery
    assert isinstance(cls.CATEGORY, str)
    in_types = cls.INPUT_TYPES()
    for name in ("text_query", "api_endpoint", "api_model", "api_key"):
        assert name in in_types["required"]
    assert isinstance(cls.RETURN_TYPES, tuple)
    assert hasattr(cls, cls.FUNCTION)


def test_run_minimal(dummy_image, monkeypatch):
    node = VisionLLMQuery()
    monkeypatch.setattr("custom_nodes.vllm_query.chat_completion", lambda *a, **k: "yes")
    monkeypatch.setattr("custom_nodes.vllm_query.fuzzy_match_bool", lambda x: True)
    monkeypatch.setattr("custom_nodes.vllm_query.image_to_bytes", lambda img: b"img")
    result = node.run(
        api_endpoint="http://x",
        api_model="gpt",
        text_query="hello",
        image=dummy_image,
    )
    assert isinstance(result, tuple) and len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], bool)
    assert isinstance(result[2], int)


def test_run_with_two_images(monkeypatch):
    node = VisionLLMQuery()

    called = {}

    def fake_chat(endpoint, model, messages, api_key=None, timeout=30, **kw):
        called["messages"] = messages
        return "yes"

    monkeypatch.setattr(vllm_query, "chat_completion", fake_chat)
    monkeypatch.setattr(vllm_query, "fuzzy_match_bool", lambda x: True)
    monkeypatch.setattr(vllm_query, "image_to_bytes", lambda img: b"img")

    result = node.run(
        api_endpoint="http://x",
        api_model="gpt",
        api_key="",
        text_query="hello",
        image="i1",
        reference_image="i2",
    )

    msgs = called["messages"][0]["content"]
    assert len(msgs) == 3
    assert msgs[0]["type"] == "text"
    assert msgs[1]["type"] == "image_url"
    assert msgs[2]["type"] == "image_url"
    assert result[1] is True

