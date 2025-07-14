import pytest
import vllm_query


def test_contract():
    assert isinstance(vllm_query.VisionLLMQuery.CATEGORY, str)
    inputs = vllm_query.VisionLLMQuery.INPUT_TYPES()
    for name in ("text_query", "api_endpoint", "api_model", "api_key"):
        assert name in inputs["required"]
    assert isinstance(vllm_query.VisionLLMQuery.RETURN_TYPES, tuple)
    assert hasattr(vllm_query.VisionLLMQuery, vllm_query.VisionLLMQuery.FUNCTION)


def test_run_minimal(dummy_image, monkeypatch):
    node = vllm_query.VisionLLMQuery()

    monkeypatch.setattr(vllm_query, "chat_completion", lambda *a, **k: "yes")
    monkeypatch.setattr(vllm_query, "fuzzy_match_bool", lambda x: True)
    monkeypatch.setattr(vllm_query, "image_to_bytes", lambda img: b"img")

    result = node.run(
        api_endpoint="http://x",
        api_model="gpt",
        api_key="",
        text_query="hello",
        image=dummy_image,
    )
    assert isinstance(result, tuple) and len(result) == 3
    assert result[0] == "yes"
    assert result[1] is True
    assert result[2] == 1
