import pytest

try:
    from custom_nodes.vllm_query import VisionLLMQuery
except Exception:  # pragma: no cover - optional node
    VisionLLMQuery = None

if VisionLLMQuery is None:
    pytest.skip("VisionLLMQuery node not available", allow_module_level=True)


def test_contract():
    assert isinstance(VisionLLMQuery.CATEGORY, str)
    input_types = VisionLLMQuery.INPUT_TYPES()
    for name in ("text_query", "api_endpoint", "api_model", "api_key"):
        assert name in input_types["required"]
    assert isinstance(VisionLLMQuery.RETURN_TYPES, tuple)
    assert hasattr(VisionLLMQuery, VisionLLMQuery.FUNCTION)


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
