import vllm_query


def test_run_with_two_images(monkeypatch):
    node = vllm_query.VisionLLMQuery()

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

