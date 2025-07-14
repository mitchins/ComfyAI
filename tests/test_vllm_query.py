import vllm_query


def test_two_image_message(monkeypatch):
    captured = {}

    def fake_chat(endpoint, model, messages, api_key=None):
        captured['messages'] = messages
        return "ok"

    monkeypatch.setattr(vllm_query, "chat_completion", fake_chat)
    monkeypatch.setattr(vllm_query, "image_to_bytes", lambda img: b"data")

    node = vllm_query.VisionLLMQuery()
    result = node.run(
        api_endpoint="http://host",
        api_model="gpt",
        api_key="k",
        text_query="hello",
        image="img1",
        reference_image="img2",
    )

    msgs = captured['messages'][0]['content']
    assert len(msgs) == 3
    assert msgs[1]['type'] == 'image_url'
    assert msgs[2]['type'] == 'image_url'
    # ensure function returns expected tuple
    assert result[0] == "ok"

