import json


def test_vision_llm_query_headless(run_graph, monkeypatch):
    import nodes.vllm_query as vllm_query

    called = {}

    def fake_chat(endpoint, model, messages, api_key=None, timeout=30, **kw):
        called['messages'] = messages
        return "yes"

    monkeypatch.setattr(vllm_query, "chat_completion", fake_chat)
    monkeypatch.setattr(vllm_query, "fuzzy_match_bool", lambda x: True)

    graph = {
        "1": {
            "class_type": "VLLMTextQuery",
            "inputs": {
                "text_query": "hello",
                "api_endpoint": "http://host",
                "api_model": "gpt",
                "api_key": "",
            },
        }
    }

    result = run_graph(json.dumps(graph))
    outputs = result["outputs"]
    assert "1" in outputs
    out = outputs["1"]
    assert isinstance(out, tuple)
    assert out[0] == "yes"
