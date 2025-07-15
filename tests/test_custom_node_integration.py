import json


def test_vllm_query_in_workflow(run_graph, monkeypatch):
    """Ensure VisionLLMQuery executes through the workflow runner."""
    monkeypatch.setattr("custom_nodes.vllm_query.chat_completion", lambda *a, **k: "ok")

    graph = {
        "1": {
            "class_type": "VisionLLMQuery",
            "inputs": {
                "text_query": "hello",
                "api_endpoint": "http://localhost",
                "api_model": "test-model",
                "api_key": "",
            },
        }
    }
    result = run_graph(json.dumps(graph))
    outputs = result["outputs"]
    assert "1" in outputs
    out = outputs["1"]
    assert isinstance(out, tuple)
    assert out[0] == "ok"
