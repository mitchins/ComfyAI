import json
import pytest


def test_vllm_query_workflow(run_graph, monkeypatch):
    """Run VisionLLMQuery in a mini workflow graph."""
    monkeypatch.setattr("custom_nodes.vllm_query.chat_completion", lambda *a, **k: "ok")
    monkeypatch.setattr("custom_nodes.vllm_query.fuzzy_match_bool", lambda x: True)
    graph = {
        "1": {
            "class_type": "VisionLLMQuery",
            "inputs": {
                "text_query": "hello",
                "api_endpoint": "http://x",
                "api_model": "gpt",
                "api_key": "",
            },
        }
    }
    result = run_graph(json.dumps(graph))
    assert result["outputs"]["1"][1] is True
