import json

def test_echo_node(run_graph):
    graph = {
        "1": {
            "class_type": "EchoNode",
            "inputs": {"text": "hello"}
        }
    }
    result = run_graph(json.dumps(graph))
    assert result["outputs"]["1"] == "hello"
