import json


def test_dummy_add_one(run_graph):
    graph = {"1": {"class_type": "DummyAddOne", "inputs": {"value": 2}}}
    result = run_graph(json.dumps(graph))
    assert result["outputs"]["1"] == (3,)
