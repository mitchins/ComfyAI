import json
from comfyai.testing import workflow_runner


def test_dummy_node(monkeypatch, run_graph):
    class DummyNode:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        def run(self, x=0):
            return x + 1

    orig_resolve = workflow_runner._resolve_node_class

    def fake_resolve(name):
        if name == "DummyNode":
            return DummyNode
        return orig_resolve(name)

    monkeypatch.setattr(workflow_runner, "_resolve_node_class", fake_resolve)

    graph = {"1": {"class_type": "DummyNode", "inputs": {"x": 2}}}
    result = run_graph(json.dumps(graph))
    assert result["outputs"]["1"] == 3
