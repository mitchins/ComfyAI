import json


def test_dummy_node_smoke(run_graph, monkeypatch):
    class DummyNode:
        FUNCTION = "run"
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"x": ("INT", {}), "y": ("INT", {})}}

        def run(self, x=0, y=0):
            return (x + y,)

    from comfyai.testing import workflow_runner as wr

    original = wr._resolve_node_class

    def _patched(name):
        if name == "DummyNode":
            return DummyNode
        return original(name)

    monkeypatch.setattr(wr, "_resolve_node_class", _patched)

    graph = {"1": {"class_type": "DummyNode", "inputs": {"x": 2, "y": 3}}}
    result = run_graph(json.dumps(graph))
    assert result["outputs"]["1"] == (5,)
