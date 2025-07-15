"""Utilities for running tiny ComfyUI workflows in tests."""

import json
import importlib
import sys
from pathlib import Path


class ExecutionCache:
    """Minimal cache storing node outputs."""

    def __init__(self, base_path: str | None = None):
        self.base_path = base_path
        self._outputs: dict[str, object] = {}

    def set_output(self, node_id: str, value):
        self._outputs[str(node_id)] = value

    def get_output(self, node_id: str):
        return self._outputs.get(str(node_id))


class WorkflowExecutor:
    """Very small executor that instantiates and runs nodes sequentially."""

    def __init__(self, cache: ExecutionCache):
        self.cache = cache
        self.nodes: list[tuple[str, type, dict]] = []

    def instantiate_node(self, node_id: str, class_type: str, inputs: dict):
        cls = _resolve_node_class(class_type)
        self.nodes.append((str(node_id), cls, inputs))

    def run_all(self):
        for node_id, cls, inputs in self.nodes:
            instance = cls() if isinstance(cls, type) else cls
            kwargs = {}
            for name, val in inputs.items():
                if isinstance(val, list) and len(val) == 2:
                    up_id, slot = val
                    out = self.cache.get_output(str(up_id))
                    if isinstance(out, tuple):
                        kwargs[name] = out[slot]
                    else:
                        kwargs[name] = out
                else:
                    kwargs[name] = val
            func_name = getattr(instance, "FUNCTION", "run")
            func = getattr(instance, func_name)
            result = func(**kwargs)
            self.cache.set_output(node_id, result)


def _resolve_node_class(class_type: str):
    """Resolve a node class from custom or built-in modules."""
    # custom nodes
    candidates = [
        "custom_nodes.conditional_save_image",
        "custom_nodes.vllm_query",
        "integration_tests.dummy_nodes",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, class_type):
                return getattr(mod, class_type)
        except Exception:
            continue
    # built-in nodes from cloned ComfyUI if available
    try:
        # Add local ComfyUI repo to path if present
        repo_root = Path(__file__).resolve().parents[2]
        comfy_path = repo_root / "ComfyUI_repo"
        if comfy_path.exists() and str(comfy_path) not in sys.path:
            sys.path.insert(0, str(comfy_path))
        nodes_mod = importlib.import_module("nodes")
        if hasattr(nodes_mod, class_type):
            return getattr(nodes_mod, class_type)
    except Exception:
        pass
    raise KeyError(f"Node class not found: {class_type}")


def run_workflow_from_json(graph_json: str, base_path: str | None = None):
    """Execute a minimal ComfyUI workflow described as JSON."""
    graph = json.loads(graph_json)
    cache = ExecutionCache(base_path=base_path)
    executor = WorkflowExecutor(cache=cache)
    for node_id, spec in graph.items():
        class_type = spec["class_type"]
        inputs = spec.get("inputs", {})
        executor.instantiate_node(node_id, class_type, inputs)
    executor.run_all()
    outputs = {nid: cache.get_output(nid) for nid in graph}
    return {"executor": executor, "outputs": outputs}

