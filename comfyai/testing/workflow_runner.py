import json
from typing import Any, Dict

# Try to import ComfyUI executor classes if available
try:
    from comfyui.workflow_executor import WorkflowExecutor, ExecutionCache  # type: ignore
except Exception:  # pragma: no cover - fallback to simple stubs
    import importlib

    def _resolve_node_class(class_type: str):
        """Find node class from either `nodes` (ComfyUI) or `custom_nodes`."""
        for mod_name in ("nodes", "custom_nodes"):
            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, "NODE_CLASS_MAPPINGS", {}).get(class_type)
                if cls is not None:
                    return cls
                if hasattr(mod, class_type):
                    return getattr(mod, class_type)
            except Exception:
                continue
        # Scan submodules of custom_nodes for the class
        try:
            import pkgutil
            import custom_nodes as cn
            for info in pkgutil.iter_modules(cn.__path__):
                try:
                    sub = importlib.import_module(f"custom_nodes.{info.name}")
                    if hasattr(sub, class_type):
                        return getattr(sub, class_type)
                except Exception:
                    continue
        except Exception:
            pass
        raise KeyError(f"Unknown node class: {class_type}")

    class ExecutionCache:
        def __init__(self, base_path: str | None = None):
            self.base_path = base_path
            self._outputs: Dict[str, Any] = {}

        def set_output(self, node_id: str, value: Any) -> None:
            self._outputs[node_id] = value

        def get_output(self, node_id: str) -> Any:
            return self._outputs.get(node_id)

    class WorkflowExecutor:
        def __init__(self, cache: ExecutionCache | None = None):
            self.cache = cache or ExecutionCache()
            self.nodes: Dict[str, Dict[str, Any]] = {}
            self.order: list[str] = []

        def instantiate_node(self, node_id: str, class_type: str, inputs: Dict[str, Any]):
            NodeClass = _resolve_node_class(class_type)
            self.nodes[node_id] = {
                "instance": NodeClass(),
                "inputs": inputs,
            }
            self.order.append(node_id)

        def _resolve_input(self, val):
            if isinstance(val, list) and len(val) == 2:
                src, idx = val
                out = self.cache.get_output(str(src))
                if isinstance(out, (list, tuple)):
                    return out[idx]
                return out
            return val

        def run_all(self):
            for node_id in self.order:
                node = self.nodes[node_id]
                inst = node["instance"]
                inputs = {k: self._resolve_input(v) for k, v in node["inputs"].items()}
                func_name = getattr(inst, "FUNCTION", "run")
                func = getattr(inst, func_name)
                output = func(**inputs)
                self.cache.set_output(node_id, output)



def run_workflow_from_json(graph_json: str, base_path: str | None = None) -> Dict[str, Any]:
    """Execute a mini ComfyUI graph headlessly and return outputs."""
    graph = json.loads(graph_json)
    cache = ExecutionCache(base_path=base_path)
    executor = WorkflowExecutor(cache=cache)
    for node_id, spec in graph.items():
        class_type = spec["class_type"]
        inputs = spec.get("inputs", {})
        executor.instantiate_node(str(node_id), class_type, inputs)
    executor.run_all()
    return {
        "executor": executor,
        "outputs": {nid: cache.get_output(nid) for nid in graph},
    }
