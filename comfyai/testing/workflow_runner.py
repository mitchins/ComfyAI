import json
import pkgutil
import importlib

NODE_CLASS_REGISTRY = {}


def register_node_class(class_type: str, cls):
    """Register a node class for use in headless workflows."""
    NODE_CLASS_REGISTRY[class_type] = cls


def _discover_class(class_type: str):
    # search registry first
    if class_type in NODE_CLASS_REGISTRY:
        return NODE_CLASS_REGISTRY[class_type]
    # try custom_nodes package modules
    try:
        import custom_nodes
        for info in pkgutil.iter_modules(custom_nodes.__path__):
            mod = importlib.import_module(f"custom_nodes.{info.name}")
            if hasattr(mod, class_type):
                cls = getattr(mod, class_type)
                NODE_CLASS_REGISTRY[class_type] = cls
                return cls
    except Exception:
        pass
    # try built-in nodes module if present
    try:
        import nodes
        if hasattr(nodes, class_type):
            cls = getattr(nodes, class_type)
            NODE_CLASS_REGISTRY[class_type] = cls
            return cls
    except Exception:
        pass
    raise KeyError(f"Node class {class_type} not found")


class ExecutionCache:
    def __init__(self, base_path: str | None = None):
        self.base_path = base_path
        self._outputs: dict[str, tuple] = {}

    def set_output(self, node_id: str, output):
        self._outputs[node_id] = output

    def get_output(self, node_id: str):
        return self._outputs.get(node_id)


class WorkflowExecutor:
    def __init__(self, cache: ExecutionCache | None = None):
        self.cache = cache or ExecutionCache()
        self.nodes: dict[str, dict] = {}

    def instantiate_node(self, node_id: str, class_type: str, inputs: dict):
        cls = _discover_class(class_type)
        self.nodes[node_id] = {"instance": cls(), "inputs": inputs}

    def _resolve(self, value):
        if isinstance(value, list) and len(value) == 2:
            src, idx = value
            out = self.cache.get_output(str(src))
            if out is None:
                raise KeyError(f"Missing output for node {src}")
            return out[idx]
        return value

    def run_node(self, node_id: str):
        spec = self.nodes[node_id]
        inst = spec["instance"]
        inputs = {k: self._resolve(v) for k, v in spec["inputs"].items()}
        func = getattr(inst, getattr(inst, "FUNCTION", "run"))
        result = func(**inputs)
        if not isinstance(result, tuple):
            result = (result,)
        self.cache.set_output(node_id, result)

    def run_all(self):
        for nid in list(self.nodes.keys()):
            self.run_node(nid)


def run_workflow_from_json(graph_json: str, base_path: str | None = None):
    """Execute a minimal ComfyUI-like graph described in JSON."""
    graph = json.loads(graph_json)
    cache = ExecutionCache(base_path=base_path)
    executor = WorkflowExecutor(cache=cache)
    for node_id, spec in graph.items():
        class_type = spec["class_type"]
        inputs = spec.get("inputs", {})
        executor.instantiate_node(node_id, class_type, inputs)
    executor.run_all()
    return {"executor": executor, "outputs": {nid: cache.get_output(nid) for nid in graph}}
