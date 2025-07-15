import importlib
from typing import Any, Dict


class ExecutionCache:
    """Simple cache storing node outputs."""

    def __init__(self, base_path: str | None = None) -> None:
        self.base_path = base_path
        self._outputs: Dict[str, Any] = {}

    def set_output(self, node_id: str, value: Any) -> None:
        self._outputs[node_id] = value

    def get_output(self, node_id: str) -> Any:
        return self._outputs.get(node_id)


class WorkflowExecutor:
    """Very small workflow executor for tests."""

    def __init__(self, cache: ExecutionCache | None = None) -> None:
        self.cache = cache or ExecutionCache()
        self._nodes: Dict[str, tuple[Any, Dict[str, Any]]] = {}

    def _resolve_class(self, class_type: str):
        """Return node class from custom_nodes or builtin nodes."""
        try:
            from custom_nodes import NODE_CLASS_MAPPINGS
            if class_type in NODE_CLASS_MAPPINGS:
                return NODE_CLASS_MAPPINGS[class_type]
        except Exception:
            pass

        # Attempt to import module from custom_nodes package by scanning all modules
        try:
            import pkgutil
            import custom_nodes
            for _, mod_name, _ in pkgutil.iter_modules(custom_nodes.__path__):
                mod = importlib.import_module(f"custom_nodes.{mod_name}")
                if hasattr(mod, class_type):
                    return getattr(mod, class_type)
        except Exception:
            pass

        # Builtin ComfyUI nodes if available
        try:
            import nodes
            mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
            if class_type in mappings:
                return mappings[class_type]
            if hasattr(nodes, class_type):
                return getattr(nodes, class_type)
        except Exception:
            pass

        raise KeyError(f"Unknown node class: {class_type}")

    def instantiate_node(self, node_id: str, class_type: str, inputs: Dict[str, Any]):
        cls = self._resolve_class(class_type)
        instance = cls() if isinstance(cls, type) else cls
        self._nodes[node_id] = (instance, inputs)

    def _execute_node(self, node_id: str):
        instance, inputs = self._nodes[node_id]
        resolved: Dict[str, Any] = {}
        for key, val in inputs.items():
            if isinstance(val, list) and len(val) == 2:
                up_id, slot = val
                upstream = self.cache.get_output(up_id)
                if isinstance(upstream, tuple | list):
                    resolved[key] = upstream[slot]
                else:
                    resolved[key] = upstream
            else:
                resolved[key] = val
        func_name = getattr(instance, "FUNCTION", "run")
        func = getattr(instance, func_name)
        output = func(**resolved)
        self.cache.set_output(node_id, output)

    def run_all(self):
        for node_id in list(self._nodes.keys()):
            self._execute_node(node_id)
