"""Utilities for running tiny ComfyUI graphs headlessly."""

from __future__ import annotations

import json
from typing import Any, Dict

from comfyui.workflow_executor import WorkflowExecutor, ExecutionCache


def run_workflow_from_json(graph_json: str, base_path: str | None = None) -> Dict[str, Any]:
    """Execute a minimal graph encoded as JSON.

    Parameters
    ----------
    graph_json: str
        JSON mapping node_id -> {"class_type": str, "inputs": {...}}
    base_path: str | None
        Optional base path for ExecutionCache.

    Returns
    -------
    dict
        {"executor": WorkflowExecutor, "outputs": {node_id: output}}
    """
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
