# Developing Custom ComfyUI Nodes

This document explains how to create new nodes for ComfyUI using the code in this
folder as reference. It summarises the minimal API each node must expose and
shows how to surface information back to the UI.

## Node Discovery

All node classes placed inside the `nodes/` package can be automatically
registered. `__init__.py` exposes them through `NODE_CLASS_MAPPINGS` and
`NODE_DISPLAY_NAME_MAPPINGS`. When `UNIT_TEST_MODE` is not set the mappings are
populated and ComfyUI loads the node definitions.

```
if os.getenv("UNIT_TEST_MODE") != "1":
    from .vllm_query import VLLMTextQuery
    ...
    NODE_CLASS_MAPPINGS = {"VLLMTextQuery": VLLMTextQuery, ...}
```

New nodes simply need to be imported and added to these dictionaries.

## Interface Contract

Each node follows a small contract understood by ComfyUI:

- `INPUT_TYPES(cls)` – class method returning a dictionary with `required`,
  `optional` and `hidden` inputs. Each entry maps an input name to a tuple
  describing the type and optional UI hints. See `VLLMTextQuery.INPUT_TYPES`
  for an example.
- `RETURN_TYPES` – tuple of strings describing each output type.
- `RETURN_NAMES` (optional) – names shown on the output sockets in ComfyUI.
- `FUNCTION` – method called when the node executes (defaults to `run`).
- `CATEGORY` – location within the ComfyUI menu.

Input/return type strings match the types defined by ComfyUI such as `STRING`,
`BOOLEAN`, `IMAGE`, `FLOAT`, and so on.

### UI dictionaries

Node functions may return a tuple matching `RETURN_TYPES` or a dictionary
containing a `"ui"` key. Returning a `"ui"` dictionary allows a node to supply
extra information to ComfyUI's viewer. The `ConditionalSaveImage` node returns
`{"ui": {"images": [...]}}` so that saved images appear in the UI. When using
this form you can still return additional values as needed.

```
def save_images(self, condition, images, filename_prefix="ComfyUI", ...):
    if not condition:
        return {"ui": {"images": []}}
    ...
    return {"ui": {"images": results}}
```

### Error propagation

Errors inside a node are typically signalled by raising Python exceptions. If an
exception bubbles up the node is highlighted red in ComfyUI. Use standard
exceptions such as `ValueError` or `RuntimeError` as seen in `CompareFacesNode`
when `torch` is missing. When you need to fail gracefully, return a default
value or a `{"ui": ...}` dictionary with suitable placeholders.

```
if torch is None:
    raise RuntimeError("torch is required")

try:
    data = fetch_face_api_response(api_url, tmp_files[0], tmp_files[1])
except Exception:
    return 0.0, False
```

## Minimal Example

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "MyNodes"

    def run(self, text: str) -> tuple[str]:
        return (text.upper(),)
```

Add the class to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in
`__init__.py` and ComfyUI will pick it up.

## Testing Nodes

Unit tests under `tests/` show how to instantiate nodes directly. The
`comfyai.testing.workflow_runner` module provides a tiny executor that can load
and run nodes described in JSON – useful for integration tests or headless
workflows.

