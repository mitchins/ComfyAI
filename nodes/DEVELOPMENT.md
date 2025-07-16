# Developing Custom ComfyUI Nodes

This document is a short reference for creating your own nodes.  Each node is a small Python
class that exposes a **type contract** so that ComfyUI can build the user interface automatically.
The examples below are taken from the nodes shipped with this repository.

## Basic Node Skeleton

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello"}),
            },
            "optional": {
                "count": ("INT", {"default": 1, "min": 1})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "example"

    def run(self, text: str, count: int = 1):
        return text * count
```

A node may define `INPUT_TYPES` with `required`, `optional` and `hidden` inputs.
`RETURN_TYPES` and `RETURN_NAMES` describe the tuple returned from `run`.  The
`FUNCTION` attribute names the method that will be executed (defaults to `run`).
The `CATEGORY` string controls where the node appears in the ComfyUI menu.

## Propagating Errors

You can raise regular Python exceptions from inside the node.  ComfyUI will
surface these errors to the user.  For recoverable issues you may instead return
default values.  For example, [`CompareFacesNode`](compare_faces.py) catches
network failures and simply returns `0.0, False` so downstream nodes continue
running:

```python
try:
    data = fetch_face_api_response(api_url, img_a, img_b)
except Exception:
    return 0.0, False
```

For invalid inputs it is usually best to raise `ValueError` or `RuntimeError`
like `_BaseVLLMQuery` does when the API endpoint is missing:

```python
api_endpoint = inputs.get("api_endpoint", "").strip()
if not api_endpoint:
    raise ValueError("api_endpoint is required")
```

## UI Hints

Each input tuple consists of a **type string** and optional metadata.  The type
controls the widget shown in the UI (e.g. `"IMAGE"`, `"STRING"`, `"BOOLEAN"`).
Metadata keys such as `"default"`, `"min"`, `"max"` or `"tooltip"` influence the
initial value and help text.  See the source in this folder for further examples.

When returning UI elements (for example, from `ConditionalSaveImage.save_images`)
use the `{"ui": ...}` dictionary structure so that ComfyUI can display results
without additional coding.

## Registration

Place your class in any module under `nodes/` and expose it through
`NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `nodes/__init__.py`.
The classes will then be discovered automatically when ComfyUI starts.

```python
NODE_CLASS_MAPPINGS = {
    "MyNode": MyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNode": "My Example Node",
}
```

Happy coding!
