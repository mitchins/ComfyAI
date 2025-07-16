# Developing Custom ComfyUI Nodes

This document explains how to create new nodes inside the `nodes/` folder.  The examples below mirror the existing nodes in this repository.

## Basic Class Layout

Each node is a Python class with a small set of class attributes that describe its interface to ComfyUI:

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"text": ("STRING", {"default": "hello"})}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "examples"

    def run(self, text: str):
        return text.upper()
```

The `INPUT_TYPES` method declares inputs shown in the UI. `RETURN_TYPES` and
`RETURN_NAMES` describe the returned values. `FUNCTION` points at the method that
implements the behaviour.  `CATEGORY` controls where the node appears in the
ComfyUI menu.

After creating the class, register it inside `nodes/__init__.py` by adding it to
`NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` so ComfyUI can discover
it.

## Type Contracts

Inputs and outputs are defined using primitive type names understood by ComfyUI.
The existing nodes demonstrate common patterns.  For example the vision query
node defines its contract as follows:

```python
class _BaseVLLMQuery:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "text_query": ("STRING", {"default": "Describe the image.", "multiline": True}),
            "api_endpoint": ("STRING", {"default": "", "multiline": False}),
            "api_model": ("STRING", {"default": "gpt-3.5-turbo", "multiline": False}),
            "api_key": ("STRING", {"default": "", "multiline": False}),
        }
        if cls.required_images >= 1:
            required["image"] = ("IMAGE",)
        if cls.required_images >= 2:
            required["reference_image"] = ("IMAGE",)
        return {"required": required}

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("Raw Text", "Boolean", "Number (Boolean)")
    FUNCTION = "run"
    CATEGORY = "AI/Large Language Models"
```

The same style is reused by other nodes such as `CompareFacesNode` and
`ConditionalSaveImage`.

## Returning UI Data

A node can supply additional information for the ComfyUI interface by returning
an object with a top level `"ui"` field.  The `ConditionalSaveImage` node is an
example:

```python
if not condition:
    return {"ui": {"images": []}}
else:
    # ... save files ...
    return {"ui": {"images": results}}
```

This structure allows the UI to display extra data (e.g. saved image paths)
without altering the main return values.

## Error Handling

Errors are propagated in two ways.  For invalid input the node should raise an
exception which ComfyUI will surface.  The `_BaseVLLMQuery` class raises a
`ValueError` when the API endpoint is missing:

```python
api_endpoint = inputs.get("api_endpoint", "").strip()
if not api_endpoint:
    raise ValueError("api_endpoint is required")
```

For network failures or optional functionality, return a safe default instead of
raising.  `CompareFacesNode` catches errors from the face API and returns a zero
similarity score and `False`:

```python
try:
    data = fetch_face_api_response(api_url, tmp_files[0], tmp_files[1])
except Exception:
    return 0.0, False
```

Use `logging.exception()` for unexpected errors so that stack traces are logged
while still providing predictable output to the workflow.

## Testing

During unit tests, heavy dependencies are stubbed when the environment variable
`UNIT_TEST_MODE` is set.  See `ConditionalSaveImage` for an example of guarding
imports in test mode.  This allows the nodes to be tested without requiring a
full ComfyUI installation.

---

With these guidelines you can quickly prototype new ComfyUI nodes.  Inspect the
existing modules under `nodes/` for more practical examples and keep the type
contracts consistent so your nodes integrate smoothly with the UI.
