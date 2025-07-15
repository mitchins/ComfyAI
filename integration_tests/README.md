# Integration Tests

These tests run small workflows headlessly and exercise the optional ONNX server.

Ensure a clone of the core **ComfyUI** repository is available at one of the
following paths before running:

```
third_party/ComfyUI/    # preferred
comfyui-core/
ComfyUI_repo/
```

Both the core source and this project's `custom_nodes` directory are injected
into `sys.path` by `integration_tests/conftest.py`.

Execute the suite with:

```bash
pytest integration_tests
```
