# Integration Tests

These tests execute small ComfyUI workflows headlessly. To resolve built-in nodes,
clone the ComfyUI repository into `ComfyUI_repo/` at the project root or otherwise
ensure it is discoverable on `PYTHONPATH`.
`conftest.py` automatically prepends this path and the `custom_nodes/` folder when
running the tests.

Run all integration tests:

```bash
pytest integration_tests
```

In CI, both unit tests (`pytest tests`) and these integration tests
are executed.
