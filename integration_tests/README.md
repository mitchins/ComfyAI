# Integration Tests

These tests exercise the custom nodes inside a real ComfyUI workflow. A local
clone of the ComfyUI repository must be available at `third_party/ComfyUI` so
the test harness can import the built-in nodes.

Run them alongside the unit tests:

```bash
pytest tests integration_tests -q
```
