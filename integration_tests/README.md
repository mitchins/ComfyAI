# Integration Tests

These tests execute small ComfyUI workflows using the headless
`workflow_runner` utility. They require the ComfyUI core source to be
available for import.

## Requirements

- Clone the upstream **ComfyUI** repository into `third_party/ComfyUI`.
- Ensure this project's `custom_nodes` directory is importable (handled
  automatically by `conftest.py`).

## Running

Install dev dependencies and run pytest only on integration tests:

```bash
pip install -r requirements-dev.txt
pytest integration_tests -vv
```

CI runs unit tests from `tests/` and these integration tests separately.
