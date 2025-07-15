# Integration Tests

These tests run small workflows using a local clone of **ComfyUI**. The path to the ComfyUI source must be available so Python can import built-in nodes.

## Requirements
- `ComfyUI` cloned into `ComfyUI_repo/` at the repository root or otherwise on `PYTHONPATH`.
- Development dependencies from `requirements-dev.txt`.

## Running
```bash
pip install -r requirements-dev.txt
pytest tests          # unit tests
pytest integration_tests  # integration tests
```

To run only the integration suite in CI/CD, invoke `pytest integration_tests`.
