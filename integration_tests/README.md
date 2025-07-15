# Integration Tests

These tests exercise ComfyAI together with a copy of the ComfyUI core. They
require the ComfyUI source to be available so that nodes can be imported.

1. Clone the ComfyUI repository next to this project:

```bash
 git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI_repo
```

2. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Run the tests:

```bash
pytest integration_tests
```

The GitHub Actions workflow runs `pytest tests` and `pytest integration_tests`
separately. Running only the integration tests locally is as simple as invoking
`pytest integration_tests`.
