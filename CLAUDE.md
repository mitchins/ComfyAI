# CLAUDE.md - Agent Configuration

## Environment Configuration

### Conda Setup (macOS)
- Conda location: `/opt/miniconda3/bin/conda`
- Test environment: `comfy-test`
- Use: `/opt/miniconda3/bin/conda run -n comfy-test python -m pytest`

### Python Commands
- Use conda python: `/opt/miniconda3/bin/conda run -n comfy-test python`
- Run tests: `/opt/miniconda3/bin/conda run -n comfy-test python -m pytest`
- Run specific test: `/opt/miniconda3/bin/conda run -n comfy-test python -m pytest tests/test_file.py -v`

## Project Setup
- Working directory: `/Users/mitchellcurrie/Projects/ComfyAI`
- Main branch: `main`
- Current branch: `develop`

## Testing
- Test environments use mocked ONNX dependencies when real ones aren't available
- ONNX inference tests are in `tests/test_onnx_inference.py`
- Integration tests are in `integration_tests/`
- Use `-v` flag for verbose test output