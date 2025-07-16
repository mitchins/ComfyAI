"""Integration tests for ComfyAI.

These tests run small ComfyUI workflows without launching the UI. They require a
clone of the ComfyUI repository placed at ``ComfyUI_repo/`` in the project root
(or otherwise available on ``PYTHONPATH``). ``integration_tests/conftest.py``
prepends that directory and ``nodes`` to ``sys.path`` during test
initialisation.

Run these tests with::

    pytest integration_tests

Continuous integration runs ``pytest tests`` followed by ``pytest
integration_tests``.
"""
