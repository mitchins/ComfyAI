"""Thin wrapper package used for tests and compatibility."""

import importlib.util
import os
import sys

PARENT = os.path.dirname(os.path.dirname(__file__))

spec = importlib.util.spec_from_file_location("ComfyNodes_core", os.path.join(PARENT, "__init__.py"))
core = importlib.util.module_from_spec(spec)
sys.modules["ComfyNodes_core"] = core
spec.loader.exec_module(core)

# Re-export public API
VisionLLMQuery = core.VisionLLMQuery
OpenAIQuery = core.OpenAIQuery
PersistentInferenceWorker = core.PersistentInferenceWorker
ConditionalSaveImage = core.ConditionalSaveImage

NODE_CLASS_MAPPINGS = core.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = core.NODE_DISPLAY_NAME_MAPPINGS
