from .vllm_query import VisionLLMQuery, PersistentInferenceWorker
from .openai_query import OpenAIQuery
from .conditional_save_image import ConditionalSaveImage

# Expose submodules under the ``ComfyNodes`` package name when this file is
# imported as such (useful for tests and external integrations).
import sys as _sys
_pkg = __name__
_sys.modules.setdefault(f"{_pkg}.vllm_query", _sys.modules.get("vllm_query"))
_sys.modules.setdefault(f"{_pkg}.openai_query", _sys.modules.get("openai_query"))
_sys.modules.setdefault(
    f"{_pkg}.conditional_save_image",
    _sys.modules.get("conditional_save_image"),
)

NODE_CLASS_MAPPINGS = {
    "VisionLLMQuery": VisionLLMQuery,
    "OpenAIQuery": OpenAIQuery,
    "ConditionalSaveImage": ConditionalSaveImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMQuery": "Vision LLM Query",
    "OpenAIQuery": "OpenAI Query",
    "ConditionalSaveImage": "Conditional Save Image",
}