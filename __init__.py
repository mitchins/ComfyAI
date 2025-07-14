try:
    from .vllm_query import VisionLLMQuery
    from .conditional_save_image import ConditionalSaveImage
except Exception:  # pragma: no cover - optional deps in tests
    VisionLLMQuery = None
    ConditionalSaveImage = None

NODE_CLASS_MAPPINGS = {
    "VisionLLMQuery": VisionLLMQuery,
    "ConditionalSaveImage": ConditionalSaveImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMQuery": "Vision LLM Query",
    "ConditionalSaveImage": "Conditional Save Image",
}