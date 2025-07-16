from .conditional_save_image import ConditionalSaveImage
from .vllm_query import (BaseLLMQuery, ImageLLMQuery,
                         PersistentInferenceWorker, TextLLMQuery,
                         VisionLLMQuery)

NODE_CLASS_MAPPINGS = {
    "TextLLMQuery": TextLLMQuery,
    "ImageLLMQuery": ImageLLMQuery,
    "VisionLLMQuery": VisionLLMQuery,
    "ConditionalSaveImage": ConditionalSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextLLMQuery": "Text LLM Query",
    "ImageLLMQuery": "Image LLM Query",
    "VisionLLMQuery": "Vision LLM Query",
    "ConditionalSaveImage": "Conditional Save Image",
}

__all__ = [
    "BaseLLMQuery",
    "TextLLMQuery",
    "ImageLLMQuery",
    "VisionLLMQuery",
    "ConditionalSaveImage",
    "PersistentInferenceWorker",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
