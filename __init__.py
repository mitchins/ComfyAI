from .vllm_query import TextLLMQuery, OneImageLLMQuery, TwoImageLLMQuery, PersistentInferenceWorker
from .conditional_save_image import ConditionalSaveImage

NODE_CLASS_MAPPINGS = {
    "TextLLMQuery": TextLLMQuery,
    "OneImageLLMQuery": OneImageLLMQuery,
    "TwoImageLLMQuery": TwoImageLLMQuery,
    "ConditionalSaveImage": ConditionalSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextLLMQuery": "Text LLM Query",
    "OneImageLLMQuery": "One Image LLM Query",
    "TwoImageLLMQuery": "Two Image LLM Query",
    "ConditionalSaveImage": "Conditional Save Image",
}
