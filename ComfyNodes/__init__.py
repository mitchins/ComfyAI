import importlib

VisionLLMQuery = importlib.import_module('vllm_query').VisionLLMQuery
ConditionalSaveImage = importlib.import_module('conditional_save_image').ConditionalSaveImage

NODE_CLASS_MAPPINGS = {
    "VisionLLMQuery": VisionLLMQuery,
    "ConditionalSaveImage": ConditionalSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionLLMQuery": "Vision LLM Query",
    "ConditionalSaveImage": "Conditional Save Image",
}

