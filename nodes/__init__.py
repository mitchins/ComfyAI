import os

if os.getenv("UNIT_TEST_MODE") != "1":
    from .vllm_query import (
        VLLMTextQuery,
        VLLMImageQuery,
        VLLMDualImageQuery,
    )
    from .conditional_save_image import ConditionalSaveImage
    from .compare_faces import CompareFacesNode

    NODE_CLASS_MAPPINGS = {
        "VLLMTextQuery": VLLMTextQuery,
        "VLLMImageQuery": VLLMImageQuery,
        "VLLMDualImageQuery": VLLMDualImageQuery,
        "ConditionalSaveImage": ConditionalSaveImage,
        "CompareFacesNode": CompareFacesNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "VLLMTextQuery": "LLM Query (Text Only)",
        "VLLMImageQuery": "Vision LLM Query (1 Image)",
        "VLLMDualImageQuery": "Vision LLM Query (2 Images)",
        "ConditionalSaveImage": "Conditional Save Image",
        "CompareFacesNode": "Face Comparison",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
