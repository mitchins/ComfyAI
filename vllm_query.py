import logging
import base64
from string_utils import fuzzy_match_bool
from openai_client import chat_completion
from image_utils import image_to_bytes

class VisionLLMQuery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_query": ("STRING", {"default": "Describe the image.", "multiline": True}),
                "api_endpoint": ("STRING", {"default": "", "multiline": False}),
                "api_model": ("STRING", {"default": "gpt-3.5-turbo", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("Raw Text", "Boolean", "Number (Boolean)")
    FUNCTION = "run"
    CATEGORY = "AI/Large Language Models"

    def run(self, **inputs):
        api_endpoint = inputs.get("api_endpoint", "").strip()
        if not api_endpoint:
            raise ValueError("api_endpoint is required")
        api_model = inputs.get("api_model", "gpt-3.5-turbo")
        api_key = inputs.get("api_key", "") or None
        text_query = inputs.get("text_query", "")
        messages = [{"role": "user", "content": text_query}]

        images = []
        for key in ("image", "reference_image"):
            img_tensor = inputs.get(key)
            if img_tensor is not None:
                byte_data = image_to_bytes(img_tensor)
                images.append(base64.b64encode(byte_data).decode("utf-8"))

        result = chat_completion(
            api_endpoint,
            api_model,
            messages,
            api_key=api_key,
            images=images or None,
        )
        bool_output = fuzzy_match_bool(result)
        return result, bool_output, int(bool_output)
