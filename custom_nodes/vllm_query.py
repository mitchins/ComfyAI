import logging
import base64
from .string_utils import fuzzy_match_bool
from .openai_client import chat_completion
from .image_utils import image_to_bytes

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
        image = inputs.get("image")
        reference_image = inputs.get("reference_image")

        content = [{"type": "text", "text": text_query}]

        try:
            if image is not None:
                img_b = image_to_bytes(image)
                encoded = base64.b64encode(img_b).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
            if reference_image is not None:
                ref_b = image_to_bytes(reference_image)
                encoded = base64.b64encode(ref_b).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        except Exception:
            logging.exception("Failed to encode image inputs")

        messages = [{"role": "user", "content": content}]

        result = chat_completion(api_endpoint, api_model, messages, api_key=api_key)
        bool_output = fuzzy_match_bool(result)
        return result, bool_output, int(bool_output)
