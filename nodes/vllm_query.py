import logging
import base64
from .string_utils import fuzzy_match_bool
from .openai_client import chat_completion
from .image_utils import image_to_bytes


class _BaseVLLMQuery:
    """Common logic for LLM queries."""

    required_images = 0

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "text_query": ("STRING", {"default": "Describe the image.", "multiline": True}),
            "api_endpoint": ("STRING", {"default": "", "multiline": False}),
            "api_model": ("STRING", {"default": "gpt-3.5-turbo", "multiline": False}),
            "api_key": ("STRING", {"default": "", "multiline": False}),
        }
        if cls.required_images >= 1:
            required["image"] = ("IMAGE",)
        if cls.required_images >= 2:
            required["reference_image"] = ("IMAGE",)
        return {"required": required}

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

        images = []
        if self.required_images >= 1:
            images.append(inputs.get("image"))
        if self.required_images >= 2:
            images.append(inputs.get("reference_image"))

        content = [{"type": "text", "text": text_query}]

        try:
            for img in images:
                if img is not None:
                    img_b = image_to_bytes(img)
                    encoded = base64.b64encode(img_b).decode()
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        except Exception:
            logging.exception("Failed to encode image inputs")

        messages = [{"role": "user", "content": content}]

        result = chat_completion(api_endpoint, api_model, messages, api_key=api_key)
        bool_output = fuzzy_match_bool(result)
        if bool_output is None:
            bool_output = False
        return result, bool_output, int(bool_output)


class VLLMTextQuery(_BaseVLLMQuery):
    """LLM text-only query."""

    required_images = 0


class VLLMImageQuery(_BaseVLLMQuery):
    """LLM query with one image."""

    required_images = 1


class VLLMDualImageQuery(_BaseVLLMQuery):
    """LLM query with two images."""

    required_images = 2


__all__ = [
    "_BaseVLLMQuery",
    "VLLMTextQuery",
    "VLLMImageQuery",
    "VLLMDualImageQuery",
]
