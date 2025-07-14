try:
    import openai
except ImportError as e:
    raise ImportError("openai package is required for OpenAIQuery") from e

import os
from .string_utils import fuzzy_match_bool

class OpenAIQuery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_query": ("STRING", {"default": "Hello", "multiline": True}),
                "model_name": ("STRING", {"default": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")}),
                "api_base": ("STRING", {"default": os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")}),
                "api_key": ("STRING", {"default": os.getenv("OPENAI_API_KEY", ""), "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("Raw Text", "Boolean", "Number (Boolean)")
    FUNCTION = "run"
    CATEGORY = "AI/Large Language Models"

    def run(self, text_query, model_name, api_base, api_key=""):
        openai.api_base = api_base
        if api_key:
            openai.api_key = api_key

        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": text_query}],
        )
        answer = resp.choices[0].message.content
        bool_output = fuzzy_match_bool(answer)
        return answer, bool_output, int(bool_output)
