class EchoNode:
    """Simple node used for smoke testing the workflow runner."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "tests"

    def run(self, text):
        return text
