class DummyAddOne:
    """Simple node returning the input integer incremented by one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "run"

    def run(self, value):
        return (value + 1,)
