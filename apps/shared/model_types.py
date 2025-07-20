from enum import Enum


class ModelType(str, Enum):
    """Enum for different model types based on their capabilities."""
    
    TEXT_LLM = "text-llm"
    VISION_LLM = "vision-llm"  
    VISION_EMBEDDER = "vision-embedder"
    TEXT_EMBEDDER = "text-embedder"
    UNKNOWN = "unknown"

    @classmethod
    def chat_compatible_types(cls):
        """Return model types that are compatible with chat endpoints."""
        return [cls.TEXT_LLM, cls.VISION_LLM]

    @classmethod
    def vision_compatible_types(cls):
        """Return model types that are compatible with vision endpoints."""
        return [cls.VISION_LLM, cls.VISION_EMBEDDER]