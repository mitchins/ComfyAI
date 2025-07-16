import os
import sys
import types
from pathlib import Path

# Ensure the parent directory (repo root) is on the path so we can import the
# actual implementation modules.
root_parent = Path(__file__).resolve().parents[2]
if str(root_parent) not in sys.path:
    sys.path.insert(0, str(root_parent))

if os.getenv("UNIT_TEST_MODE") == "1":
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(
            device_count=lambda: 0, empty_cache=lambda: None
        )
        torch_stub.Tensor = object
        sys.modules["torch"] = torch_stub
    if "torchvision" not in sys.modules:
        tv_stub = types.ModuleType("torchvision")
        tv_stub.transforms = types.ModuleType("transforms")
        sys.modules["torchvision"] = tv_stub
    if "qwen_vl_utils" not in sys.modules:
        qwen_stub = types.ModuleType("qwen_vl_utils")
        qwen_stub.process_vision_info = lambda *args, **kwargs: ([], [])
        sys.modules["qwen_vl_utils"] = qwen_stub
    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoConfig = object
        transformers_stub.AutoProcessor = object
        transformers_stub.AutoModelForVision2Seq = object
        models_module = types.ModuleType("transformers.models")
        auto_module = types.ModuleType("transformers.models.auto")
        modeling_auto_module = types.ModuleType(
            "transformers.models.auto.modeling_auto"
        )
        modeling_auto_module.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}
        sys.modules["transformers"] = transformers_stub
        sys.modules["transformers.models"] = models_module
        sys.modules["transformers.models.auto"] = auto_module
        sys.modules["transformers.models.auto.modeling_auto"] = modeling_auto_module
        transformers_stub.models = types.SimpleNamespace(auto=auto_module)

from ComfyAI import (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
                     BaseLLMQuery, ConditionalSaveImage, ImageLLMQuery,
                     PersistentInferenceWorker, TextLLMQuery, VisionLLMQuery,
                     vllm_query)

__all__ = [
    "BaseLLMQuery",
    "TextLLMQuery",
    "ImageLLMQuery",
    "VisionLLMQuery",
    "ConditionalSaveImage",
    "PersistentInferenceWorker",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "vllm_query",
]
