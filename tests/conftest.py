import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ["UNIT_TEST_MODE"] = "1"

# Stub heavy modules so tests can run without optional deps
for name in ["torch", "torchvision", "onnxruntime", "PIL"]:
    if name not in sys.modules:
        module = types.ModuleType(name)
        if name == "torchvision":
            module.transforms = types.ModuleType("transforms")
        sys.modules[name] = module
