import os
import sys
import types
import pytest
from PIL import Image
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub heavy modules so imports never fail
for mod in ("torch", "onnxruntime", "fastapi", "uvicorn"):
    sys.modules.setdefault(mod, types.ModuleType(mod))

@pytest.fixture
def dummy_image():
    """16×16 black RGB image"""
    return Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

@pytest.fixture
def dummy_string():
    return "test"

@pytest.fixture
def dummy_int():
    return 0
