import unittest
import types, sys, os
os.environ.setdefault("UNIT_TEST_MODE", "1")
torch_stub = types.ModuleType('torch')
torch_stub.cuda = types.SimpleNamespace(device_count=lambda: 0, empty_cache=lambda: None)
torch_stub.Tensor = object
sys.modules.setdefault('torch', torch_stub)
tv_stub = types.ModuleType('torchvision')
tv_stub.transforms = types.ModuleType('transforms')
sys.modules.setdefault('torchvision', tv_stub)
pil_stub = types.ModuleType('PIL')
pil_image_module = types.ModuleType('PIL.Image')
class DummyImage: pass
pil_image_module.Image = DummyImage
pil_stub.Image = pil_image_module
sys.modules.setdefault('PIL', pil_stub)
sys.modules.setdefault('PIL.Image', pil_image_module)
png_stub = types.ModuleType('PIL.PngImagePlugin')
png_stub.PngInfo = object
sys.modules.setdefault('PIL.PngImagePlugin', png_stub)
rf_stub = types.ModuleType('rapidfuzz')
rf_stub.fuzz = types.SimpleNamespace(ratio=lambda a,b: 100 if a==b else 0)
sys.modules.setdefault('rapidfuzz', rf_stub)
sys.modules.setdefault('rapidfuzz.fuzz', rf_stub.fuzz)
hf_stub = types.ModuleType('huggingface_hub')
hf_stub.scan_cache_dir = lambda: types.SimpleNamespace(repos=[])
sys.modules.setdefault('huggingface_hub', hf_stub)
tr_stub = types.ModuleType('transformers')
tr_stub.AutoConfig = object
tr_stub.AutoProcessor = object
tr_stub.AutoModelForVision2Seq = object
sys.modules.setdefault('transformers', tr_stub)
modeling_auto_stub = types.ModuleType('modeling_auto')
modeling_auto_stub.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}
sys.modules.setdefault('transformers.models.auto.modeling_auto', modeling_auto_stub)
qwen_stub = types.ModuleType('qwen_vl_utils')
qwen_stub.process_vision_info = lambda x: (None, None)
sys.modules.setdefault('qwen_vl_utils', qwen_stub)
sys.modules.setdefault('numpy', types.ModuleType('numpy'))
from ComfyNodes import TextLLMQuery, OneImageLLMQuery, TwoImageLLMQuery
from ComfyNodes.util.task_data import TaskData

class FakeWorker:
    def __init__(self):
        self.task = None
    def submit_task(self, task):
        self.task = task
    def get_result(self):
        return "yes"
    def shutdown(self):
        pass

class TestableText(TextLLMQuery):
    __test__ = False
    def create_worker(self, gpu_device, model_name):
        return FakeWorker()

    def build_task(self, **inputs):
        return TaskData(image_bytes=None, reference_bytes=None, text_query=inputs.get("text_query", ""))

class TestableOne(OneImageLLMQuery):
    __test__ = False
    def create_worker(self, gpu_device, model_name):
        return FakeWorker()

    def build_task(self, **inputs):
        return TaskData(image_bytes=b"img", reference_bytes=None, text_query=inputs.get("text_query", ""))

class TestableTwo(TwoImageLLMQuery):
    __test__ = False
    def create_worker(self, gpu_device, model_name):
        return FakeWorker()

    def build_task(self, **inputs):
        return TaskData(image_bytes=b"img", reference_bytes=b"ref", text_query=inputs.get("text_query", ""))

class TestVLLMQuery(unittest.TestCase):
    def setUp(self):
        self.image = b'dummy'

    def test_text_only(self):
        node = TestableText()
        result = node.run(text_query="test", model_name="dummy", gpu_device="cpu")
        self.assertEqual(result[0], "yes")
        self.assertIsNone(node.worker.task.image_bytes)

    def test_one_image(self):
        node = TestableOne()
        result = node.run(image=self.image, text_query="test", model_name="dummy", gpu_device="cpu")
        self.assertEqual(result[0], "yes")
        self.assertIsNotNone(node.worker.task.image_bytes)
        self.assertIsNone(node.worker.task.reference_bytes)

    def test_two_image(self):
        node = TestableTwo()
        result = node.run(image=self.image, reference_image=self.image, text_query="test", model_name="dummy", gpu_device="cpu")
        self.assertEqual(result[0], "yes")
        self.assertIsNotNone(node.worker.task.image_bytes)
        self.assertIsNotNone(node.worker.task.reference_bytes)

if __name__ == '__main__':
    unittest.main()
