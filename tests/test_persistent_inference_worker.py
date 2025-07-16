import unittest
import time
from multiprocessing import Pipe
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
from ComfyNodes import PersistentInferenceWorker
import subprocess
import os

DUMMY_WORKER_PATH = os.path.join(os.path.dirname(__file__), "dummy_worker.py")
os.environ["UNIT_TEST_MODE"] = "1"

class TestPersistentInferenceWorker(unittest.TestCase):
    
    def setUp(self):
        """Initialize worker before each test."""
        self.worker = PersistentInferenceWorker(gpu_device="10000", model_name="dummy", worker_module="dummy_worker")

    def tearDown(self):
        """Shutdown worker after each test."""
        self.worker.shutdown()

    def test_basic_task_execution(self):
        """Test if the worker correctly processes a task."""
        self.worker.start_worker()
        self.worker.submit_task("Hello, Worker!")
        result = self.worker.get_result()

        self.assertEqual(result, "Processed: Hello, Worker!", "Worker should return processed response.")

    def test_worker_crash_and_recovery(self):
        """Test if the worker recovers from a crash (simulated timeout)."""
        self.worker.shutdown()
        self.worker = PersistentInferenceWorker(gpu_device="100", model_name="dummy", worker_module="dummy_worker")

        # Launch worker that crashes after 100ms
        self.worker.start_worker(extra_args=["100"])

        self.worker.submit_task("Test Crash Recovery")
        time.sleep(0.2)  # Wait for crash to occur

        self.assertIsNone(self.worker.get_result(), "Worker should detect the crash and return None.")

        # Ensure worker restarts
        self.worker.submit_task("Hello Again!")
        result = self.worker.get_result()

        self.assertEqual(result, "Processed: Hello Again!", "Worker should recover and process new tasks.")

    def test_worker_signal_termination(self):
        """Test if the worker recovers from an external termination signal."""
        self.worker.shutdown()
        self.worker = PersistentInferenceWorker(gpu_device="10000", model_name="dummy", worker_module="dummy_worker")

        self.worker.start_worker(extra_args=["0", "--crash-on-signal"])

        self.worker.submit_task("Test Signal")
        time.sleep(0.5)

        # Kill worker
        self.worker.worker.terminate()
        time.sleep(0.5)  # Give it time to crash

        self.worker.submit_task("Post Crash Task")
        result = self.worker.get_result()

        self.assertEqual(result, "Processed: Post Crash Task", "Worker should recover and process new tasks.")


if __name__ == "__main__":
    unittest.main()