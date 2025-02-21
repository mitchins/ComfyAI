import unittest
import time
from multiprocessing import Pipe
from ComfyNodes import PersistentInferenceWorker
import subprocess
import os

DUMMY_WORKER_PATH = os.path.join(os.path.dirname(__file__), "dummy_worker.py")
os.environ["UNIT_TEST_MODE"] = "1"

class TestPersistentInferenceWorker(unittest.TestCase):
    
    def setUp(self):
        """Initialize worker before each test."""
        self.worker = PersistentInferenceWorker(gpu_device="10000", worker_module="dummy_worker")

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
        self.worker = PersistentInferenceWorker(gpu_device="100", worker_module="dummy_worker")

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
        self.worker = PersistentInferenceWorker(gpu_device="10000", worker_module="dummy_worker")

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