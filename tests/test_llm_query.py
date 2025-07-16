import os
import unittest
from unittest.mock import patch

# ensure unit test mode for conditional imports
os.environ["UNIT_TEST_MODE"] = "1"
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ComfyNodes import ImageLLMQuery, TextLLMQuery, VisionLLMQuery


class FakeWorker:
    def __init__(self, *_, **__):
        self.task = None

    def submit_task(self, task):
        self.task = task

    def get_result(self):
        return f"Processed: {self.task.text_query}"

    def shutdown(self):
        pass


def run_query(query_cls, **inputs):
    query = query_cls()
    with patch("ComfyNodes.vllm_query.PersistentInferenceWorker", FakeWorker), patch(
        "ComfyNodes.vllm_query.image_to_bytes", lambda *_: b""
    ):
        return query.run(**inputs)


class TestLLMQueries(unittest.TestCase):
    def test_text_query(self):
        res = run_query(
            TextLLMQuery, text_query="hello", gpu_device="cpu", model_name="dummy"
        )
        self.assertEqual(res[0], "Processed: hello")

    def test_image_query(self):
        res = run_query(
            ImageLLMQuery,
            image=None,
            text_query="hi",
            gpu_device="cpu",
            model_name="dummy",
        )
        self.assertEqual(res[0], "Processed: hi")

    def test_vision_query(self):
        res = run_query(
            VisionLLMQuery,
            image=None,
            text_query="what",
            reference_image=None,
            gpu_device="cpu",
            model_name="dummy",
        )
        self.assertEqual(res[0], "Processed: what")


if __name__ == "__main__":
    unittest.main()
