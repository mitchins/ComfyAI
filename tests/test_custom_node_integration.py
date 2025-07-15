import json
from PIL import Image
from comfyai.testing.workflow_runner import register_node_class
import custom_nodes.vllm_query as vllm_query

class DummyLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "test"

    def run(self, path):
        return (Image.open(path),)

register_node_class("DummyLoadImage", DummyLoadImage)


def test_vllm_query_workflow(run_graph, dummy_image, tmp_path, monkeypatch):
    img_path = tmp_path / "dummy.png"
    dummy_image.save(img_path)

    monkeypatch.setattr(vllm_query, "chat_completion", lambda *a, **k: "yes")
    monkeypatch.setattr(vllm_query, "fuzzy_match_bool", lambda x: True)
    monkeypatch.setattr(vllm_query, "image_to_bytes", lambda img: b"img")

    graph = {
        "1": {
            "class_type": "DummyLoadImage",
            "inputs": {"path": str(img_path)}
        },
        "2": {
            "class_type": "VisionLLMQuery",
            "inputs": {
                "image": ["1", 0],
                "text_query": "hi",
                "api_endpoint": "http://x",
                "api_model": "gpt",
                "api_key": ""
            }
        }
    }

    result = run_graph(json.dumps(graph))
    outputs = result["outputs"]
    assert outputs["2"][0] == "yes"
    assert outputs["2"][1] is True
    assert outputs["2"][2] == 1
