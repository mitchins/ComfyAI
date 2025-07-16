import os
import tempfile
import requests

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from custom_nodes.image_utils import tensor_to_pil


class CompareFacesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "api_url": ("STRING", {"default": "http://127.0.0.1:7860/v1/image/compare_faces"}),
                "threshold": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("similarity_score", "same_person")
    FUNCTION = "run"
    CATEGORY = "AI/Faces"

    def run(self, image_a, image_b, api_url, threshold=0.72):
        if torch is None:
            raise RuntimeError("torch is required")

        tmp_files = []
        try:
            for tensor in (image_a, image_b):
                img = tensor_to_pil(tensor)
                fd, path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                img.save(path, format="PNG")
                tmp_files.append(path)

            with open(tmp_files[0], "rb") as f1, open(tmp_files[1], "rb") as f2:
                files = {"image_a": f1, "image_b": f2}
                try:
                    resp = requests.post(api_url, files=files, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    return 0.0, False

            similarity = float(data.get("similarity", 0.0))
            same = similarity >= float(threshold)
            return similarity, same
        finally:
            for p in tmp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass


__all__ = ["CompareFacesNode"]
