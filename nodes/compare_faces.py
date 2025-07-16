import os
import tempfile
import requests
import numpy as np
from PIL import Image


class CompareFacesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "api_url": (
                    "STRING",
                    {"default": "http://127.0.0.1:7860/v1/image/compare_faces"},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.72, "min": 0.0, "max": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("similarity_score", "same_person")
    FUNCTION = "run"
    CATEGORY = "AI/Face"

    def _tensor_to_pil(self, tensor):
        arr = np.clip(tensor.cpu().numpy() * 255.0, 0, 255).astype("uint8")
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr)

    def run(self, image_a, image_b, api_url, threshold=0.72):
        tmp_files = []
        try:
            for tensor in (image_a, image_b):
                img = self._tensor_to_pil(tensor)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name, format="PNG")
                tmp_files.append(tmp.name)
            with open(tmp_files[0], "rb") as fa, open(tmp_files[1], "rb") as fb:
                resp = requests.post(api_url, files={"image_a": fa, "image_b": fb}, timeout=30)
            if resp.status_code != 200:
                return 0.0, False
            data = resp.json()
            sim = float(data.get("similarity", 0.0))
            return sim, sim >= threshold
        except Exception:
            return 0.0, False
        finally:
            for path in tmp_files:
                try:
                    os.unlink(path)
                except Exception:
                    pass


__all__ = ["CompareFacesNode"]
