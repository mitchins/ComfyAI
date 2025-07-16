import os
import tempfile
import requests
from PIL import Image
from typing import Tuple

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
    FUNCTION = "compare"
    CATEGORY = "AI/Face"

    def compare(self, image_a, image_b, api_url: str, threshold: float = 0.72) -> Tuple[float, bool]:
        tmp_paths = []
        try:
            pil_a = tensor_to_pil(image_a)
            pil_b = tensor_to_pil(image_b)
            f_a = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_a.save(f_a.name, format="PNG")
            f_a.close()
            tmp_paths.append(f_a.name)
            f_b = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_b.save(f_b.name, format="PNG")
            f_b.close()
            tmp_paths.append(f_b.name)

            with open(f_a.name, "rb") as fa, open(f_b.name, "rb") as fb:
                files = {"image_a": fa, "image_b": fb}
                resp = requests.post(api_url, files=files, timeout=30)

            if resp.status_code != 200:
                return 0.0, False
            data = resp.json()
            sim = float(data.get("similarity", 0.0))
            return sim, sim >= threshold
        except Exception:
            return 0.0, False
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
