import os
import tempfile
from typing import Tuple

import requests
from PIL import Image

from custom_nodes.image_utils import tensor_to_pil


class CompareFacesNode:
    """Send two face images to a comparison API and return similarity."""

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
    FUNCTION = "compare"
    CATEGORY = "image"

    def compare(
        self,
        image_a,
        image_b,
        api_url: str = "http://127.0.0.1:7860/v1/image/compare_faces",
        threshold: float = 0.72,
    ) -> Tuple[float, bool]:
        tmp_files = []
        similarity = 0.0
        same = False
        try:
            pil_a = tensor_to_pil(image_a)
            pil_b = tensor_to_pil(image_b)
            f1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            pil_a.save(f1, format="PNG")
            f1.close()
            tmp_files.append(f1.name)

            f2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            pil_b.save(f2, format="PNG")
            f2.close()
            tmp_files.append(f2.name)

            with open(f1.name, "rb") as fa, open(f2.name, "rb") as fb:
                resp = requests.post(
                    api_url,
                    files={
                        "image_a": ("image_a.png", fa, "image/png"),
                        "image_b": ("image_b.png", fb, "image/png"),
                    },
                    timeout=30,
                )
            if resp.status_code == 200:
                data = resp.json()
                similarity = float(data.get("similarity", 0.0))
                same = similarity >= threshold
        except Exception:
            similarity = 0.0
            same = False
        finally:
            for p in tmp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass
        return similarity, same


__all__ = ["CompareFacesNode"]
