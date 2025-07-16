import os
import tempfile
import time
import requests

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from custom_nodes.image_utils import tensor_to_pil


def fetch_face_api_response(
    api_url: str,
    image_a: str,
    image_b: str,
    timeout: float = 5.0,
    retries: int = 3,
    backoff: float = 0.5,
) -> dict:
    """Send two image files to the face API and return the parsed JSON."""
    delay = backoff
    for attempt in range(retries):
        with open(image_a, "rb") as f1, open(image_b, "rb") as f2:
            files = {"image_a": f1, "image_b": f2}
            try:
                resp = requests.post(api_url, files=files, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt == retries - 1:
                    raise
        time.sleep(delay)
        delay *= 2


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

    def run(
        self,
        image_a,
        image_b,
        api_url: str,
        threshold: float = 0.72,
    ) -> tuple[float, bool]:
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

            try:
                data = fetch_face_api_response(api_url, tmp_files[0], tmp_files[1])
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


__all__ = ["CompareFacesNode", "fetch_face_api_response"]
