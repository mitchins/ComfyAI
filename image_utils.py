try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

import logging
from PIL import Image
import io


def image_to_bytes(image_tensor):
    """Converts a PyTorch tensor to PNG bytes."""
    image_pil = tensor_to_pil(image_tensor)
    with io.BytesIO() as buffer:
        image_pil.save(buffer, format="PNG")
        return buffer.getvalue()


def tensor_to_pil(image):
    """Convert a PyTorch tensor (B, C, H, W) or (C, H, W) to a PIL image."""
    if torch is None:
        raise ImportError("torch is required for tensor operations")
    if image is None:
        return None

    logging.debug(f"Original tensor shape: {image.shape}")

    if image.ndim == 4:
        image = image.squeeze(0)

    if image.ndim == 3 and image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    if image.ndim == 3 and image.shape[0] in [3, 4]:
        image = image.permute(1, 2, 0)

    image_np = (image.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    logging.debug(f"Processed image shape: {image_np.shape}")
    return Image.fromarray(image_np)
