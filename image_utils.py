import torch
import logging
from PIL import Image
import io
import logging


def image_to_bytes(image_tensor):
    """Converts a PyTorch tensor to a PNG byte stream."""
    image_pil = tensor_to_pil(image_tensor)  # ✅ Convert tensor to PIL image
    with io.BytesIO() as buffer:
        image_pil.save(buffer, format="PNG")  # ✅ Save as PNG bytes
        return buffer.getvalue()
    
def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor (B, C, H, W) or (C, H, W) to a PIL image (H, W, C)."""
    if image is None:
        return None  # Handle missing images safely

    logging.debug(f"Original tensor shape: {image.shape}")

    # Remove batch dimension if exists (B, C, H, W) → (C, H, W)
    if image.ndim == 4:
        image = image.squeeze(0)

    # Handle grayscale images (1, H, W) by repeating channels
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.repeat(3, 1, 1)  # Convert grayscale to RGB

    # Ensure (H, W, C) format for PIL
    if image.ndim == 3 and image.shape[0] in [3, 4]:  # RGB or RGBA
        image = image.permute(1, 2, 0)  # (C, H, W) → (H, W, C)

    # Convert to NumPy and scale to uint8
    image_np = (image.cpu().numpy() * 255).clip(0, 255).astype("uint8")

    logging.debug(f"Processed image shape: {image_np.shape}")

    return Image.fromarray(image_np)