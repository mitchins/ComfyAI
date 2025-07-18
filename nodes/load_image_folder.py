import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import logging

def is_unit_test():
    """Checks if the code is running in a unit test environment."""
    return os.getenv("UNIT_TEST_MODE") == "1"

if is_unit_test():
    import types
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")
    sys.modules["folder_paths"].get_input_directory = lambda: "/tmp"
    sys.modules["folder_paths"].get_annotated_filepath = lambda x: x

try:
    import torch
except ImportError:
    if not is_unit_test():
        raise
    # Mock torch for unit tests
    import types
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda x: x
    torch.stack = lambda x, dim: x[0] if len(x) == 1 else x
    torch.Tensor = type("MockTensor", (), {"shape": [1, 64, 64, 3]})

try:
    import folder_paths
except ImportError:
    pass


class LoadImageFolder:
    """
    Load all images from a folder and return them as a list of IMAGE tensors.
    Unlike the built-in LoadImage node which emits per-image workflows,
    this node emits all images as a single batch for processing.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory() if not is_unit_test() else "/tmp"
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_extensions": ("STRING", {"default": ".jpg,.jpeg,.png,.bmp,.webp,.tiff", "tooltip": "Comma-separated list of image extensions to include"}),
                "max_images": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Maximum number of images to load. 0 = no limit"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_images"
    CATEGORY = "image"
    DESCRIPTION = "Load all images from a folder as a batch of IMAGE tensors"

    def load_images(self, folder_path, image_extensions=".jpg,.jpeg,.png,.bmp,.webp,.tiff", max_images=0):
        if not folder_path:
            raise ValueError("folder_path cannot be empty")
        
        # Handle relative paths
        if not os.path.isabs(folder_path):
            if not is_unit_test():
                base_dir = folder_paths.get_input_directory()
                folder_path = os.path.join(base_dir, folder_path)
        
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Parse extensions
        extensions = [ext.strip().lower() for ext in image_extensions.split(",")]
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        
        # Find all image files
        image_files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(folder_path, file))
        
        image_files.sort()  # Ensure consistent ordering
        
        if not image_files:
            raise ValueError(f"No image files found in {folder_path} with extensions {extensions}")
        
        # Apply max_images limit
        if max_images > 0:
            image_files = image_files[:max_images]
        
        logging.info(f"Loading {len(image_files)} images from {folder_path}")
        
        # Load all images
        images = []
        for image_path in image_files:
            try:
                img = Image.open(image_path)
                img = img.convert("RGB")  # Ensure RGB format
                
                # Convert to tensor (H, W, C) format as expected by ComfyUI
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)
                
                images.append(img_tensor)
                
            except Exception as e:
                logging.warning(f"Failed to load image {image_path}: {e}")
                continue
        
        if not images:
            raise ValueError("No images could be loaded successfully")
        
        # Stack all images into a single batch tensor
        # All images must have the same dimensions for batching
        # If they don't, we'll need to resize them to a common size
        heights = [img.shape[0] for img in images]
        widths = [img.shape[1] for img in images]
        
        if len(set(heights)) > 1 or len(set(widths)) > 1:
            # Resize all images to the size of the first image
            target_height, target_width = heights[0], widths[0]
            resized_images = []
            
            for img_tensor in images:
                img_pil = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8))
                img_pil = img_pil.resize((target_width, target_height), Image.LANCZOS)
                img_array = np.array(img_pil).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)
                resized_images.append(img_tensor)
            
            images = resized_images
            logging.info(f"Resized all images to {target_width}x{target_height}")
        
        # Stack into batch: (N, H, W, C)
        batch_tensor = torch.stack(images, dim=0)
        
        logging.info(f"Loaded batch of {batch_tensor.shape[0]} images with shape {batch_tensor.shape}")
        
        return (batch_tensor,)