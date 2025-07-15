# This node saves images to the output directory if a condition is met.
# The condition is a boolean input, and the images are the images to save.
# The filename_prefix is the prefix for the file to save.
# Shamelessly copied from ComfyUI's built-in save image node. 😅

import os
import sys
# 🛠️ Detect if we are running under unittest (do not rely on Comfy)
def is_unit_test():
    """Checks if the code is running in a unit test environment."""
    return os.getenv("UNIT_TEST_MODE") == "1"

if is_unit_test():
    import types
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")  # Stub it out!
    sys.modules["folder_paths"].some_fake_function = lambda: None  # Stub any needed attributes
    sys.modules["comfy.cli_args"] = types.ModuleType("comfy.cli_args")  # Stub it out!
    sys.modules["comfy.cli_args"].args = {}  # Stub `args` as an empty dict or object

# ✅ Now, safely import everything
import folder_paths  # This won't fail under tests
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
from comfy.cli_args import args
import json

import logging

class ConditionalSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "condition": ("BOOLEAN",),  # Now a connectable input
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, condition, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        if not condition:
            logging.info(f"Condition not met, skipping saving images. Condition: {condition}")
            return { "ui": { "images": [] } }
        else:
            logging.info(f"Condition met, saving images. Condition: {condition}")
            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
            results = list()
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1
            return { "ui": { "images": results } }