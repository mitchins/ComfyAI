# Developing ComfyUI Nodes

This document explains how to create new nodes for ComfyAI.
It is intended as both a quick guide and a reference.

## Node discovery

Any Python file under `nodes/` can define one or more node classes.
`nodes/__init__.py` exposes these classes through `NODE_CLASS_MAPPINGS`
so ComfyUI can register them automatically. Registration is skipped
when the environment variable `UNIT_TEST_MODE` is set.

```python
import os

if os.getenv("UNIT_TEST_MODE") != "1":
    from .vllm_query import (
        VLLMTextQuery,
        VLLMImageQuery,
        VLLMDualImageQuery,
    )
    from .conditional_save_image import ConditionalSaveImage
    from .compare_faces import CompareFacesNode

    NODE_CLASS_MAPPINGS = {
        "VLLMTextQuery": VLLMTextQuery,
        "VLLMImageQuery": VLLMImageQuery,
        "VLLMDualImageQuery": VLLMDualImageQuery,
        "ConditionalSaveImage": ConditionalSaveImage,
        "CompareFacesNode": CompareFacesNode,
    }
``` 
Only when `UNIT_TEST_MODE` is **not** set to `"1"` are the node classes
added to `NODE_CLASS_MAPPINGS` so ComfyUI can discover them.

## Basic node structure

A node is a class with a few class attributes and one public method.
A minimal node looks like:

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "custom"

    def run(self, text: str):
        return (text.upper(),)
```

### INPUT_TYPES
The value returned by `INPUT_TYPES()` describes the fields shown in the UI.
Each entry maps an input name to a **type** string such as `STRING`,
`IMAGE`, `INT`, `FLOAT` or `BOOLEAN`.  An optional dictionary can
specify defaults or other UI hints.

Below is a real example from `_BaseVLLMQuery`:

```python
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "text_query": ("STRING", {"default": "Describe the image.", "multiline": True}),
            "api_endpoint": ("STRING", {"default": "", "multiline": False}),
            "api_model": ("STRING", {"default": "gpt-3.5-turbo", "multiline": False}),
            "api_key": ("STRING", {"default": "", "multiline": False}),
        }
        if cls.required_images >= 1:
            required["image"] = ("IMAGE",)
        if cls.required_images >= 2:
            required["reference_image"] = ("IMAGE",)
        return {"required": required}
``` 

### Return values

The outputs of your node are described by `RETURN_TYPES`. The method named
by `FUNCTION` must return a tuple of the same length.  Optionally it may
return a dictionary with a `"ui"` field that will be displayed in
ComfyUI.

`ConditionalSaveImage.save_images` demonstrates this pattern:

```python
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
```

The return value is a dictionary with a `ui` entry containing a list of
images to display. Returning a tuple instead of a dict is also valid
when you only output tensors or primitive types.

### Error propagation

Fatal problems should be reported by raising an exception. Recoverable
errors may be caught and turned into default output values.
The `_BaseVLLMQuery` class, for instance, ensures `api_endpoint` is
provided and propagates encoding errors via the Python logging system:

```python
    def run(self, **inputs):
        api_endpoint = inputs.get("api_endpoint", "").strip()
        if not api_endpoint:
            raise ValueError("api_endpoint is required")
        api_model = inputs.get("api_model", "gpt-3.5-turbo")
        api_key = inputs.get("api_key", "") or None
        text_query = inputs.get("text_query", "")

        images = []
        if self.required_images >= 1:
            images.append(inputs.get("image"))
        if self.required_images >= 2:
            images.append(inputs.get("reference_image"))

        content = [{"type": "text", "text": text_query}]

        try:
            for img in images:
                if img is not None:
                    img_b = image_to_bytes(img)
                    encoded = base64.b64encode(img_b).decode()
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        except Exception:
            logging.exception("Failed to encode image inputs")

        messages = [{"role": "user", "content": content}]

        result = chat_completion(api_endpoint, api_model, messages, api_key=api_key)
        bool_output = fuzzy_match_bool(result)
        if bool_output is None:
            bool_output = False
        return result, bool_output, int(bool_output)
```

If `api_endpoint` is missing the method raises a `ValueError` and the
error surfaces in ComfyUI. Encoding issues are logged but the node still
returns a result.

`CompareFacesNode` demonstrates another pattern where external API
failures fall back to a safe value:

```python
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
```

If the remote face API fails, the node returns `0.0` and `False` instead
of raising an exception. Temporary files are always cleaned up in the
`finally` block.

## Adding your node

1. Create a new Python file in `nodes/` and implement the class as
   described above.
2. Register the class in `NODE_CLASS_MAPPINGS` inside `nodes/__init__.py`.
3. Provide meaningful `CATEGORY`, `RETURN_TYPES` and `INPUT_TYPES` so the
   UI can display your node correctly.

That's it – restart ComfyUI and your node should appear in the menu.

## New Nodes Added

### LoadImageFolder
- **Purpose**: Load all images from a folder as a batch instead of individual workflow iterations
- **Key Features**: 
  - Batch processing of multiple images
  - Configurable file extensions
  - Automatic image resizing for consistent batching
  - Max images limit option

### ImageSimilarityChecker
- **Purpose**: Use CLIP embeddings to check image similarity against reference images
- **Key Features**:
  - CLIP-based embeddings for semantic similarity
  - Configurable similarity threshold
  - Reference consistency metrics (0.0-1.0)
  - Detailed debug information

## Reference Consistency Metrics

The `reference_consistency` output (0.0-1.0) indicates how similar your reference images are to each other:

- **0.8-1.0**: Very consistent style (good for game backgrounds, artistic styles)
- **0.6-0.8**: Moderately consistent (some variation in style/composition)
- **0.4-0.6**: Mixed styles (may need more focused reference set)
- **0.0-0.4**: Very diverse references (may not be suitable for style matching)

For game backgrounds or artistic style matching, you typically want consistency scores above 0.7.

## Testing

All nodes include comprehensive unit tests with proper mocking for the test environment. Run tests with:

```bash
python -m pytest tests/test_load_image_folder.py tests/test_image_similarity_checker.py -v
```

## Dependencies

The new nodes require:
- `transformers>=4.21.0` (for CLIP model)
- `torch>=1.13.0` (for tensor operations)
- `pillow>=9.0` (for image processing)
- `numpy>=1.23` (for array operations)
