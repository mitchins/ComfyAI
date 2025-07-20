import os
from typing import List, Dict
import onnx
from .model_types import ModelType

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils._cache_manager import scan_cache_dir, _try_delete_path
    from huggingface_hub.errors import CacheNotFound
except Exception:  # pragma: no cover - optional dependency
    HfApi = None  # type: ignore
    hf_hub_download = None  # type: ignore
    scan_cache_dir = None  # type: ignore
    _try_delete_path = None  # type: ignore
    CacheNotFound = Exception  # type: ignore


def _classify_model_type(input_names: List[str]) -> ModelType:
    """Classify model type based on input names."""
    has_text_inputs = any(name in input_names for name in ["input_ids", "attention_mask", "token_type_ids"])
    has_vision_inputs = any(name in input_names for name in ["pixel_values", "image", "images"])
    
    if has_text_inputs and has_vision_inputs:
        return ModelType.VISION_LLM
    elif has_vision_inputs:
        return ModelType.VISION_EMBEDDER
    elif has_text_inputs:
        return ModelType.TEXT_LLM
    elif any("pixel" in name.lower() for name in input_names):
        return ModelType.VISION_EMBEDDER
    else:
        return ModelType.UNKNOWN


def _require() -> None:
    if HfApi is None or hf_hub_download is None or scan_cache_dir is None:
        raise RuntimeError("huggingface_hub not available")


def list_repo_files(repo_id: str) -> List[Dict[str, int]]:
    """Return files in a remote repository with size information."""
    _require()
    api = HfApi()
    info = api.repo_info(repo_id, files_metadata=True)
    return [{"path": f.rfilename, "size": f.size} for f in info.siblings]


def list_cached_entries() -> List[Dict[str, float]]:
    """Return entries currently present in the local HuggingFace cache."""
    _require()
    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return []
    entries = []
    for repo in cache.repos:
        for rev in repo.revisions:
            for file in rev.files:
                entry = {
                    "repo": repo.repo_id,
                    "path": file.file_name,
                    "size": file.size_on_disk,
                    "last_used": file.blob_last_accessed,
                    "framework": None,
                    "kind": ModelType.UNKNOWN.value,
                    "inputs": [],
                }
                if file.file_name.endswith(".onnx"):
                    try:
                        model = onnx.load_model(file.file_path, load_external_data=False)
                        # Extract producer_name from metadata_props
                        framework = None
                        if model.metadata_props:
                            for prop in model.metadata_props:
                                if prop.key == "producer_name":
                                    framework = prop.value
                                    break
                        entry["framework"] = framework
                        # Determine kind using the classification function
                        input_names = [inp.name for inp in model.graph.input]
                        entry["kind"] = _classify_model_type(input_names).value
                        # Collect inputs
                        inputs = []
                        for inp in model.graph.input:
                            dims = []
                            for dim in inp.type.tensor_type.shape.dim:
                                if dim.HasField("dim_value"):
                                    dims.append(dim.dim_value)
                                else:
                                    dims.append(None)
                            inputs.append({inp.name: dims})
                        entry["inputs"] = inputs
                    except Exception:
                        entry["framework"] = None
                        entry["kind"] = ModelType.UNKNOWN.value
                        entry["inputs"] = []
                entries.append(entry)
    return entries


def download_file(repo_id: str, file_path: str) -> None:
    """Ensure a file from HF repo is present in the local cache."""
    _require()
    hf_hub_download(repo_id=repo_id, filename=file_path)


# Helper to download only ONNX file and minimal config for a model
def download_model_components(repo_id: str, onnx_path: str) -> None:
    """
    Download only the necessary components for an ONNX model:
    - config.json from the repo root
    - the specified ONNX file (relative path)
    """
    _require()
    # Download config.json
    try:
        hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception:
        # config.json may not exist at root, ignore if not found
        pass
    # Download the ONNX file
    hf_hub_download(repo_id=repo_id, filename=onnx_path)


def delete_cached_file(repo_id: str, file_path: str) -> None:
    """Delete a cached file and its blob from the HF cache."""
    _require()
    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return
    for repo in cache.repos.values():
        if repo.repo_id != repo_id:
            continue
        for rev in repo.revisions:
            for file in rev.files:
                if file.file_name == os.path.basename(file_path):
                    _try_delete_path(file.file_path)
                    _try_delete_path(file.blob_path)
