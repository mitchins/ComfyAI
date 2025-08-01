"""Unified ONNX Model Registry - Enumerates all models across chat and face servers."""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from pathlib import Path
import os

from .model_types import REFERENCE_MODELS, MODEL_QUANT_CONFIGS


@dataclass
class ModelFile:
    """Represents a single ONNX model file."""
    path: str
    repo_id: str
    size_mb: Optional[float] = None
    downloaded: bool = False
    

@dataclass
class UnifiedModel:
    """Represents a complete model (single file or multi-component configuration)."""
    id: str
    name: str
    server: Literal["chat", "face"]
    architecture: Literal["single-file", "multi-component"]
    description: str
    files: List[ModelFile] = field(default_factory=list)
    quantizations: List[str] = field(default_factory=list)
    status: Literal["not-downloaded", "partial", "downloaded", "error"] = "not-downloaded"
    total_size_mb: float = 0.0


class UnifiedModelRegistry:
    """Registry that discovers and manages all ONNX models across servers."""
    
    def __init__(self):
        self.models: Dict[str, UnifiedModel] = {}
        self._discover_models()
    
    def _discover_models(self):
        """Discover all models from chat and face servers."""
        self._discover_chat_models()
        self._discover_face_models()
    
    def _discover_chat_models(self):
        """Discover curated chat models from model_types.py."""
        # Add multi-component chat models from curated configurations
        for config_name, file_config in MODEL_QUANT_CONFIGS.items():
            model_name, quant = config_name.split('/', 1)
            
            model_id = f"chat-{model_name.lower().replace('-', '_')}"
            
            if model_id not in self.models:
                # Create base model entry
                display_name = model_name.replace('-', ' ').replace('_', ' ').title()
                
                # Get description from reference models
                description = "Multimodal chat model"
                for ref_model, spec in REFERENCE_MODELS.items():
                    if model_name in spec.repo_id:
                        description = spec.description
                        break
                
                self.models[model_id] = UnifiedModel(
                    id=model_id,
                    name=display_name,
                    server="chat",
                    architecture="multi-component",
                    description=description,
                    quantizations=[],
                    files=[]
                )
            
            # Add this quantization
            model = self.models[model_id]
            if quant not in model.quantizations:
                model.quantizations.append(quant)
            
            # Add files for this quantization
            repo_id = self._get_repo_id_for_model(model_name)
            if repo_id:
                for component, file_path in file_config.items():
                    # Add the main file
                    model_file = ModelFile(
                        path=file_path,
                        repo_id=repo_id
                    )
                    # Avoid duplicates
                    if not any(f.path == file_path and f.repo_id == repo_id for f in model.files):
                        model.files.append(model_file)
                    
                    # Check for companion data files that actually exist in the repo
                    companion_files = self._discover_actual_companion_files(repo_id, file_path)
                    for companion_path in companion_files:
                        companion_file = ModelFile(
                            path=companion_path,
                            repo_id=repo_id
                        )
                        # Avoid duplicates
                        if not any(f.path == companion_path and f.repo_id == repo_id for f in model.files):
                            model.files.append(companion_file)
    
    def _discover_face_models(self):
        """Discover face detection and embedding models."""
        # Face detection models from apps/face_api/main.py
        face_detection_models = {
            "face_detect_v1.4_s": ("Face Detection v1.4 Small", "onnx-community/insightface", "face_detect_v1.4_s/model.onnx"),
            "face_detect_v1.4_n": ("Face Detection v1.4 Normal", "onnx-community/insightface", "face_detect_v1.4_n/model.onnx"),
            "face_detect_v1.3_n": ("Face Detection v1.3 Normal", "onnx-community/insightface", "face_detect_v1.3_n/model.onnx"),
            "face_detect_v1.3_s": ("Face Detection v1.3 Small", "onnx-community/insightface", "face_detect_v1.3_s/model.onnx"),
            "face_detect_v1.2_s": ("Face Detection v1.2 Small", "onnx-community/insightface", "face_detect_v1.2_s/model.onnx"),
        }
        
        for model_key, (display_name, repo_id, file_path) in face_detection_models.items():
            model_id = f"face-{model_key.replace('_', '-')}"
            
            self.models[model_id] = UnifiedModel(
                id=model_id,
                name=display_name,
                server="face",
                architecture="single-file",
                description="ONNX face detection model for identifying faces in images",
                files=[ModelFile(path=file_path, repo_id=repo_id)]
            )
        
        # Face embedding models
        face_embedding_models = {
            "arcface_resnet100": ("ArcFace ResNet100", "openailab/onnx-arcface-resnet100-ms1m", "model.onnx"),
            "clip_vit_base": ("CLIP ViT Base", "Xenova/clip-vit-base-patch32", "onnx/vision_model.onnx"),
        }
        
        # Additional face detection models from presets
        additional_face_detection_models = {
            "real_face_detect_v1.4_s": ("Real Face Detection v1.4 Small", "deepghs/real_face_detection", "face_detect_v1.4_s/model.onnx"),
            "real_face_detect_v1.4_n": ("Real Face Detection v1.4 Normal", "deepghs/real_face_detection", "face_detect_v1.4_n/model.onnx"),
            "anime_face_detect_v1.4_s": ("Anime Face Detection v1.4 Small", "deepghs/anime_face_detection", "face_detect_v1.4_s/model.onnx"),
        }
        
        for model_key, (display_name, repo_id, file_path) in face_embedding_models.items():
            model_id = f"face-{model_key.replace('_', '-')}"
            
            # Determine description based on model type
            if "clip" in model_key.lower():
                description = "ONNX CLIP model for anime/CG face embedding and similarity comparison"
            else:
                description = "ONNX face embedding model for generating face feature vectors"
            
            self.models[model_id] = UnifiedModel(
                id=model_id,
                name=display_name,
                server="face",
                architecture="single-file", 
                description=description,
                files=[ModelFile(path=file_path, repo_id=repo_id)]
            )
        
        # Add additional face detection models from presets
        for model_key, (display_name, repo_id, file_path) in additional_face_detection_models.items():
            model_id = f"face-{model_key.replace('_', '-')}"
            
            # Determine description based on model type
            if "anime" in model_key.lower():
                description = "ONNX face detection model specialized for anime-style images"
            elif "real" in model_key.lower():
                description = "ONNX face detection model optimized for real photographs"
            else:
                description = "ONNX face detection model for identifying faces in images"
            
            self.models[model_id] = UnifiedModel(
                id=model_id,
                name=display_name,
                server="face",
                architecture="single-file",
                description=description,
                files=[ModelFile(path=file_path, repo_id=repo_id)]
            )
    
    def _get_repo_id_for_model(self, model_name: str) -> Optional[str]:
        """Get repository ID for a chat model name."""
        model_repo_mapping = {
            "Gemma-3n-E2B-it-ONNX": "onnx-community/gemma-3n-E2B-it-ONNX",
            "SmolVLM-256M-Instruct": "HuggingFaceTB/SmolVLM-256M-Instruct",
        }
        return model_repo_mapping.get(model_name)
    
    def _get_companion_data_files(self, main_file_path: str) -> List[str]:
        """Get companion data files for a main ONNX file.
        
        Returns _data.onnx and _data_0.onnx, _data_1.onnx, etc. files
        that are actually expected based on the model.
        """
        companion_files = []
        
        # Remove .onnx extension to get base name
        if main_file_path.endswith('.onnx'):
            base_path = main_file_path[:-5]  # Remove '.onnx'
            
            # Only add _data.onnx for now - the numbered ones will be added
            # based on actual repository scanning or known patterns
            data_file = f"{base_path}_data.onnx"
            companion_files.append(data_file)
            
            # For now, only add a few numbered data files that are common
            # This can be expanded based on actual model requirements
            for i in range(3):  # 0, 1, 2 are most common
                numbered_data_file = f"{base_path}_data_{i}.onnx"
                companion_files.append(numbered_data_file)
        
        return companion_files
    
    def _discover_actual_companion_files(self, repo_id: str, main_file_path: str) -> List[str]:
        """Discover companion data files that actually exist in the repository."""
        companion_files = []
        
        try:
            from huggingface_hub import list_repo_files
            
            # Get all files in the repository
            repo_files = list_repo_files(repo_id)
            
            if main_file_path.endswith('.onnx'):
                # Check for multiple companion file patterns:
                
                # Pattern 1: filename.onnx_data (like decoder_model_merged_fp16.onnx_data)
                data_suffix_file = f"{main_file_path}_data"
                if data_suffix_file in repo_files:
                    companion_files.append(data_suffix_file)
                
                # Pattern 2: filename_data.onnx (like decoder_model_merged_fp16_data.onnx)
                base_path = main_file_path[:-5]  # Remove '.onnx'
                data_file = f"{base_path}_data.onnx"
                if data_file in repo_files:
                    companion_files.append(data_file)
                
                # Pattern 3: numbered data files filename_data_0.onnx, filename_data_1.onnx, etc.
                for i in range(20):  # Check up to _data_19.onnx
                    numbered_data_file = f"{base_path}_data_{i}.onnx"
                    if numbered_data_file in repo_files:
                        companion_files.append(numbered_data_file)
                    else:
                        # Stop at first missing numbered file (they're usually sequential)
                        break
                        
        except Exception as e:
            # If we can't fetch repo files, fall back to conservative companion file detection
            # This maintains backwards compatibility
            if main_file_path.endswith('.onnx'):
                base_path = main_file_path[:-5]
                # Only add _data.onnx, not numbered variants
                companion_files.append(f"{base_path}_data.onnx")
        
        return companion_files
    
    def _get_files_for_quantization(self, model: UnifiedModel, quantization: str) -> List[ModelFile]:
        """Get files that belong to a specific quantization."""
        if model.architecture == "single-file":
            return model.files
        
        # For multi-component models, filter by quantization suffix
        quantization_patterns = {
            'Q4': ['_q4'],
            'Q4_MIXED': ['_q4', '_quantized'],
            'Q4_F16': ['_q4f16', '_q4_f16'],
            'FP16': ['_fp16', '_16'],
            'FP32': ['_fp32', '_32', '.onnx'],  # FP32 often has no suffix
            'INT8': ['_int8'],
            'BNB4': ['_bnb4'],
            'UINT8': ['_uint8'],
            'QUANTIZED': ['_quantized']
        }
        
        patterns = quantization_patterns.get(quantization, [])
        if not patterns:
            return []
        
        matched_files = []
        for file in model.files:
            file_lower = file.path.lower()
            
            # Special case for FP32 - match files without specific quantization suffixes
            if quantization == 'FP32':
                has_other_quant = any(
                    pattern in file_lower 
                    for quant, patterns in quantization_patterns.items() 
                    if quant != 'FP32' 
                    for pattern in patterns
                    if pattern != '.onnx'
                )
                if not has_other_quant:
                    matched_files.append(file)
            else:
                # Match files containing any of the quantization patterns
                if any(pattern in file_lower for pattern in patterns):
                    matched_files.append(file)
        
        return matched_files
    
    def get_all_models(self) -> List[UnifiedModel]:
        """Get all discovered models."""
        return list(self.models.values())
    
    def get_model(self, model_id: str) -> Optional[UnifiedModel]:
        """Get a specific model by ID."""
        return self.models.get(model_id)
    
    def get_models_by_server(self, server: Literal["chat", "face"]) -> List[UnifiedModel]:
        """Get all models for a specific server."""
        return [model for model in self.models.values() if model.server == server]
    
    def update_download_status(self):
        """Update download status for all models by checking HuggingFace cache."""
        import os
        from pathlib import Path
        
        try:
            # Try HuggingFace cache scanning first
            from huggingface_hub import scan_cache_dir
            hf_cache_info = scan_cache_dir()
            cached_repos = {repo.repo_id: repo for repo in hf_cache_info.repos}
            
            for model in self.models.values():
                downloaded_files = 0
                total_files = len(model.files)
                total_size = 0.0
                
                for file in model.files:
                    file.downloaded = False
                    file.size_mb = None
                    
                    if file.repo_id in cached_repos:
                        repo_info = cached_repos[file.repo_id]
                        # Check if specific file exists
                        for revision in repo_info.revisions:
                            for file_info in revision.files:
                                if file.path in file_info.file_path:
                                    file.downloaded = True
                                    file.size_mb = file_info.size_on_disk / (1024 * 1024)
                                    downloaded_files += 1
                                    total_size += file.size_mb
                                    break
                
                model.total_size_mb = total_size
                
                if downloaded_files == 0:
                    model.status = "not-downloaded"
                elif downloaded_files == total_files:
                    model.status = "downloaded"
                else:
                    # For multi-component models, show quantization count instead of "partial"
                    if model.architecture == "multi-component" and model.quantizations:
                        downloaded_quants = 0
                        for quant in model.quantizations:
                            quant_files = self._get_files_for_quantization(model, quant)
                            if len(quant_files) == 0:
                                continue
                                
                            quant_downloaded = sum(1 for f in quant_files if f.downloaded)
                            
                            # Only count a quantization as "downloaded" if ALL files are present
                            # This ensures the quantization is actually usable
                            if quant_downloaded == len(quant_files) and len(quant_files) > 0:
                                downloaded_quants += 1
                        
                        total_quants = len(model.quantizations)
                        if downloaded_quants > 0:
                            model.status = f"{downloaded_quants}/{total_quants}-quants"
                        else:
                            model.status = "partial-files"
                    else:
                        model.status = "partial"
                    
        except Exception as e:
            # If cache scanning fails, fall back to basic file existence check
            try:
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                
                for model in self.models.values():
                    downloaded_files = 0
                    total_files = len(model.files)
                    
                    for file in model.files:
                        file.downloaded = False
                        file.size_mb = None
                        
                        # Try to find the file in cache directory
                        repo_cache_name = f"models--{file.repo_id.replace('/', '--')}"
                        possible_paths = [
                            cache_dir / repo_cache_name / "snapshots" / "*" / file.path,
                            cache_dir / repo_cache_name / "blobs" / "*"
                        ]
                        
                        # Simple existence check
                        repo_dir = cache_dir / repo_cache_name
                        if repo_dir.exists():
                            for snapshot_dir in repo_dir.glob("snapshots/*"):
                                file_path = snapshot_dir / file.path
                                if file_path.exists():
                                    file.downloaded = True
                                    file.size_mb = file_path.stat().st_size / (1024 * 1024)
                                    downloaded_files += 1
                                    break
                    
                    if downloaded_files == 0:
                        model.status = "not-downloaded"
                    elif downloaded_files == total_files:
                        model.status = "downloaded"
                    else:
                        # For multi-component models, show quantization count instead of "partial"
                        if model.architecture == "multi-component" and model.quantizations:
                            downloaded_quants = 0
                            for quant in model.quantizations:
                                quant_files = self._get_files_for_quantization(model, quant)
                                if len(quant_files) == 0:
                                    continue
                                    
                                quant_downloaded = sum(1 for f in quant_files if f.downloaded)
                                
                                # Only count a quantization as "downloaded" if ALL files are present
                                if quant_downloaded == len(quant_files) and len(quant_files) > 0:
                                    downloaded_quants += 1
                            
                            total_quants = len(model.quantizations)
                            if downloaded_quants > 0:
                                model.status = f"{downloaded_quants}/{total_quants}-quants"
                            else:
                                model.status = "partial-files"
                        else:
                            model.status = "partial"
                        
            except Exception:
                # Final fallback - set all to not-downloaded instead of error
                for model in self.models.values():
                    model.status = "not-downloaded"
                    for file in model.files:
                        file.downloaded = False
                        file.size_mb = None


# Global registry instance
registry = UnifiedModelRegistry()