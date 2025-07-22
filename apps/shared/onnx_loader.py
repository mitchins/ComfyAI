"""ONNX model loader with support for multi-component models."""

from typing import Dict, Optional, List, Any
import logging
import numpy as np


import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
ONNX_AVAILABLE = True
    # Mock classes for testing environments
    # class MockSession:
    #     def run(self, *args, **kwargs):
    #         return [np.random.rand(1, 10, 32000)]
    #     def get_inputs(self):
    #         return [type('MockInput', (), {'name': 'input_ids'})()]
    
    # class MockTokenizer:
    #     def __init__(self, *args, **kwargs):
    #         pass
    #     def __call__(self, *args, **kwargs):
    #         return {'input_ids': np.array([[1, 2, 3]])}
    #     @property 
    #     def eos_token_id(self):
    #         return 2
    #     def decode(self, *args, **kwargs):
    #         return "mocked response"
    #     @classmethod
    #     def from_pretrained(cls, *args, **kwargs):
    #         return cls()
    
    # def mock_hf_hub_download(*args, **kwargs):
    #     return "/mock/path/model.onnx"
    
    # ort = type('MockORT', (), {'InferenceSession': MockSession})
    # AutoTokenizer = MockTokenizer
    # hf_hub_download = mock_hf_hub_download
    # ONNX_AVAILABLE = False
    # print(f"Warning: ONNX dependencies not available: {e}")

from .model_types import (
    ONNXModelConfig, get_onnx_model_config, create_position_ids, initialize_kv_cache, 
    REFERENCE_MODELS, MODEL_QUANT_CONFIGS, get_curated_model_config, get_available_model_quants,
    get_smallest_quant_for_model
)


logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """Loader for multi-component ONNX models."""
    
    def __init__(self):
        self.sessions_cache: Dict[str, Dict[str, Any]] = {}
        self.tokenizers_cache: Dict[str, AutoTokenizer] = {}
        self.configs_cache: Dict[str, ONNXModelConfig] = {}
    
    def parse_model_name(self, name: str, default_file: str = "model.onnx") -> tuple[str, str]:
        """Split model identifier into HF repo and filename."""
        if ":" in name:
            repo_id, filename = (name.split(":", 1) + [default_file])[:2]
        else:
            parts = name.split("/")
            if name.endswith(".onnx") and len(parts) >= 3:
                repo_id = "/".join(parts[:-1])
                filename = parts[-1]
            else:
                repo_id = name
                filename = default_file
        return repo_id, filename
    
    def extract_quantization_suffix(self, filename: str) -> str:
        """Extract quantization suffix from filename."""
        if filename == "model.onnx" or "decoder_model_merged" not in filename:
            return ""
        
        base_name = filename.replace(".onnx", "").replace("decoder_model_merged", "")
        if base_name and base_name.startswith("_"):
            return base_name
        return ""
    
    def download_components(self, repo_id: str, config: ONNXModelConfig, quant_suffix: str = "") -> Dict[str, str]:
        """Download required ONNX components and return their paths."""
        component_paths = {}
        
        # Define fallback quantization options if the requested one doesn't exist
        # For Gemma 3n: _q4 works for decoder, but not for embed_tokens/vision/audio
        if "gemma" in repo_id.lower():
            fallback_suffixes = ["_q4", "_int8", "_uint8", "_quantized", "_fp16", ""]
        else:
            fallback_suffixes = ["_q4", "_fp16", "_int8", "_quantized", "_uint8", ""]
        
        if quant_suffix and quant_suffix not in fallback_suffixes:
            fallback_suffixes.insert(0, quant_suffix)
        
        for component_name, filename_pattern in config.components.items():
            component_downloaded = False
            
            # Try the requested quantization first, then fallbacks
            for suffix in fallback_suffixes:
                filename = filename_pattern.format(suffix=suffix)
                if config.subfolder:
                    filename = f"{config.subfolder}/{filename}"
                
                try:
                    path = hf_hub_download(repo_id=repo_id, filename=filename)
                    
                    # Also download companion .onnx_data file if it exists
                    if filename.endswith('.onnx'):
                        data_filename = filename + '_data'
                        try:
                            data_path = hf_hub_download(repo_id=repo_id, filename=data_filename)
                            logger.info(f"Downloaded companion data file: {data_filename} -> {data_path}")
                        except Exception as e:
                            # .onnx_data file doesn't exist, which is fine for quantized models
                            logger.debug(f"No companion data file found for {filename}: {e}")
                            pass
                    
                    component_paths[component_name] = path
                    logger.info(f"Downloaded {component_name}: {filename}")
                    component_downloaded = True
                    break
                except Exception as e:
                    logger.debug(f"Failed to download {component_name} ({filename}): {e}")
                    continue
            
            # If we couldn't download any version of this component
            if not component_downloaded:
                logger.warning(f"Could not download any version of {component_name}")
                # For optional components like vision/audio, this might be okay
                if component_name not in ['vision', 'audio']:
                    raise ValueError(f"Required component {component_name} not found in repository {repo_id}")
        
        return component_paths
    
    def download_auxiliary_files(self, repo_id: str) -> None:
        """Download auxiliary files like tokenizer config."""
        auxiliary_files = [
            "config.json",
            "merges.txt",
            "vocab.json", 
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
            "generation_config.json",
            "added_tokens.json",
            "chat_template.json"
        ]
        
        for aux_file in auxiliary_files:
            try:
                hf_hub_download(repo_id=repo_id, filename=aux_file)
            except Exception:
                # These files are optional
                pass
    
    def load_curated_model(self, model_quant_name: str) -> tuple[Dict[str, Any], AutoTokenizer, ONNXModelConfig]:
        """Load ONNX model using curated model/quant configurations."""
        if model_quant_name in self.sessions_cache:
            return (
                self.sessions_cache[model_quant_name],
                self.tokenizers_cache[model_quant_name], 
                self.configs_cache[model_quant_name]
            )
        
        # Get curated component paths
        component_config = get_curated_model_config(model_quant_name)
        if component_config is None:
            available_configs = get_available_model_quants()
            raise ValueError(
                f"Model/quant '{model_quant_name}' not in curated list. "
                f"Available: {', '.join(available_configs[:5])}..." if len(available_configs) > 5 
                else f"Available: {', '.join(available_configs)}"
            )
        
        # Extract repo_id from model name (before the slash)
        model_name = model_quant_name.split('/')[0]
        repo_id = self._get_repo_id_from_model_name(model_name)
        if repo_id is None:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Get model config from reference models
        config = get_onnx_model_config(repo_id)
        if config is None:
            raise ValueError(f"No configuration found for model: {repo_id}")
        
        # Download auxiliary files
        self.download_auxiliary_files(repo_id)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=True)
        
        # Download components using curated paths
        component_paths = self.download_curated_components(repo_id, component_config)
        
        # Load ONNX sessions
        sessions = {}
        for component_name, path in component_paths.items():
            print(f"DEBUG CURATED: About to create InferenceSession for {component_name}")
            print(f"DEBUG CURATED: ort type: {type(ort)}")
            print(f"DEBUG CURATED: ort: {ort}")
            print(f"DEBUG CURATED: hasattr InferenceSession: {hasattr(ort, 'InferenceSession')}")
            print(ort)
            if hasattr(ort, 'InferenceSession'):
                print(f"DEBUG CURATED: InferenceSession: {ort.InferenceSession}")
            else:
                print(f"DEBUG CURATED: ort attributes: {[attr for attr in dir(ort) if not attr.startswith('_')]}")
            sessions[component_name] = ort.InferenceSession(path)
            logger.info(f"Loaded {component_name} from {path}")
        
        # Cache everything
        self.sessions_cache[model_quant_name] = sessions
        self.tokenizers_cache[model_quant_name] = tokenizer
        self.configs_cache[model_quant_name] = config
        
        return sessions, tokenizer, config
    
    def _get_repo_id_from_model_name(self, model_name: str) -> Optional[str]:
        """Map model name to repo_id using curated model mappings."""
        model_repo_mapping = {
            "Qwen2-VL-2B-Instruct": "onnx-community/Qwen2-VL-2B-Instruct",
            "Gemma-3n-E2B-it-ONNX": "onnx-community/gemma-3n-E2B-it-ONNX", 
            "Phi-3.5-vision-instruct": "onnx-community/Phi-3.5-vision-instruct",
        }
        return model_repo_mapping.get(model_name)
    
    def download_curated_components(self, repo_id: str, component_config: Dict[str, str]) -> Dict[str, str]:
        """Download components using curated file paths."""
        component_paths = {}
        
        for component_name, filename in component_config.items():
            try:
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                
                # Also download companion .onnx_data file if it exists
                if filename.endswith('.onnx'):
                    data_filename = filename + '_data'
                    try:
                        data_path = hf_hub_download(repo_id=repo_id, filename=data_filename)
                        logger.info(f"Downloaded companion data file: {data_filename} -> {data_path}")
                    except Exception as e:
                        # .onnx_data file doesn't exist, which is fine for quantized models
                        logger.debug(f"No companion data file found for {filename}: {e}")
                        pass
                
                component_paths[component_name] = path
                logger.info(f"Downloaded {component_name}: {filename}")
            except Exception as e:
                logger.error(f"Failed to download {component_name} ({filename}): {e}")
                raise ValueError(f"Required component {component_name} not found in repository {repo_id}")
        
        return component_paths

    def load_model(self, model_name: str) -> tuple[Dict[str, Any], AutoTokenizer, ONNXModelConfig]:
        """Load ONNX model components, tokenizer, and config."""
        # Check if this is a curated model/quant combo first
        if '/' in model_name and get_curated_model_config(model_name) is not None:
            return self.load_curated_model(model_name)
        
        # Fall back to legacy loading for backwards compatibility
        if model_name in self.sessions_cache:
            return (
                self.sessions_cache[model_name],
                self.tokenizers_cache[model_name], 
                self.configs_cache[model_name]
            )
        
        repo_id, filename = self.parse_model_name(model_name)
        config = get_onnx_model_config(repo_id)
        
        if config is None:
            # Provide helpful error with available models
            try:
                available_repos = list(REFERENCE_MODELS.keys())
                raise ValueError(
                    f"Unsupported model: {repo_id}. "
                    f"Supported models: {', '.join(available_repos)}"
                )
            except:
                raise ValueError(f"Unsupported model: {repo_id}. Check model configuration.")
        
        # Download auxiliary files
        self.download_auxiliary_files(repo_id)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=True)
        
        # Extract quantization suffix and download components
        quant_suffix = self.extract_quantization_suffix(filename)
        component_paths = self.download_components(repo_id, config, quant_suffix)
        
        # Load ONNX sessions
        sessions = {}
        for component_name, path in component_paths.items():
            sessions[component_name] = ort.InferenceSession(path)
            logger.info(f"Loaded {component_name} from {path}")
        
        # Cache everything
        self.sessions_cache[model_name] = sessions
        self.tokenizers_cache[model_name] = tokenizer
        self.configs_cache[model_name] = config
        
        return sessions, tokenizer, config


class ONNXInferenceEngine:
    """Inference engine for ONNX models."""
    
    def __init__(self, sessions: Dict[str, Any], tokenizer: AutoTokenizer, config: ONNXModelConfig):
        self.sessions = sessions
        self.tokenizer = tokenizer
        self.config = config
        
        # Handle different model architectures
        if 'model' in sessions:
            # Single model file (like Granite)
            self.model = sessions['model']
            self.embed_model = None
            self.decoder_model = None
            self.prepare_inputs_embeds_model = None
            self.model_input_names = [inp.name for inp in self.model.get_inputs()]
        else:
            # Multi-component models
            self.model = None
            self.decoder_model = sessions['decoder']
            self.decoder_input_names = [inp.name for inp in self.decoder_model.get_inputs()]
            
            # Different embedding architectures
            if 'embed' in sessions:
                # Qwen2-VL style: separate embed_tokens
                self.embed_model = sessions['embed']
                self.prepare_inputs_embeds_model = None
            elif 'embed_tokens' in sessions:
                # Gemma style: embed_tokens
                self.embed_model = sessions['embed_tokens']
                self.prepare_inputs_embeds_model = None
            elif 'prepare_inputs_embeds' in sessions:
                # Phi-3.5 style: prepare_inputs_embeds
                self.embed_model = None
                self.prepare_inputs_embeds_model = sessions['prepare_inputs_embeds']
            else:
                # No embedding model found
                self.embed_model = None
                self.prepare_inputs_embeds_model = None
            
        # Optional components
        self.vision_model = sessions.get('vision') or sessions.get('vision_encoder')  # Optional
        self.audio_model = sessions.get('audio') or sessions.get('audio_encoder')   # Optional
    
    def generate_text(self, text: str, max_tokens: int = 100, images: Optional[List[str]] = None) -> str:
        """Generate text using the ONNX model."""
        if images and not self.config.has_vision:
            raise ValueError("Images provided but model doesn't support vision")
        
        # Handle single model vs multi-component models
        if self.model is not None:
            return self._generate_single_model(text, max_tokens, images)
        else:
            return self._generate_multi_component(text, max_tokens, images)
    
    def _generate_single_model(self, text: str, max_tokens: int, images: Optional[List[str]]) -> str:
        """Generate text using single ONNX model (like Granite)."""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='np')
        input_ids = inputs['input_ids']
        
        # Simple generation loop for single model
        generated_ids = input_ids.copy()
        
        for _ in range(max_tokens):
            # Run model
            model_inputs = {'input_ids': generated_ids}
            if 'attention_mask' in self.model_input_names:
                model_inputs['attention_mask'] = np.ones_like(generated_ids)
                
            outputs = self.model.run(None, model_inputs)
            logits = outputs[0]
            
            # Get next token
            next_token = np.argmax(logits[0, -1, :])
            
            if next_token == self.tokenizer.eos_token_id:
                break
                
            # Update sequence
            generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def _generate_multi_component(self, text: str, max_tokens: int, images: Optional[List[str]]) -> str:
        """Generate text using multi-component ONNX model (like Qwen2-VL)."""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='np')
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', np.ones_like(input_ids))
        
        # Get embeddings based on architecture
        if self.embed_model is not None:
            # Standard embedding model (Qwen2-VL, Gemma)
            embed_outputs = self.embed_model.run(None, {'input_ids': input_ids})
            inputs_embeds = embed_outputs[0]
        elif self.prepare_inputs_embeds_model is not None:
            # Phi-3.5 style prepare_inputs_embeds - requires image_features input
            # For text-only inference, provide empty image_features indicating no images
            dummy_image_features = np.zeros((0, 3072), dtype=np.float32)  # No images (empty sequence)
            embed_inputs = {
                'input_ids': input_ids,
                'image_features': dummy_image_features
            }
            embed_outputs = self.prepare_inputs_embeds_model.run(None, embed_inputs)
            inputs_embeds = embed_outputs[0]
        else:
            raise ValueError("No embedding model available for multi-component inference")
        
        # Initialize sequence tracking
        batch_size = 1
        seq_length = input_ids.shape[1]
        generated_ids = input_ids.copy()
        
        # Prepare initial decoder inputs
        onnx_inputs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask
        }
        
        # Add position_ids if needed
        if 'position_ids' in self.decoder_input_names:
            onnx_inputs['position_ids'] = create_position_ids(seq_length, self.config)
        
        # Initialize KV cache
        kv_cache = initialize_kv_cache(self.config, batch_size)
        for key, value in kv_cache.items():
            if key in self.decoder_input_names:
                onnx_inputs[key] = value
        
        # Generation loop
        for step in range(max_tokens):
            # Run decoder
            outputs = self.decoder_model.run(None, onnx_inputs)
            
            # Get next token
            logits = outputs[0]
            next_token = np.argmax(logits[0, -1, :])
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Update sequence
            generated_ids = np.concatenate([generated_ids, [[next_token]]], axis=1)
            
            # Get next token embeddings based on architecture
            if self.embed_model is not None:
                # Standard embedding model
                next_embed = self.embed_model.run(None, {'input_ids': np.array([[next_token]])})
            elif self.prepare_inputs_embeds_model is not None:
                # Phi-3.5 style prepare_inputs_embeds - requires image_features input  
                dummy_image_features = np.zeros((0, 3072), dtype=np.float32)  # No images (empty sequence)
                next_embed_inputs = {
                    'input_ids': np.array([[next_token]]),
                    'image_features': dummy_image_features
                }
                next_embed = self.prepare_inputs_embeds_model.run(None, next_embed_inputs)
            else:
                raise ValueError("No embedding model available for next token generation")
            
            # Update inputs for next iteration
            onnx_inputs['inputs_embeds'] = next_embed[0]
            onnx_inputs['attention_mask'] = np.ones((1, generated_ids.shape[1]), dtype=np.int64)
            
            # Update position_ids
            if 'position_ids' in self.decoder_input_names:
                onnx_inputs['position_ids'] = create_position_ids(seq_length, self.config, step + 1)
            
            # Update KV cache
            output_idx = 1  # Skip logits
            for i in range(self.config.num_layers):
                key_name = f'past_key_values.{i}.key'
                value_name = f'past_key_values.{i}.value'
                if key_name in self.decoder_input_names:
                    onnx_inputs[key_name] = outputs[output_idx]
                    onnx_inputs[value_name] = outputs[output_idx + 1]
                    output_idx += 2
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)