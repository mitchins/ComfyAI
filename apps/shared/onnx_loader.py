"""ONNX model loader with support for multi-component models."""

from typing import Dict, Optional, List, Any
import logging
import numpy as np
import os

# Check for explicit mock environment variable
USE_MOCK_ONNX = os.getenv("USE_MOCK_ONNX", "false").lower() in ("true", "1", "yes")

if USE_MOCK_ONNX:
    # Mock implementations for testing
    class MockSession:
        def run(self, *args, **kwargs):
            return [np.random.rand(1, 10, 32000)]
        def get_inputs(self):
            return [type('MockInput', (), {'name': 'input_ids'})()]
    
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return {'input_ids': np.array([[1, 2, 3]])}
        @property 
        def eos_token_id(self):
            return 2
        def decode(self, *args, **kwargs):
            return "mocked response"
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
    
    class MockConfig:
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
    
    def mock_hf_hub_download(*args, **kwargs):
        return "/mock/path/model.onnx"
    
    # Mock the imports
    ort = type('MockORT', (), {'InferenceSession': MockSession})
    AutoTokenizer = MockTokenizer
    hf_hub_download = mock_hf_hub_download
    ONNX_AVAILABLE = False
    print("Using mock ONNX dependencies for testing")
else:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    ONNX_AVAILABLE = True

    # Mock classes for testing environments
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
        
        # Load tokenizer and config for dynamic architecture parameters
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True, trust_remote_code=True)
        
        # Get actual architecture parameters from model config (like test_gemma3n.py)
        if USE_MOCK_ONNX:
            # Mock AutoConfig for testing
            hf_config = MockConfig()
        else:
            from transformers import AutoConfig
        
        try:
            if not USE_MOCK_ONNX:
                hf_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
            else:
                hf_config = MockConfig()
            # Update config with actual model parameters
            if hasattr(hf_config, 'text_config'):
                # Multi-modal models have text_config
                text_config = hf_config.text_config
                config.num_layers = getattr(text_config, 'num_hidden_layers', config.num_layers)
                config.num_heads = getattr(text_config, 'num_attention_heads', config.num_heads)
                config.num_kv_heads = getattr(text_config, 'num_key_value_heads', config.num_kv_heads)
                # Calculate head_dim from hidden_size / num_attention_heads if available
                hidden_size = getattr(text_config, 'hidden_size', None)
                if hidden_size and config.num_heads:
                    config.head_dim = hidden_size // config.num_heads
            else:
                # Single-modal models have config directly
                config.num_layers = getattr(hf_config, 'num_hidden_layers', config.num_layers)
                config.num_heads = getattr(hf_config, 'num_attention_heads', config.num_heads)
                config.num_kv_heads = getattr(hf_config, 'num_key_value_heads', config.num_kv_heads)
                # Calculate head_dim from hidden_size / num_attention_heads if available
                hidden_size = getattr(hf_config, 'hidden_size', None)
                if hidden_size and config.num_heads:
                    config.head_dim = hidden_size // config.num_heads
            
            logger.info(f"Updated config from model: layers={config.num_layers}, heads={config.num_heads}, kv_heads={config.num_kv_heads}, head_dim={config.head_dim}")
        except Exception as e:
            logger.warning(f"Could not load dynamic config, using static config: {e}")
        
        # Download components using curated paths
        component_paths = self.download_curated_components(repo_id, component_config)
        
        # Load ONNX sessions
        sessions = {}
        for component_name, path in component_paths.items():
            # Use CPU-only for tests to avoid CoreML dynamic shape issues
            # In production, you might want to use ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            providers = ['CPUExecutionProvider']
            sessions[component_name] = ort.InferenceSession(path, providers=providers)
            logger.info(f"Loaded {component_name} from {path}")
        
        # Cache everything
        self.sessions_cache[model_quant_name] = sessions
        self.tokenizers_cache[model_quant_name] = tokenizer
        self.configs_cache[model_quant_name] = config
        
        return sessions, tokenizer, config
    
    def _get_repo_id_from_model_name(self, model_name: str) -> Optional[str]:
        """Map model name to repo_id using curated model mappings."""
        model_repo_mapping = {
            "Gemma-3n-E2B-it-ONNX": "onnx-community/gemma-3n-E2B-it-ONNX", 
            "SmolVLM-256M-Instruct": "HuggingFaceTB/SmolVLM-256M-Instruct",
        }
        return model_repo_mapping.get(model_name)
    
    def download_curated_components(self, repo_id: str, component_config: Dict[str, str]) -> Dict[str, str]:
        """Download components using curated file paths."""
        component_paths = {}
        
        for component_name, filename in component_config.items():
            try:
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                
                # Also download companion .onnx_data files (numbered and non-numbered)
                if filename.endswith('.onnx'):
                    # First try non-numbered data file
                    data_filename = filename + '_data'
                    try:
                        data_path = hf_hub_download(repo_id=repo_id, filename=data_filename)
                        logger.info(f"Downloaded companion data file: {data_filename} -> {data_path}")
                    except Exception:
                        logger.debug(f"No base data file found for {filename}")
                    
                    # Then try numbered data files (e.g., _data_0, _data_1, etc.)
                    # Some models start at _data_1 instead of _data_0
                    for data_file_index in range(10):  # Try indices 0-9
                        data_filename = f"{filename}_data_{data_file_index}"
                        try:
                            data_path = hf_hub_download(repo_id=repo_id, filename=data_filename)
                            logger.info(f"Downloaded companion data file: {data_filename} -> {data_path}")
                        except Exception:
                            # This numbered data file doesn't exist, continue to next
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
            # Use CPU-only for tests to avoid CoreML dynamic shape issues
            providers = ['CPUExecutionProvider']
            sessions[component_name] = ort.InferenceSession(path, providers=providers)
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
        self.current_vision_features = None  # Initialize vision features storage
        
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
    
    def generate_text(self, text: str, max_tokens: int = 100, images: Optional[List[str]] = None, audio: Optional[List[str]] = None) -> str:
        """Generate text using the ONNX model."""
        if images and not self.config.has_vision:
            raise ValueError("Images provided but model doesn't support vision")
        if audio and not self.audio_model:
            raise ValueError("Audio provided but model doesn't support audio")
        
        # Handle single model vs multi-component models
        if self.model is not None:
            return self._generate_single_model(text, max_tokens, images)
        else:
            return self._generate_multi_component(text, max_tokens, images, audio)
    
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
    
    def _generate_multi_component(self, text: str, max_tokens: int, images: Optional[List[str]] = None, audio: Optional[List[str]] = None) -> str:
        """Generate text using multi-component ONNX model (like Qwen2-VL, Gemma3n, SmolVLM)."""
        # Special handling for different model architectures
        if "gemma" in self.tokenizer.name_or_path.lower() and (images or audio):
            return self._generate_gemma3n_multimodal(text, max_tokens, images, audio)
        elif "smolvlm" in self.tokenizer.name_or_path.lower() and images:
            return self._generate_smolvlm_multimodal(text, max_tokens, images)
        
        # Standard tokenization for other models
        inputs = self.tokenizer(text, return_tensors='np')
        input_ids = inputs['input_ids']
        attention_mask_2d = inputs.get('attention_mask', np.ones_like(input_ids))

        # Initialize per_layer_inputs to None by default
        per_layer_inputs = None

        # Get embeddings based on architecture
        if self.embed_model is not None:
            # Standard embedding model (Qwen2-VL, non-Gemma)
            embed_outputs = self.embed_model.run(None, {'input_ids': input_ids})
            inputs_embeds = embed_outputs[0]

            # Gemma3n models return per_layer_inputs as second output
            per_layer_inputs = embed_outputs[1] if len(embed_outputs) > 1 else None
        elif self.prepare_inputs_embeds_model is not None:
            # Phi-3.5 style prepare_inputs_embeds - requires image_features input
            # Use real vision features if available, otherwise empty
            if hasattr(self, 'current_vision_features') and self.current_vision_features is not None:
                image_features = self.current_vision_features
                logger.info(f"Using real vision features with shape: {image_features.shape}")
            else:
                image_features = np.zeros((0, 3072), dtype=np.float32)  # No images (empty sequence)
            
            embed_inputs = {
                'input_ids': input_ids,
                'image_features': image_features
            }
            embed_outputs = self.prepare_inputs_embeds_model.run(None, embed_inputs)
            inputs_embeds = embed_outputs[0]
            # per_layer_inputs remains None for Phi-3.5
        else:
            raise ValueError("No embedding model available for multi-component inference")

        # Initialize sequence tracking
        batch_size = 1
        seq_length = input_ids.shape[1]
        generated_ids = input_ids.copy()

        # Prepare initial decoder inputs
        onnx_inputs = {
            'inputs_embeds': inputs_embeds
        }

        # Add 2D attention_mask only for non-Gemma models
        if 'attention_mask' in self.decoder_input_names and per_layer_inputs is None:
            onnx_inputs['attention_mask'] = attention_mask_2d

        # Add per_layer_inputs if available (Gemma3n models)
        if per_layer_inputs is not None:
            onnx_inputs['per_layer_inputs'] = per_layer_inputs

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

            # Get embeddings based on architecture
            if self.embed_model is not None:
                # For Gemma3n (has per_layer_inputs), recompute on full sequence
                if per_layer_inputs is not None:
                    embed_outputs = self.embed_model.run(None, {'input_ids': generated_ids})
                    onnx_inputs['inputs_embeds'] = embed_outputs[0]
                    onnx_inputs['per_layer_inputs'] = embed_outputs[1]
                else:
                    # Non-Gemma incremental embed
                    next_embed_outputs = self.embed_model.run(None, {'input_ids': np.array([[next_token]])})
                    onnx_inputs['inputs_embeds'] = next_embed_outputs[0]
            elif self.prepare_inputs_embeds_model is not None:
                # Phi-3.5 style prepare_inputs_embeds
                # For subsequent tokens, we don't need to pass image features again
                # The model has already encoded them in the initial prompt
                image_features = np.zeros((0, 3072), dtype=np.float32)
                next_embed_inputs = {
                    'input_ids': np.array([[next_token]]),
                    'image_features': image_features
                }
                next_embed_outputs = self.prepare_inputs_embeds_model.run(None, next_embed_inputs)
                onnx_inputs['inputs_embeds'] = next_embed_outputs[0]
            else:
                raise ValueError("No embedding model available for next token generation")

            # Update 2D attention_mask only for non-Gemma models
            if 'attention_mask' in self.decoder_input_names and per_layer_inputs is None:
                onnx_inputs['attention_mask'] = np.ones_like(generated_ids)

            # Update position_ids
            if 'position_ids' in self.decoder_input_names:
                # Position IDs should be for the *next* token, which is at the end of the current sequence
                onnx_inputs['position_ids'] = create_position_ids(generated_ids.shape[1], self.config, generated_ids.shape[1] - 1)

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
    
    def _generate_gemma3n_multimodal(self, text: str, max_tokens: int, images: Optional[List[str]] = None, audio: Optional[List[str]] = None) -> str:
        """Generate text using Gemma3n with AutoProcessor like test_gemma3n.py."""
        try:
            import tempfile
            import os
            from PIL import Image
            import io
            import base64
            
            # Use AutoProcessor like test_gemma3n.py for proper multimodal processing
            if USE_MOCK_ONNX:
                processor = None
            else:
                from transformers import AutoProcessor
                
                # Get model_id from tokenizer name_or_path
                model_id = getattr(self.tokenizer, 'name_or_path', 'google/gemma-3n-E2B-it')
                
                try:
                    processor = AutoProcessor.from_pretrained(model_id)
                    # Try patching fast image processor to allow fallback if necessary
                    if hasattr(processor, "image_processor") and getattr(processor.image_processor, "is_fast", False):
                        processor.image_processor = processor.image_processor.__class__.from_pretrained(model_id, use_fast=False)
                except ValueError:
                    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
            
            if processor is None:
                logger.warning("AutoProcessor unavailable for Gemma3n, falling back to standard generation")
                return self._generate_multi_component_fallback(text, max_tokens, images, audio)
            
            # Build message structure like test_gemma3n.py
            content_parts = [{"type": "text", "text": text}]
            
            # Process images
            temp_files = []
            if images:
                for img_base64 in images:
                    # Decode base64 image and save to temp file for processor
                    img_bytes = base64.b64decode(img_base64)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save to temp file since processor expects file path or URL
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        img.save(tmp_file.name, 'JPEG')
                        temp_image_path = tmp_file.name
                        temp_files.append(temp_image_path)
                        content_parts.append({"type": "image", "image": temp_image_path})
            
            # Process audio (placeholder - would need actual audio handling)
            if audio:
                logger.warning("Audio processing not yet implemented for Gemma3n")
            
            try:
                # Create messages like test_gemma3n.py
                messages = [
                    {
                        "role": "user",
                        "content": content_parts,
                    },
                ]
                
                # Process with AutoProcessor like test_gemma3n.py
                try:
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="np",
                    )
                except ValueError:
                    # Fallback to PyTorch tensors if numpy not supported
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                
                # Convert to numpy if needed
                def to_numpy(x):
                    return x.numpy() if hasattr(x, "numpy") else x
                
                input_ids = to_numpy(inputs["input_ids"])
                attention_mask = to_numpy(inputs.get("attention_mask", np.ones_like(input_ids)))
                pixel_values = to_numpy(inputs.get("pixel_values")) if inputs.get("pixel_values") is not None else None
                input_features = to_numpy(inputs.get("input_features")).astype(np.float32) if inputs.get("input_features") is not None else None
                input_features_mask = to_numpy(inputs.get("input_features_mask")) if inputs.get("input_features_mask") is not None else None
                
                logger.info(f"Gemma3n AutoProcessor: input_ids shape: {input_ids.shape}")
                if pixel_values is not None:
                    logger.info(f"Gemma3n AutoProcessor: pixel_values shape: {pixel_values.shape}")
                
                # Follow the exact generation loop from test_gemma3n.py
                batch_size = input_ids.shape[0]
                position_ids = np.cumsum(attention_mask, axis=-1) - 1
                
                # Get config values like test_gemma3n.py
                image_token_id = getattr(self.tokenizer, 'image_token_id', 256012)
                audio_token_id = getattr(self.tokenizer, 'audio_token_id', 256013)
                
                # Initialize KV cache
                past_key_values = {
                    f"past_key_values.{layer}.{kv}": np.zeros([batch_size, self.config.num_kv_heads, 0, self.config.head_dim], dtype=np.float32)
                    for layer in range(self.config.num_layers)
                    for kv in ("key", "value")
                }
                
                # Generation loop like test_gemma3n.py
                generated_tokens = np.array([[]], dtype=np.int64)
                image_features = None
                audio_features = None
                
                for i in range(max_tokens):
                    # Get embeddings like test_gemma3n.py
                    inputs_embeds, per_layer_inputs = self.embed_model.run(None, {"input_ids": input_ids})
                    
                    # Process vision features once like test_gemma3n.py
                    if image_features is None and pixel_values is not None:
                        image_features = self.vision_model.run(["image_features"], {"pixel_values": pixel_values})[0]
                        mask = (input_ids == image_token_id).reshape(-1)
                        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                        flat_embeds[mask] = image_features.reshape(-1, image_features.shape[-1])
                        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                        logger.info(f"Injected vision features at {np.sum(mask)} positions")
                    
                    # Process audio features once like test_gemma3n.py
                    if audio_features is None and input_features is not None and input_features_mask is not None:
                        audio_features = self.audio_model.run(["audio_features"], {
                            "input_features": input_features,
                            "input_features_mask": input_features_mask,
                        })[0]
                        mask = (input_ids == audio_token_id).reshape(-1)
                        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                        flat_embeds[mask] = audio_features.reshape(-1, audio_features.shape[-1])
                        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                        logger.info(f"Injected audio features at {np.sum(mask)} positions")
                    
                    # Run decoder like test_gemma3n.py
                    decoder_inputs = dict(
                        inputs_embeds=inputs_embeds,
                        per_layer_inputs=per_layer_inputs,
                        position_ids=position_ids,
                        **past_key_values,
                    )
                    
                    outputs = self.decoder_model.run(None, decoder_inputs)
                    
                    # Get next token
                    logits = outputs[0]
                    next_token = np.argmax(logits[0, -1, :])
                    
                    if next_token == self.tokenizer.eos_token_id:
                        break
                    
                    # Update sequence
                    generated_tokens = np.concatenate([generated_tokens, [[next_token]]], axis=1)
                    input_ids = np.array([[next_token]])
                    attention_mask = np.ones_like(input_ids)
                    position_ids = position_ids[:, -1:] + 1
                    
                    # Update KV cache
                    output_idx = 1  # Skip logits
                    for layer in range(self.config.num_layers):
                        key_name = f'past_key_values.{layer}.key'
                        value_name = f'past_key_values.{layer}.value'
                        past_key_values[key_name] = outputs[output_idx]
                        past_key_values[value_name] = outputs[output_idx + 1]
                        output_idx += 2
                
                # Decode the result
                result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                logger.info(f"Gemma3n generated {generated_tokens.shape[-1]} tokens")
                return result
                
            finally:
                # Clean up temp files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
        
        except Exception as e:
            logger.error(f"Gemma3n multimodal generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to standard generation
            return self._generate_multi_component_fallback(text, max_tokens, images, audio)
    
    def _generate_smolvlm_multimodal(self, text: str, max_tokens: int, images: Optional[List[str]] = None) -> str:
        """Generate text using SmolVLM with official transformers + ONNXRuntime approach."""
        try:
            import tempfile
            import os
            from PIL import Image
            import io
            import base64
            
            # Use transformers processor like official SmolVLM example
            if USE_MOCK_ONNX:
                processor = None
            else:
                from transformers import AutoProcessor
                from transformers.image_utils import load_image
                
                # Get model_id from tokenizer name_or_path
                model_id = getattr(self.tokenizer, 'name_or_path', 'HuggingFaceTB/SmolVLM-256M-Instruct')
                
                try:
                    processor = AutoProcessor.from_pretrained(model_id)
                except Exception as e:
                    logger.error(f"Failed to load SmolVLM processor: {e}")
                    processor = None
            
            if processor is None:
                logger.warning("AutoProcessor unavailable for SmolVLM, falling back to standard generation")
                return self._generate_multi_component_fallback(text, max_tokens, images, None)
            
            # Process images like official example
            processed_images = []
            temp_files = []
            
            if images:
                for img_base64 in images:
                    # Decode base64 image and save to temp file
                    img_bytes = base64.b64decode(img_base64)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save to temp file since transformers load_image expects path or URL
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        img.save(tmp_file.name, 'JPEG')
                        temp_image_path = tmp_file.name
                        temp_files.append(temp_image_path)
                        
                        # Load image using transformers utility
                        processed_images.append(load_image(temp_image_path))
            
            try:
                # Create messages like official SmolVLM example
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},  # SmolVLM uses simple image placeholder
                            {"type": "text", "text": text}
                        ]
                    },
                ]
                
                # Apply chat template like official example
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                
                # Process with images like official example
                inputs = processor(text=prompt, images=processed_images, return_tensors="np")
                
                # Convert to numpy if needed
                def to_numpy(x):
                    return x.numpy() if hasattr(x, "numpy") else x
                
                input_ids = to_numpy(inputs["input_ids"])
                attention_mask = to_numpy(inputs["attention_mask"])
                pixel_values = to_numpy(inputs["pixel_values"]) if "pixel_values" in inputs else None
                pixel_attention_mask = to_numpy(inputs["pixel_attention_mask"]) if "pixel_attention_mask" in inputs else None
                
                logger.info(f"SmolVLM: input_ids shape: {input_ids.shape}")
                if pixel_values is not None:
                    logger.info(f"SmolVLM: pixel_values shape: {pixel_values.shape}")
                if pixel_attention_mask is not None:
                    logger.info(f"SmolVLM: pixel_attention_mask shape: {pixel_attention_mask.shape}")
                
                # Get config values like official example
                image_token_id = getattr(self.tokenizer, 'image_token_id', None)
                if hasattr(self.tokenizer, 'special_tokens_map') and self.tokenizer.special_tokens_map:
                    # Try to find image token in special tokens
                    for token_name, token_value in self.tokenizer.special_tokens_map.items():
                        if 'image' in token_name.lower():
                            image_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                            break
                
                if image_token_id is None:
                    # SmolVLM typically uses a specific image token ID, let's check the config
                    try:
                        from transformers import AutoConfig
                        config = AutoConfig.from_pretrained(model_id)
                        image_token_id = getattr(config, 'image_token_id', None)
                        logger.info(f"SmolVLM image_token_id from config: {image_token_id}")
                    except:
                        # Fallback - SmolVLM commonly uses specific token IDs
                        image_token_id = 151646  # Common SmolVLM image token
                        logger.warning(f"Using fallback SmolVLM image_token_id: {image_token_id}")
                
                # Initialize KV cache like official example
                batch_size = input_ids.shape[0]
                past_key_values = {
                    f"past_key_values.{layer}.{kv}": np.zeros([batch_size, self.config.num_kv_heads, 0, self.config.head_dim], dtype=np.float32)
                    for layer in range(self.config.num_layers)
                    for kv in ("key", "value")
                }
                
                # Generation loop like official SmolVLM example
                generated_tokens = np.array([[]], dtype=np.int64)
                image_features = None
                position_ids = np.cumsum(attention_mask, axis=-1)
                
                for i in range(max_tokens):
                    # Get embeddings
                    inputs_embeds = self.embed_model.run(None, {"input_ids": input_ids})[0]
                    
                    # Process vision features once like official example
                    if image_features is None and pixel_values is not None:
                        vision_inputs = {"pixel_values": pixel_values}
                        
                        # Add pixel_attention_mask if available (as boolean like official example)
                        if pixel_attention_mask is not None:
                            vision_inputs["pixel_attention_mask"] = pixel_attention_mask.astype(np.bool_)
                        
                        image_features = self.vision_model.run(["image_features"], vision_inputs)[0]
                        
                        # Merge text and vision embeddings like official example
                        if image_token_id is not None:
                            # Only inject on first sequence, not subsequent single tokens
                            if input_ids.shape[1] > 1:  # Initial long sequence
                                mask = (input_ids == image_token_id)
                                vision_features_flat = image_features.reshape(-1, image_features.shape[-1])
                                num_image_tokens = np.sum(mask)
                                
                                if vision_features_flat.shape[0] == num_image_tokens:
                                    inputs_embeds[mask] = vision_features_flat
                                    logger.info(f"SmolVLM: Injected {num_image_tokens} vision features on initial sequence")
                                else:
                                    logger.warning(f"SmolVLM: Vision feature mismatch - {num_image_tokens} tokens vs {vision_features_flat.shape[0]} features")
                            else:
                                logger.debug("SmolVLM: Skipping vision injection for single token")
                        else:
                            logger.warning("SmolVLM: No image_token_id found, skipping vision feature injection")
                    
                    # Run decoder like official example
                    decoder_inputs = dict(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **past_key_values,
                    )
                    
                    outputs = self.decoder_model.run(None, decoder_inputs)
                    
                    # Get next token
                    logits = outputs[0]
                    next_token = np.argmax(logits[:, -1], axis=-1, keepdims=True)
                    
                    if (next_token == self.tokenizer.eos_token_id).all():
                        break
                    
                    # Update sequence like official example
                    generated_tokens = np.concatenate([generated_tokens, next_token], axis=-1)
                    input_ids = next_token
                    attention_mask = np.ones_like(input_ids)
                    position_ids = position_ids[:, -1:] + 1
                    
                    # Update KV cache
                    output_idx = 1  # Skip logits
                    for layer in range(self.config.num_layers):
                        key_name = f'past_key_values.{layer}.key'
                        value_name = f'past_key_values.{layer}.value'
                        past_key_values[key_name] = outputs[output_idx]
                        past_key_values[value_name] = outputs[output_idx + 1]
                        output_idx += 2
                
                # Decode the result like official example
                result = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                logger.info(f"SmolVLM generated {generated_tokens.shape[-1]} tokens")
                return result
                
            finally:
                # Clean up temp files
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
        
        except Exception as e:
            logger.error(f"SmolVLM multimodal generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to standard generation
            return self._generate_multi_component_fallback(text, max_tokens, images, None)
    
    def _generate_multi_component_fallback(self, text: str, max_tokens: int, images: Optional[List[str]] = None, audio: Optional[List[str]] = None) -> str:
        """Fallback to original multi-component generation."""
        # Standard tokenization for other models
        inputs = self.tokenizer(text, return_tensors='np')
        input_ids = inputs['input_ids']
        attention_mask_2d = inputs.get('attention_mask', np.ones_like(input_ids))

        # Initialize per_layer_inputs to None by default
        per_layer_inputs = None

        # Get embeddings based on architecture
        if self.embed_model is not None:
            # Standard embedding model (Qwen2-VL, non-Gemma)
            embed_outputs = self.embed_model.run(None, {'input_ids': input_ids})
            inputs_embeds = embed_outputs[0]

            # Gemma3n models return per_layer_inputs as second output
            per_layer_inputs = embed_outputs[1] if len(embed_outputs) > 1 else None

            # Handle Gemma3n multimodal feature injection (for fallback)
            if self.vision_model and self.audio_model and ("gemma" in self.tokenizer.name_or_path.lower()):
                inputs_embeds = self._inject_gemma3n_features(inputs_embeds, input_ids, images, audio)
        elif self.prepare_inputs_embeds_model is not None:
            # Phi-3.5 style prepare_inputs_embeds - requires image_features input
            # Use real vision features if available, otherwise empty
            if hasattr(self, 'current_vision_features') and self.current_vision_features is not None:
                image_features = self.current_vision_features
                logger.info(f"Using real vision features with shape: {image_features.shape}")
            else:
                image_features = np.zeros((0, 3072), dtype=np.float32)  # No images (empty sequence)
            
            embed_inputs = {
                'input_ids': input_ids,
                'image_features': image_features
            }
            embed_outputs = self.prepare_inputs_embeds_model.run(None, embed_inputs)
            inputs_embeds = embed_outputs[0]
            # per_layer_inputs remains None for Phi-3.5
        else:
            raise ValueError("No embedding model available for multi-component inference")

        # Initialize sequence tracking
        batch_size = 1
        seq_length = input_ids.shape[1]
        generated_ids = input_ids.copy()

        # Prepare initial decoder inputs
        onnx_inputs = {
            'inputs_embeds': inputs_embeds
        }

        # Add 2D attention_mask only for non-Gemma models
        if 'attention_mask' in self.decoder_input_names and per_layer_inputs is None:
            onnx_inputs['attention_mask'] = attention_mask_2d

        # Add per_layer_inputs if available (Gemma3n models)
        if per_layer_inputs is not None:
            onnx_inputs['per_layer_inputs'] = per_layer_inputs

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

            # Get embeddings based on architecture
            if self.embed_model is not None:
                # For Gemma3n (has per_layer_inputs), recompute on full sequence
                if per_layer_inputs is not None:
                    embed_outputs = self.embed_model.run(None, {'input_ids': generated_ids})
                    onnx_inputs['inputs_embeds'] = embed_outputs[0]
                    onnx_inputs['per_layer_inputs'] = embed_outputs[1]
                else:
                    # Non-Gemma incremental embed
                    next_embed_outputs = self.embed_model.run(None, {'input_ids': np.array([[next_token]])})
                    onnx_inputs['inputs_embeds'] = next_embed_outputs[0]
            elif self.prepare_inputs_embeds_model is not None:
                # Phi-3.5 style prepare_inputs_embeds
                # For subsequent tokens, we don't need to pass image features again
                # The model has already encoded them in the initial prompt
                image_features = np.zeros((0, 3072), dtype=np.float32)
                next_embed_inputs = {
                    'input_ids': np.array([[next_token]]),
                    'image_features': image_features
                }
                next_embed_outputs = self.prepare_inputs_embeds_model.run(None, next_embed_inputs)
                onnx_inputs['inputs_embeds'] = next_embed_outputs[0]
            else:
                raise ValueError("No embedding model available for next token generation")

            # Update 2D attention_mask only for non-Gemma models
            if 'attention_mask' in self.decoder_input_names and per_layer_inputs is None:
                onnx_inputs['attention_mask'] = np.ones_like(generated_ids)

            # Update position_ids
            if 'position_ids' in self.decoder_input_names:
                # Position IDs should be for the *next* token, which is at the end of the current sequence
                onnx_inputs['position_ids'] = create_position_ids(generated_ids.shape[1], self.config, generated_ids.shape[1] - 1)

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
    
    def _inject_gemma3n_features(self, inputs_embeds: np.ndarray, input_ids: np.ndarray, images: Optional[List[str]], audio: Optional[List[str]]) -> np.ndarray:
        """Inject image and audio features into embeddings for Gemma3n models."""
        # Get special token IDs from tokenizer config
        try:
            image_token_id = getattr(self.tokenizer, 'image_token_id', None) 
            audio_token_id = getattr(self.tokenizer, 'audio_token_id', None)
        except:
            # Fallback values from test_gemma3n.py
            image_token_id = 256012  # Default from Gemma3n config
            audio_token_id = 256013  # Default from Gemma3n config
        
        # Process vision features if images provided (Gemma-3n specific)
        if images and self.vision_model and image_token_id:
            try:
                from PIL import Image
                import io
                import base64
                import tempfile
                import os
                
                # Use AutoProcessor like test_gemma3n.py for proper image preprocessing
                if USE_MOCK_ONNX:
                    from transformers import AutoConfig
                    processor = None
                else:
                    from transformers import AutoProcessor, AutoConfig
                    
                    # Get model_id from tokenizer name_or_path
                    model_id = getattr(self.tokenizer, 'name_or_path', 'google/gemma-3n-E2B-it')
                    
                    try:
                        processor = AutoProcessor.from_pretrained(model_id)
                        # Try patching fast image processor to allow fallback if necessary
                        if hasattr(processor, "image_processor") and getattr(processor.image_processor, "is_fast", False):
                            processor.image_processor = processor.image_processor.__class__.from_pretrained(model_id, use_fast=False)
                    except ValueError:
                        processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
                
                if processor is None:
                    # Fallback to manual preprocessing if processor unavailable
                    logger.warning("AutoProcessor unavailable, falling back to manual preprocessing")
                    vision_features_list = []
                    for img_base64 in images:
                        # Decode base64 image
                        img_bytes = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # For Gemma-3n: Resize to 768x768 (as per working example)
                        img = img.resize((768, 768))
                        
                        # Convert to numpy array and preprocess (matching working example)
                        img_array = np.array(img).astype(np.float32)
                        img_array = img_array / 255.0  # Normalize to [0, 1]
                        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
                        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                        
                        logger.info(f"Manual Gemma-3n image preprocessing: {img.size} -> {img_array.shape}")
                        
                        # Run through vision encoder
                        vision_outputs = self.vision_model.run(None, {'pixel_values': img_array})
                        vision_features = vision_outputs[0]
                        vision_features_list.append(vision_features)
                        
                        logger.info(f"Manual Gemma-3n vision features shape: {vision_features.shape}")
                else:
                    # Use AutoProcessor like test_gemma3n.py
                    vision_features_list = []
                    for img_base64 in images:
                        # Decode base64 image and save to temp file for processor
                        img_bytes = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to temp file since processor expects file path or URL
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            img.save(tmp_file.name, 'JPEG')
                            temp_image_path = tmp_file.name
                        
                        try:
                            # Create messages like test_gemma3n.py
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": temp_image_path},
                                    ],
                                },
                            ]
                            
                            # Process with AutoProcessor like test_gemma3n.py
                            try:
                                inputs = processor.apply_chat_template(
                                    messages,
                                    add_generation_prompt=False,  # Just preprocessing, no generation prompt
                                    tokenize=False,  # Don't tokenize, just get pixel_values
                                    return_dict=True,
                                    return_tensors="np",
                                )
                            except ValueError:
                                # Fallback to PyTorch tensors if numpy not supported
                                inputs = processor.apply_chat_template(
                                    messages,
                                    add_generation_prompt=False,
                                    tokenize=False,
                                    return_dict=True,
                                    return_tensors="pt",
                                )
                            
                            # Extract pixel_values and convert to numpy if needed
                            pixel_values = inputs.get("pixel_values")
                            if pixel_values is not None:
                                if hasattr(pixel_values, 'numpy'):
                                    pixel_values = pixel_values.numpy()
                                pixel_values = pixel_values.astype(np.float32)
                                
                                logger.info(f"AutoProcessor Gemma-3n pixel_values shape: {pixel_values.shape}")
                                
                                # Run through vision encoder with processor-generated pixel_values
                                vision_outputs = self.vision_model.run(None, {'pixel_values': pixel_values})
                                vision_features = vision_outputs[0]
                                vision_features_list.append(vision_features)
                                
                                logger.info(f"AutoProcessor Gemma-3n vision features shape: {vision_features.shape}")
                            else:
                                logger.warning("No pixel_values returned from AutoProcessor")
                                
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_image_path):
                                os.unlink(temp_image_path)
                
                # Store vision features for injection into embeddings
                if vision_features_list:
                    self.current_vision_features = vision_features_list[0]  # Use first image
                    logger.info(f"Stored Gemma-3n vision features: {self.current_vision_features.shape}")
                    
                    # Find positions of image tokens in the input
                    image_token_positions = np.where(input_ids[0] == image_token_id)[0]
                    
                    if len(image_token_positions) > 0:
                        logger.info(f"Found {len(image_token_positions)} image token positions in Gemma-3n")
                        
                        # Replace image tokens with vision features (like working example)
                        mask = (input_ids == image_token_id).reshape(-1)
                        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                        
                        # Reshape vision features to match embedding dimension
                        vision_flat = self.current_vision_features.reshape(-1, self.current_vision_features.shape[-1])
                        
                        # Replace image token embeddings with vision features
                        if np.sum(mask) == vision_flat.shape[0]:
                            flat_embeds[mask] = vision_flat
                            inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)
                            logger.info(f"Successfully replaced {np.sum(mask)} image tokens with vision features")
                        else:
                            logger.warning(f"Token count mismatch: {np.sum(mask)} tokens vs {vision_flat.shape[0]} features")
                    else:
                        logger.warning(f"No image tokens found in Gemma-3n input! Image token ID: {image_token_id}")
                else:
                    self.current_vision_features = None
                    
            except Exception as e:
                logger.error(f"Failed to process images: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Continue without vision features rather than failing completely
        
        # Process audio features if audio provided  
        if audio and self.audio_model and audio_token_id:
            # For now, skip actual audio processing - would need processor integration
            # This is a placeholder for when we have proper multimodal input processing
            pass
            
        return inputs_embeds