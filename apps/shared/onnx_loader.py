"""ONNX model loader with support for multi-component models."""

from typing import Dict, Optional, List, Any
import logging
import numpy as np

# Optional imports for testing environments
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    ONNX_AVAILABLE = True
except ImportError as e:
    # Mock classes for testing environments
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
    
    def mock_hf_hub_download(*args, **kwargs):
        return "/mock/path/model.onnx"
    
    ort = type('MockORT', (), {'InferenceSession': MockSession})()
    AutoTokenizer = MockTokenizer
    hf_hub_download = mock_hf_hub_download
    ONNX_AVAILABLE = False
    print(f"Warning: ONNX dependencies not available: {e}")

from .model_types import ONNXModelConfig, get_onnx_model_config, create_position_ids, initialize_kv_cache


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
        fallback_suffixes = ["", "_fp16", "_int8", "_quantized", "_uint8"]
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
    
    def load_model(self, model_name: str) -> tuple[Dict[str, Any], AutoTokenizer, ONNXModelConfig]:
        """Load ONNX model components, tokenizer, and config."""
        if model_name in self.sessions_cache:
            return (
                self.sessions_cache[model_name],
                self.tokenizers_cache[model_name], 
                self.configs_cache[model_name]
            )
        
        repo_id, filename = self.parse_model_name(model_name)
        config = get_onnx_model_config(repo_id)
        
        if config is None:
            raise ValueError(f"Unsupported model: {repo_id}. Add configuration to ONNX_MODEL_CONFIGS.")
        
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
            if ONNX_AVAILABLE:
                sessions[component_name] = ort.InferenceSession(path)
            else:
                sessions[component_name] = ort.InferenceSession()  # Mock session
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
            self.model_input_names = [inp.name for inp in self.model.get_inputs()]
        else:
            # Multi-component model (like Qwen2-VL)
            self.embed_model = sessions['embed']
            self.decoder_model = sessions['decoder']
            self.model = None
            self.decoder_input_names = [inp.name for inp in self.decoder_model.get_inputs()]
            
        self.vision_model = sessions.get('vision')  # Optional
        self.audio_model = sessions.get('audio')   # Optional
    
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
        
        # Get embeddings
        embed_outputs = self.embed_model.run(None, {'input_ids': input_ids})
        inputs_embeds = embed_outputs[0]
        
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
            
            # Get next token embeddings
            next_embed = self.embed_model.run(None, {'input_ids': np.array([[next_token]])})
            
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