import logging
import sys
import io
import os
import time
import importlib
from huggingface_hub import scan_cache_dir
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

WARNING_PREFIX = "⚠ "  # Unicode warning icon

# Dictionary to store model issues, indexed by model name
model_issues = {}
model_configs = {}

# Lazily cached model configs
def get_model_config(model_name):
    try:
        # Load only the model's config
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        model_configs[model_name] = config
    except Exception as e:
        logging.warning(f"⚠️ Could not load config for {model_name}: {e}")
    return AutoConfig.from_pretrained(model_name)

# strip the warning prefix off models names if needed
def strip_warning_prefix(model_name):
    """Strip the warning prefix off model names."""
    return model_name[len(WARNING_PREFIX):] if has_model_issues(model_name) else model_name

def has_model_issues(model_name):
    """Returns True if the model name has been flagged with a warning (⚠ prefix)."""
    return model_name.startswith(WARNING_PREFIX)

def is_module_installed(module_name):
    """Check if a Python module is installed without importing it."""
    return importlib.util.find_spec(module_name) is not None

def is_gptq_model(model_name):
    """Check if the model name suggests it is GPTQ quantized."""
    return "GPTQ" in model_name or "-gptq" in model_name.lower()

def is_pixtral_model(config):
    """Detect Pixtral models based on vision model type."""
    try:
        return config.vision_config.model_type in {"pixtral", "mistral"}
    except AttributeError:
        return False  # If `vision_config` is missing, assume it's not Pixtral
    
def is_mllama_model(config):
    return config.model_type == "mllama"

def check_gptq_dependency():
    """Check if optimum is installed when loading GPTQ models."""
    return is_module_installed("optimum")

def is_flash_attention_installed():
    """Check if Flash Attention 2.0 is installed."""
    return is_module_installed("flash_attn")

def requires_flash_attention(config):
    """Check if a model explicitly requires FA2."""
    return config.model_type == "qwen2_vl"  # Qwen models don't work properly without FA2

def supports_flash_attention(config):
    """
    Check if a model supports Flash Attention 2.
    - If the config explicitly states it, return True/False.
    - If unknown, assume False to avoid errors.
    """
    return getattr(config, "supports_flash_attention_2", False)

def get_attention_implementation(model_name):
    """
    Decide which attention implementation to use:
    - If FA2 is REQUIRED, enable it (but warn if it's missing).
    - If FA2 is available, let it be used.
    - Otherwise, OMIT `attn_implementation` and let the model decide.
    """
    fa_installed = is_flash_attention_installed()

    config = get_model_config(model_name)

    if requires_flash_attention(config):
        if not fa_installed:
            logging.warning(f"⚠️ {model_name} REQUIRES Flash Attention 2.0, but it is NOT installed! Install it with `pip install flash-attn`.")
            return None  # Omit `attn_implementation`
        logging.info(f"⚡ {model_name} requires Flash Attention 2.0. Enabling FA2.")
        return "flash_attention_2"

    if supports_flash_attention(config):
        if not fa_installed:
            logging.warning(f"⚠️ {model_name} supports Flash Attention 2.0, but FA2 is not installed. Omitting `attn_implementation` to let the model decide.")
            return None  # Omit `attn_implementation`
        logging.info(f"✅ {model_name} supports Flash Attention 2.0. Enabling FA2.")
        return "flash_attention_2"

    logging.info(f"🚨 {model_name} does NOT support Flash Attention 2.0. Omitting `attn_implementation`.")
    return None  # Omit `attn_implementation`

def check_model_issues(model_name):
    """Check for missing dependencies or known issues with the model."""
    issues = []

    # Load model config once and pass it down
    try:
        config = get_model_config(model_name)
    except Exception as e:
        logging.error(f"⚠️ Failed to load config for {model_name}: {e}")
        issues.append("Could not retrieve model config.")
        return issues  # Prevent further checks

    # Detect unsupported Pixtral models based on model type
    # TODO: https://mistral.ai/news/pixtral-12b?utm_source=tldrai
    if is_pixtral_model(config):
        issues.append("⚠️ Pixtral architecture is not supported yet. Model will not load.")
    elif is_mllama_model(config):
        issues.append("⚠️ MLLama models are not supported yet. Model will not load.")

    # GPTQ Model Check (name-based detection)
    if is_gptq_model(model_name) and not check_gptq_dependency():
        issues.append("Requires `optimum` (pip install optimum)")

    return issues

def get_model_issues(model_name):
    """Retrieve indexed issues for a given model."""
    return model_issues.get(model_name, [])

def list_cached_vision_models():
    """Finds all locally cached models that are compatible with AutoModelForVision2Seq."""
    cache_info = scan_cache_dir()
    valid_models = []

    for repo in cache_info.repos:
        model_name = repo.repo_id  # Get the HF model name

        for revision in repo.revisions:
            # Ensure config.json exists before attempting to load
            if not any(file.file_name == "config.json" for file in revision.files):
                continue

            try:
                # Load only the model's config
                config = get_model_config(model_name)

                # Ensure model config has necessary attributes
                if not hasattr(config, "model_type") or not hasattr(config, "architectures"):
                    continue

                model_type = config.model_type
                model_arch = config.architectures[0]

                # Check if it's a known vision-capable model
                if model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES:
                    logging.debug(f"✅ Found compatible model: {model_name} | Type: {model_type} | Arch: {model_arch}")
                    issues = check_model_issues(model_name)
                    if issues:
                        warning_text = f"{WARNING_PREFIX}{model_name}"
                        valid_models.append(warning_text)
                        model_issues[warning_text] = issues
                        logging.warning(f"{model_name} has issues: {', '.join(issues)}")
                    else:
                        valid_models.append(model_name)
                    break  # Stop checking once verified

            except Exception as e:
                logging.warning(f"⚠️ Skipping {model_name}: {e}")
                continue  # Skip models that fail config loading

    return valid_models