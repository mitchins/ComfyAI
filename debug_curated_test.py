"""Debug the exact curated model loading issue without try/except."""

def test_curated_model_debug():
    """Debug curated model loading without exception handling."""
    from apps.shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine, ONNX_AVAILABLE
    from apps.shared.model_types import get_smallest_quant_for_model
    
    print("\n=== CURATED MODEL DEBUG ===")
    print("ONNX_AVAILABLE:", ONNX_AVAILABLE)
    
    loader = ONNXModelLoader()
    model_quant = get_smallest_quant_for_model("Qwen2-VL-2B-Instruct")
    print("Model quant:", model_quant)
    
    print("About to call loader.load_model...")
    
    # This will fail with the actual error, not be caught
    sessions, tokenizer, config = loader.load_model(model_quant)
    print("SUCCESS: Model loaded!")


if __name__ == "__main__":
    test_curated_model_debug()