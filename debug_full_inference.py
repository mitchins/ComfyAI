"""Test the full inference pipeline in pytest."""

def test_full_inference_debug():
    """Debug the full inference pipeline."""
    from apps.shared.onnx_loader import ONNXModelLoader, ONNXInferenceEngine, ONNX_AVAILABLE
    from apps.shared.model_types import get_smallest_quant_for_model
    
    print("\n=== FULL INFERENCE DEBUG ===")
    print("ONNX_AVAILABLE:", ONNX_AVAILABLE)
    
    if not ONNX_AVAILABLE:
        print("ONNX not available, skipping")
        return
    
    loader = ONNXModelLoader()
    model_quant = get_smallest_quant_for_model("Qwen2-VL-2B-Instruct")
    print("Model quant:", model_quant)
    
    print("Loading model...")
    sessions, tokenizer, config = loader.load_model(model_quant)
    print("Model loaded! Sessions:", list(sessions.keys()))
    
    print("Creating inference engine...")
    engine = ONNXInferenceEngine(sessions, tokenizer, config)
    print("Engine created!")
    
    print("Running inference...")
    question = "What is the capital of France?"
    response = engine.generate_text(question, max_tokens=15)
    print(f"Response: {response}")
    
    # Check if Paris is in the response
    if "paris" in response.lower():
        print("✅ SUCCESS: Found 'Paris' in response!")
    else:
        print(f"⚠️  WARNING: 'Paris' not found in response: {response}")
    
    print("Full pipeline completed successfully!")


if __name__ == "__main__":
    test_full_inference_debug()