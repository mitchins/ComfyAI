"""Debug ONNX runtime issues in pytest vs direct execution."""

def test_onnx_import():
    """Debug ONNX import in pytest context."""
    print("\n=== PYTEST EXECUTION ===")
    
    # Test direct onnxruntime import
    try:
        import onnxruntime
        print("Direct onnxruntime import: SUCCESS")
        print("onnxruntime version:", getattr(onnxruntime, '__version__', 'unknown'))
        print("Has InferenceSession:", hasattr(onnxruntime, 'InferenceSession'))
        if hasattr(onnxruntime, 'InferenceSession'):
            print("InferenceSession:", onnxruntime.InferenceSession)
    except Exception as e:
        print("Direct onnxruntime import: FAILED -", e)
    
    # Test our module import
    try:
        from apps.shared.onnx_loader import ONNX_AVAILABLE, ort
        print("Our module ONNX_AVAILABLE:", ONNX_AVAILABLE)
        print("Our module ort type:", type(ort))
        print("Our module ort:", ort)
        print("Our module hasattr InferenceSession:", hasattr(ort, 'InferenceSession'))
        if hasattr(ort, 'InferenceSession'):
            print("Our module InferenceSession:", ort.InferenceSession)
        else:
            print("Our module ort attributes:", [attr for attr in dir(ort) if not attr.startswith('_')])
    except Exception as e:
        print("Our module import: FAILED -", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_onnx_import()