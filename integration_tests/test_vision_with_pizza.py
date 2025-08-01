"""Integration test to ensure vision models can see pizza and not hallucinate."""

import pytest
import base64
from pathlib import Path

# Mark this as integration test
pytestmark = pytest.mark.integration


class TestVisionWithPizza:
    """Test that vision models correctly process the pizza.jpg image."""
    
    def setup_method(self):
        """Load the pizza image for testing."""
        self.pizza_path = Path(__file__).parent.parent / "tests" / "pizza.jpg"
        assert self.pizza_path.exists(), f"Pizza test image not found at {self.pizza_path}"
        
        # Load and encode image
        with open(self.pizza_path, "rb") as f:
            self.pizza_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    @pytest.mark.asyncio
    async def test_vision_models_see_pizza(self):
        """Test that vision models identify pizza/food in the image."""
        try:
            from apps.onnx_chat.main import generate_text
        except ImportError:
            pytest.skip("ONNX chat module not available")
        
        # Keywords that should appear when describing pizza
        expected_keywords = ['pizza', 'food', 'cheese', 'meal', 'dish', 'slice', 'crust']
        
        # Test with different models
        test_cases = [
            ("Qwen2-VL-2B-Instruct/UINT8", "Qwen2-VL"),
            ("SmolVLM-256M-Instruct/UINT8", "SmolVLM"),
        ]
        
        for model_name, model_label in test_cases:
            print(f"\nTesting {model_label}...")
            
            try:
                # Generate description of the image
                response = await generate_text(
                    text="What's in this image? Describe what you see.",
                    model_name=model_name,
                    max_tokens=100,
                    images=[self.pizza_base64]
                )
                
                print(f"Response: {response}")
                
                # Check if any food-related keywords appear
                response_lower = response.lower()
                found_keywords = [kw for kw in expected_keywords if kw in response_lower]
                
                # Fail if we see hallucination patterns
                hallucination_patterns = [
                    "black and white photo of a black and white",
                    "vintage botanical illustration",
                    "ceramic mug",
                    "woman",
                    "cat",
                    "scenery"
                ]
                
                found_hallucinations = [pattern for pattern in hallucination_patterns if pattern in response_lower]
                
                assert len(found_keywords) > 0, (
                    f"{model_label} failed to identify food/pizza. "
                    f"Response: {response}. "
                    f"This suggests images are being dropped!"
                )
                
                assert len(found_hallucinations) == 0, (
                    f"{model_label} is hallucinating. "
                    f"Found patterns: {found_hallucinations}. "
                    f"Response: {response}"
                )
                
                print(f"✅ {model_label} correctly identified: {found_keywords}")
                
            except Exception as e:
                # Log the error but don't fail the entire test suite
                print(f"⚠️  {model_label} error: {e}")
                # Re-raise to fail this specific model test
                raise
    
    def test_pizza_image_exists(self):
        """Verify the pizza test image exists and is valid."""
        assert self.pizza_path.exists()
        assert self.pizza_path.stat().st_size > 0
        
        # Verify it's a valid image
        try:
            from PIL import Image
            img = Image.open(self.pizza_path)
            assert img.size[0] > 0 and img.size[1] > 0
            print(f"✅ Pizza image valid: {img.size} {img.mode}")
        except Exception as e:
            pytest.fail(f"Invalid pizza image: {e}")