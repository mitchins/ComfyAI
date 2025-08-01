#!/usr/bin/env python3
"""Quick test to verify vision strategy pattern works."""

import base64
import requests
from pathlib import Path

def test_vision_strategy():
    """Test vision strategy pattern with pizza image."""
    pizza_path = Path("tests/pizza.jpg")
    if not pizza_path.exists():
        print(f"❌ Pizza image not found at {pizza_path}")
        return False
    
    # Convert to base64
    with open(pizza_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Test Qwen2-VL first (should work best with our implementation)
    print(f"\n🧪 Testing Qwen2-VL vision strategy...")
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "Qwen2-VL-2B-Instruct/UINT8",
                "messages": [{"role": "user", "content": "What is shown in the provided image?"}],
                "images": [image_base64],
                "max_tokens": 50
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"Response: {content}")
            
            # Check for food keywords
            food_keywords = ['pizza', 'food', 'cheese', 'crust', 'slice', 'meal']
            found_food = any(keyword in content.lower() for keyword in food_keywords)
            
            # Check for hallucination patterns
            hallucination_patterns = ['black and white', 'botanical', 'ceramic', 'mug', 'woman', 'cat']
            found_hallucination = any(pattern in content.lower() for pattern in hallucination_patterns)
            
            if found_food and not found_hallucination:
                print(f"✅ SUCCESS - Qwen2-VL vision strategy working!")
                print(f"   Found food keywords, no hallucinations detected")
                return True
            elif found_food:
                print(f"⚠️  PARTIAL - Found food but also some hallucinations")
                return True
            else:
                print(f"❌ FAIL - Still hallucinating or not seeing image")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_vision_strategy()
    print(f"\n{'🎉 Vision strategy test PASSED!' if success else '💥 Vision strategy test FAILED!'}")