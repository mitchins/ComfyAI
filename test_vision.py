#!/usr/bin/env python3
"""Test vision capabilities of ONNX chat server."""

import requests
import base64
import json

def test_vision():
    """Test vision + text with Qwen2-VL model."""
    
    # Read and encode the image
    image_path = "/Users/mitchellcurrie/Downloads/ComfyUI_00040_.png"
    try:
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        print(f"✅ Image loaded: {len(image_b64)} characters")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return False
    
    # Test vision + text request
    payload = {
        "model": "onnx-community/Qwen2-VL-2B-Instruct",
        "messages": [
            {
                "role": "user", 
                "content": "What do you see in this image? Describe it briefly."
            }
        ],
        "images": [image_b64],
        "max_tokens": 100
    }
    
    try:
        print("🔄 Sending vision request...")
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60  # Vision processing can take longer
        )
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"✅ Vision response: {content}")
            return True
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🖼️ Testing ONNX Vision Capabilities...")
    success = test_vision()
    if success:
        print("\n🎉 Vision test completed successfully!")
    else:
        print("\n💥 Vision test failed!")