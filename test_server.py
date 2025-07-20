#!/usr/bin/env python3
"""Test script to verify ONNX chat server is working."""

import requests
import json
import sys

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
            return True
    except Exception as e:
        print(f"Health check failed: {e}")
    return False

def test_chat():
    """Test chat endpoint."""
    payload = {
        "model": "onnx-community/gemma-3n-E2B-it-ONNX",
        "messages": [{"role": "user", "content": "Hello, explain quantum computing briefly"}],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        print(f"Chat request: {response.status_code}")
        if response.status_code == 200:
            print(f"Chat response: {response.json()}")
            return True
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Chat request failed: {e}")
    return False

if __name__ == "__main__":
    print("Testing ONNX Chat Server...")
    
    if test_health():
        print("✅ Health check passed")
        if test_chat():
            print("✅ Chat endpoint working")
        else:
            print("❌ Chat endpoint failed")
            sys.exit(1)
    else:
        print("❌ Health check failed - is the server running?")
        sys.exit(1)