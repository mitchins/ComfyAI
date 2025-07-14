import base64
import json
import requests


def image_bytes_to_data_url(image_bytes: bytes) -> str:
    """Convert raw image bytes to a base64 data URL."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def chat_completion(endpoint: str, model: str, messages: list, api_key: str | None = None) -> str:
    """Send a chat completion request to an OpenAI compatible endpoint."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages}
    response = requests.post(f"{endpoint}/v1/chat/completions", headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

