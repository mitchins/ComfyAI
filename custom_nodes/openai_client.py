import requests


def chat_completion(endpoint: str, model: str, messages, api_key: str | None = None, timeout: int = 30, **kwargs) -> str:
    """Send a chat completion request to an OpenAI compatible endpoint."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages}
    payload.update(kwargs)
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
