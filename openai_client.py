import requests


import base64


def chat_completion(
    endpoint: str,
    model: str,
    messages,
    api_key: str | None = None,
    timeout: int = 30,
    images=None,
    **kwargs,
) -> str:
    """Send a chat completion request to an OpenAI compatible endpoint.

    Parameters
    ----------
    endpoint : str
        Base URL of the API.
    model : str
        Model name to query.
    messages : list
        Chat message history.
    api_key : str | None, optional
        Optional API key.
    timeout : int, optional
        Request timeout.
    images : list[bytes | str] | None, optional
        Optional list of images to include in the request. ``bytes`` are
        base64-encoded automatically.
    """

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "messages": messages}

    if images:
        encoded = []
        for img in images:
            if isinstance(img, bytes):
                img = base64.b64encode(img).decode("utf-8")
            encoded.append(img)
        payload["images"] = encoded

    payload.update(kwargs)

    url = endpoint.rstrip("/") + "/v1/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
