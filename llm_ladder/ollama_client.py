from __future__ import annotations

import requests


class OllamaConnectionError(Exception):
    """Raised when communication with the Ollama endpoint fails."""


def chat(
    model: str,
    prompt: str,
    host: str = "http://127.0.0.1:11434",
) -> dict:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise OllamaConnectionError(
            f"Could not reach Ollama endpoint at {host}: {exc}"
        ) from exc
    except requests.HTTPError as exc:
        raise OllamaConnectionError(
            f"Ollama endpoint at {host} returned status {exc.response.status_code}: {exc}"
        ) from exc
    except ValueError as exc:
        raise OllamaConnectionError(
            f"Ollama endpoint at {host} returned a response that is not valid JSON: {exc}"
        ) from exc


def chat_n(
    model: str,
    prompt: str,
    n: int,
    host: str = "http://127.0.0.1:11434",
) -> list[str]:
    results: list[str] = []
    for _ in range(n):
        resp = chat(model, prompt, host)
        message = resp.get("message", {})
        content = message.get("content", "")
        results.append(content)
    return results
