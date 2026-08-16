from __future__ import annotations

import os

import requests

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaConnectionError(Exception):
    """Raised when communication with the Ollama endpoint fails."""


class OllamaModelNotFoundError(OllamaConnectionError):
    """Raised when Ollama reports the requested model isn't pulled (HTTP 404)."""


def resolve_host(host: str | None = None) -> str:
    return host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)


def chat(
    model: str,
    prompt: str,
    host: str | None = None,
) -> dict:
    host = resolve_host(host)
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
        if exc.response is not None and exc.response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{model}' isn't available at {host}. Run `ollama pull {model}` first."
            ) from exc
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
    host: str | None = None,
) -> list[str]:
    results: list[str] = []
    for _ in range(n):
        resp = chat(model, prompt, host)
        message = resp.get("message", {})
        content = message.get("content", "")
        results.append(content)
    return results
