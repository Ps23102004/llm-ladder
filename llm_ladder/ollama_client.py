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


DEFAULT_TIMEOUT = 120


def chat(
    model: str,
    prompt: str,
    host: str | None = None,
    images: list[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """`images` is a list of base64-encoded images (no data: prefix) for a
    vision model; omitted entirely for text models, so text callers are
    unaffected.

    `timeout` defaults to the previous hard-coded 120s. A vision model on a
    cold start pays a multi-GB load before it emits its first token and can
    exceed that, so image callers raise it."""
    host = resolve_host(host)
    url = f"{host}/api/chat"
    message: dict = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    payload = {
        "model": model,
        "messages": [message],
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
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
