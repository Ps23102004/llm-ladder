"""Provider abstraction over model backends.

Defines a minimal ModelProvider protocol and two implementations:

- OllamaProvider: wraps the existing llm_ladder.ollama_client.chat.
- APIKeyProvider: any OpenAI-compatible /chat/completions endpoint
  authenticated with a bearer API key.
"""

from __future__ import annotations

from typing import Protocol

import requests

from llm_ladder import ollama_client


class ModelProvider(Protocol):
    def chat(self, prompt: str, model: str) -> str:
        """Send a single user prompt to `model` and return its reply text."""
        ...


class OllamaProvider:
    """Provider backed by a local Ollama host via llm_ladder.ollama_client."""

    def __init__(self, host: str | None = None) -> None:
        self.host = host

    def chat(self, prompt: str, model: str) -> str:
        response = ollama_client.chat(model=model, prompt=prompt, host=self.host)
        message = response.get("message", {})
        return message.get("content", "")


class APIKeyProvider:
    """Provider for a hosted chat API using an API key.

    Supports two request/response shapes, selected by ``api_style``:

    - "openai" (default): OpenAI-compatible ``POST {base_url}/chat/completions``,
      ``Authorization: Bearer <key>``, reply at
      ``response.json()["choices"][0]["message"]["content"]``. Most hosted
      APIs (OpenAI itself, Groq, Together, etc.) use this shape.
    - "anthropic": Anthropic Messages API shape, e.g. Claude or z.ai's
      Anthropic-compatible endpoint (``https://api.z.ai/api/anthropic``) —
      ``POST {base_url}/v1/messages``, ``x-api-key: <key>`` +
      ``anthropic-version`` header, reply at
      ``response.json()["content"][0]["text"]``.
    """

    def __init__(self, base_url: str, api_key: str, api_style: str = "openai") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_style = api_style if api_style in ("openai", "anthropic") else "openai"

    def chat(self, prompt: str, model: str) -> str:
        if self.api_style == "anthropic":
            return self._chat_anthropic(prompt, model)
        return self._chat_openai(prompt, model)

    def _chat_openai(self, prompt: str, model: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._post(url, payload, headers)
        return response.json()["choices"][0]["message"]["content"]

    def _chat_anthropic(self, prompt: str, model: str) -> str:
        url = f"{self.base_url}/v1/messages"
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._post(url, payload, headers)
        return response.json()["content"][0]["text"]

    def _post(self, url: str, payload: dict, headers: dict) -> requests.Response:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        if not response.ok:
            # Truncate the body: some APIs echo request headers (including
            # Authorization/x-api-key) back in error bodies, and the full
            # text could leak the user's API key to whatever reads this error.
            snippet = response.text[:200]
            raise RuntimeError(
                f"{url} returned status {response.status_code}: {snippet}"
            )
        return response
