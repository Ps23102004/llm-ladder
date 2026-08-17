from unittest.mock import patch

import pytest

from llm_ladder.providers import APIKeyProvider, OllamaProvider


class _FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text

    def json(self):
        return self._payload


def test_ollama_provider_returns_message_content():
    provider = OllamaProvider(host="http://ollama:11434")
    with patch(
        "llm_ladder.ollama_client.chat",
        return_value={"message": {"content": "hi there"}},
    ) as fake_chat:
        reply = provider.chat("prompt", "llama3")
    assert reply == "hi there"
    assert fake_chat.call_count == 1
    assert fake_chat.call_args.kwargs == {
        "model": "llama3",
        "prompt": "prompt",
        "host": "http://ollama:11434",
    }


def test_ollama_provider_empty_response_returns_empty_string():
    provider = OllamaProvider()
    with patch("llm_ladder.ollama_client.chat", return_value={}):
        assert provider.chat("p", "m") == ""


def test_api_key_provider_openai_style_parses_content():
    provider = APIKeyProvider(base_url="https://api.example.com/v1", api_key="sk-test")
    response = _FakeResponse({"choices": [{"message": {"content": "answer"}}]})
    with patch("llm_ladder.providers.requests.post", return_value=response) as fake_post:
        reply = provider.chat("hello", "gpt-4o")
    assert reply == "answer"
    url, = fake_post.call_args.args
    assert url == "https://api.example.com/v1/chat/completions"
    headers = fake_post.call_args.kwargs["headers"]
    assert headers == {"Authorization": "Bearer sk-test"}
    assert fake_post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]


def test_api_key_provider_anthropic_style_parses_content():
    provider = APIKeyProvider(
        base_url="https://api.z.ai/api/anthropic/",
        api_key="ak-test",
        api_style="anthropic",
    )
    response = _FakeResponse({"content": [{"type": "text", "text": "claude reply"}]})
    with patch("llm_ladder.providers.requests.post", return_value=response) as fake_post:
        reply = provider.chat("hello", "claude-3")
    assert reply == "claude reply"
    url, = fake_post.call_args.args
    assert url == "https://api.z.ai/api/anthropic/v1/messages"
    headers = fake_post.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "ak-test"
    assert headers["anthropic-version"] == "2023-06-01"
    payload = fake_post.call_args.kwargs["json"]
    assert payload["model"] == "claude-3"
    assert payload["max_tokens"] == 1024


def test_api_key_provider_unknown_style_falls_back_to_openai():
    provider = APIKeyProvider(base_url="https://api.example.com", api_key="k", api_style="weird")
    assert provider.api_style == "openai"
    response = _FakeResponse({"choices": [{"message": {"content": "x"}}]})
    with patch("llm_ladder.providers.requests.post", return_value=response) as fake_post:
        provider.chat("p", "m")
    assert fake_post.call_args.args[0] == "https://api.example.com/chat/completions"


def test_api_key_provider_strips_trailing_slash_from_base_url():
    provider = APIKeyProvider(base_url="https://api.example.com/v1/", api_key="k")
    assert provider.base_url == "https://api.example.com/v1"


def test_api_key_provider_non_2xx_raises_with_status():
    provider = APIKeyProvider(base_url="https://api.example.com", api_key="k")
    response = _FakeResponse({}, status_code=500, text="server exploded")
    with patch("llm_ladder.providers.requests.post", return_value=response):
        with pytest.raises(RuntimeError) as exc:
            provider.chat("p", "m")
    assert "500" in str(exc.value)
    assert "server exploded" in str(exc.value)


def test_api_key_provider_error_truncates_body_to_200_chars():
    provider = APIKeyProvider(base_url="https://api.example.com", api_key="k")
    response = _FakeResponse({}, status_code=400, text="x" * 1000)
    with patch("llm_ladder.providers.requests.post", return_value=response):
        with pytest.raises(RuntimeError) as exc:
            provider.chat("p", "m")
    assert "x" * 200 in str(exc.value)
    assert "x" * 201 not in str(exc.value)
