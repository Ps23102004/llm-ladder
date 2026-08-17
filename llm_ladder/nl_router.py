from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_PROMPT_TEMPLATE = """Classify the user's input into JSON: {{"kind": "triage_repo" or "ask", "repo": "owner/repo" or null}}.

"triage_repo" means the user wants to go through / sort / triage open issues on a specific GitHub repo they named.
"ask" means anything else, including general questions or requests with no repo named.

Examples:
Input: triage the issues in facebook/react
Output: {{"kind": "triage_repo", "repo": "facebook/react"}}

Input: can you go through open issues on torvalds/linux and sort them
Output: {{"kind": "triage_repo", "repo": "torvalds/linux"}}

Input: what's a good commit message style
Output: {{"kind": "ask", "repo": null}}

Input: explain what this repo does
Output: {{"kind": "ask", "repo": null}}

Input: can you help me out with some issues
Output: {{"kind": "ask", "repo": null}}

Input: {text}
Output:"""


@dataclass
class Intent:
    kind: str  # "triage_repo" or "ask"
    repo: str | None  # "owner/repo" slot, only set when kind == "triage_repo"
    text: str  # original user input, always present


def route_command(text: str, model: str, chat_fn: Callable[[str, str], str]) -> Intent:
    """Classify `text` into an Intent by calling `chat_fn(prompt, model)`.

    `chat_fn` is the chat callable of the currently-configured provider
    (OllamaProvider().chat or APIKeyProvider(...).chat), so NL routing
    works even when no local Ollama is installed. Any failure from the
    backend falls back to Intent(kind="ask").
    """
    prompt = _PROMPT_TEMPLATE.format(text=text)
    fallback = Intent(kind="ask", repo=None, text=text)
    try:
        content = chat_fn(prompt, model)
    except Exception:  # noqa: BLE001 - any backend failure means "can't classify"
        return fallback
    if not isinstance(content, str):
        return fallback

    cleaned = _CODE_FENCE_RE.sub("", content).strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return fallback

    if not isinstance(data, dict):
        return fallback

    kind = data.get("kind")
    if kind == "ask":
        return Intent(kind="ask", repo=None, text=text)

    if kind == "triage_repo":
        repo = data.get("repo")
        if isinstance(repo, str) and _REPO_RE.match(repo):
            return Intent(kind="triage_repo", repo=repo, text=text)
        return fallback

    return fallback


if __name__ == "__main__":

    def _chat_fn_factory(content: str):
        def _chat(prompt: str, model: str) -> str:
            return content
        return _chat

    cases = [
        ('{"kind": "triage_repo", "repo": "facebook/react"}', "triage_repo", "facebook/react"),
        ('{"kind": "ask", "repo": null}', "ask", None),
        ("not json at all", "ask", None),
        ('{"kind": "triage_repo"}', "ask", None),
        ('{"kind": "triage_repo", "repo": "not-a-repo-slug"}', "ask", None),
        ('```json\n{"kind": "triage_repo", "repo": "torvalds/linux"}\n```', "triage_repo", "torvalds/linux"),
    ]

    for canned, expect_kind, expect_repo in cases:
        intent = route_command("some input", "test-model", _chat_fn_factory(canned))
        assert intent.kind == expect_kind, f"{canned!r} -> kind {intent.kind!r}, expected {expect_kind!r}"
        assert intent.repo == expect_repo, f"{canned!r} -> repo {intent.repo!r}, expected {expect_repo!r}"
        assert intent.text == "some input"

    print(f"all {len(cases)} self-test cases passed")
