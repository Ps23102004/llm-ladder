from llm_ladder.nl_router import Intent, route_command


def _chat_fn(content):
    def chat(prompt, model):
        return content
    return chat


def _route(content, text="go through issues in facebook/react"):
    return route_command(text, "test-model", _chat_fn(content))


def test_triage_repo_with_valid_slug():
    intent = _route('{"kind": "triage_repo", "repo": "facebook/react"}')
    assert intent.kind == "triage_repo"
    assert intent.repo == "facebook/react"
    assert intent.text == "go through issues in facebook/react"


def test_ask_kind():
    intent = _route('{"kind": "ask", "repo": null}')
    assert intent == Intent(kind="ask", repo=None, text="go through issues in facebook/react")


def test_code_fenced_json_is_unwrapped():
    intent = _route('```json\n{"kind": "triage_repo", "repo": "torvalds/linux"}\n```')
    assert intent.kind == "triage_repo"
    assert intent.repo == "torvalds/linux"


def test_non_json_reply_falls_back_to_ask():
    assert _route("not json at all").kind == "ask"


def test_triage_repo_without_repo_falls_back_to_ask():
    assert _route('{"kind": "triage_repo"}').kind == "ask"


def test_triage_repo_with_invalid_slug_falls_back_to_ask():
    assert _route('{"kind": "triage_repo", "repo": "not a repo slug"}').kind == "ask"


def test_triage_repo_with_sluggy_non_repo_string_falls_back_to_ask():
    # Matches the "looks like a repo" shape loosely but has two slashes.
    assert _route('{"kind": "triage_repo", "repo": "a/b/c"}').kind == "ask"


def test_unknown_kind_falls_back_to_ask():
    assert _route('{"kind": "whatever", "repo": "facebook/react"}').kind == "ask"


def test_json_list_reply_falls_back_to_ask():
    assert _route('["triage_repo", "facebook/react"]').kind == "ask"


def test_non_string_reply_falls_back_to_ask():
    intent = route_command("hi", "m", lambda prompt, model: None)
    assert intent.kind == "ask"


def test_backend_exception_falls_back_to_ask():
    def boom(prompt, model):
        raise RuntimeError("provider down")
    intent = route_command("triage issues in facebook/react", "m", boom)
    assert intent == Intent(kind="ask", repo=None, text="triage issues in facebook/react")


def test_original_text_preserved_in_fallback():
    intent = _route("garbage")
    assert intent.text == "go through issues in facebook/react"


def test_prompt_includes_user_text():
    seen = {}
    def spy(prompt, model):
        seen["prompt"] = prompt
        seen["model"] = model
        return '{"kind": "ask", "repo": null}'
    route_command("triage facebook/react issues", "my-model", spy)
    assert "triage facebook/react issues" in seen["prompt"]
    assert seen["model"] == "my-model"
