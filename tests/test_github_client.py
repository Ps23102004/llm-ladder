from unittest.mock import patch

import pytest
import requests

from llm_ladder.github_client import GITHUB_API_BASE, Issue, fetch_open_issues


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def _issue_payload(number, title="bug", body=None, with_pr=False):
    item = {
        "number": number,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/o/r/issues/{number}",
    }
    if with_pr:
        item["pull_request"] = {"url": "https://api.github.com/pulls/1"}
    return item


def test_parses_issues_from_response():
    payload = [
        _issue_payload(1, title="crash on startup", body="steps..."),
        _issue_payload(2, title="no body"),
    ]
    with patch("llm_ladder.github_client.requests.get", return_value=_FakeResponse(payload)) as fake_get:
        issues = fetch_open_issues("owner/repo")
    assert issues == [
        Issue(number=1, title="crash on startup", body="steps...",
              url="https://github.com/o/r/issues/1"),
        Issue(number=2, title="no body", body="", url="https://github.com/o/r/issues/2"),
    ]
    url, = fake_get.call_args.args
    assert url == f"{GITHUB_API_BASE}/owner/repo/issues"
    params = fake_get.call_args.kwargs["params"]
    assert params == {"state": "open", "per_page": 30}


def test_excludes_pull_requests():
    payload = [
        _issue_payload(1),
        _issue_payload(2, title="a PR", with_pr=True),
    ]
    with patch("llm_ladder.github_client.requests.get", return_value=_FakeResponse(payload)):
        issues = fetch_open_issues("owner/repo")
    assert [i.number for i in issues] == [1]


def test_null_body_becomes_empty_string():
    with patch("llm_ladder.github_client.requests.get",
               return_value=_FakeResponse([_issue_payload(1, body=None)])):
        assert fetch_open_issues("o/r")[0].body == ""


def test_limit_truncates_results():
    payload = [_issue_payload(i) for i in range(1, 6)]
    with patch("llm_ladder.github_client.requests.get", return_value=_FakeResponse(payload)):
        issues = fetch_open_issues("o/r", limit=2)
    assert [i.number for i in issues] == [1, 2]


def test_per_page_capped_at_100():
    with patch("llm_ladder.github_client.requests.get",
               return_value=_FakeResponse([])) as fake_get:
        fetch_open_issues("o/r", limit=500)
    assert fake_get.call_args.kwargs["params"]["per_page"] == 100


def test_token_sets_authorization_header():
    with patch("llm_ladder.github_client.requests.get",
               return_value=_FakeResponse([])) as fake_get:
        fetch_open_issues("o/r", token="ghp-secret")
    assert fake_get.call_args.kwargs["headers"] == {"Authorization": "token ghp-secret"}


def test_no_token_sends_empty_headers():
    with patch("llm_ladder.github_client.requests.get",
               return_value=_FakeResponse([])) as fake_get:
        fetch_open_issues("o/r")
    assert fake_get.call_args.kwargs["headers"] == {}


def test_404_raises_value_error_with_repo():
    with patch("llm_ladder.github_client.requests.get",
               return_value=_FakeResponse({}, status_code=404)):
        with pytest.raises(ValueError) as exc:
            fetch_open_issues("nope/nope")
    assert "nope/nope" in str(exc.value)


def test_403_raises_rate_limit_error():
    with patch("llm_ladder.github_client.requests.get",
               return_value=_FakeResponse({}, status_code=403)):
        with pytest.raises(ValueError) as exc:
            fetch_open_issues("o/r")
    assert "rate limit" in str(exc.value).lower()


def test_other_error_raises_http_error():
    with patch("llm_ladder.github_client.requests.get",
               return_value=_FakeResponse({}, status_code=500)):
        with pytest.raises(requests.HTTPError):
            fetch_open_issues("o/r")
