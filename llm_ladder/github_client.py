"""Fetch open GitHub issues for a repository.

Thin wrapper over the GitHub REST API used by the ladder benchmark
subsystem to turn a repo's open issues into candidate tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

GITHUB_API_BASE = "https://api.github.com/repos"


@dataclass
class Issue:
    number: int
    title: str
    body: str
    url: str


def fetch_open_issues(
    repo: str, token: str | None = None, limit: int = 30
) -> list[Issue]:
    """Return up to `limit` open issues for `repo` (pull requests excluded).

    Raises:
        ValueError: on 404 (repo not found) or 403 (rate limit exceeded).
        requests.HTTPError: on any other non-2xx response.
    """
    url = f"{GITHUB_API_BASE}/{repo}/issues"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    params = {"state": "open", "per_page": min(limit, 100)}

    response = requests.get(url, headers=headers, params=params, timeout=10)
    if response.status_code == 404:
        raise ValueError(f"repo not found: {repo}")
    if response.status_code == 403:
        raise ValueError("GitHub API rate limit exceeded, set GITHUB_TOKEN")
    response.raise_for_status()

    issues = [
        Issue(
            number=item["number"],
            title=item["title"],
            body=item.get("body") or "",
            url=item["html_url"],
        )
        for item in response.json()
        if "pull_request" not in item
    ]
    return issues[:limit]
