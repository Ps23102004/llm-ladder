from __future__ import annotations

import os
import subprocess
from typing import Any, Callable

from . import jobs

# Small, commonly-available Ollama tags offered by the model-picker UI.
CURATED_MODELS = ["gemma3:4b", "qwen2.5:7b"]


def _safe_env() -> dict[str, str]:
    """Environment for child `ollama` processes.

    A copy of the current environment with LLM_LADDER_API_KEY removed —
    the user's API key must never be exposed to a child process.
    """
    env = dict(os.environ)
    env.pop("LLM_LADDER_API_KEY", None)
    return env


def _installed_tags() -> list[str]:
    """Run `ollama list` and return the NAME column entries.

    Ollama's output looks like::

        NAME              ID          SIZE      MODIFIED
        gemma3:4b         abc123...   3.3 GB    2 days ago
        llama3:latest     def456...   4.7 GB    3 weeks ago

    We parse the first whitespace-delimited token of every non-header line
    and tolerate table borders/blank lines.
    """
    proc = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, timeout=10,
        env=_safe_env()
    )
    if proc.returncode != 0:
        raise RuntimeError(f"'ollama list' failed: {proc.stderr.strip()}")
    tags: list[str] = []
    for line in proc.stdout.splitlines():
        name = line.split()[0] if line.split() else ""
        if not name or name == "NAME" or set(name) <= {"-", "=", " "}:
            continue
        tags.append(name)
    return tags


def list_installed_models() -> list[str]:
    """Tags already present in `ollama list`, or [] if that can't be determined."""
    try:
        return _installed_tags()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, RuntimeError):
        return []


def is_model_pulled(tag: str) -> bool:
    """True if ``tag`` already appears in `ollama list`.

    Matching is tolerant of ':latest' quirks: 'llama3' matches a listed
    'llama3:latest' and vice versa.
    """
    try:
        installed = _installed_tags()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, RuntimeError):
        return False
    wanted = tag.removesuffix(":latest")
    for listed in installed:
        listed_base = listed.removesuffix(":latest")
        if listed == tag or listed_base == wanted:
            return True
    return False


def pull_model_job(tag: str) -> str:
    """Start a background job that runs `ollama pull <tag>`.

    Progress lines from the pull stream are surfaced via the job's
    ``progress`` field. Returns the job id immediately.
    """
    def _pull(report_progress: Callable[[dict], None]) -> dict[str, Any]:
        proc = subprocess.Popen(
            ["ollama", "pull", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_safe_env(),
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if line:
                report_progress({"line": line})
                tail.append(line)
                del tail[:-5]  # keep only the last few lines for error reporting
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"`ollama pull {tag}` exited with {returncode}: {' | '.join(tail[-3:])}"
            )
        return {"tag": tag, "pulled": True}

    return jobs.start_job(_pull)
