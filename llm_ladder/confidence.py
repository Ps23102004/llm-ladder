from __future__ import annotations

import re
from collections import Counter

# Markdown emphasis/code-fence wrapping at the very start/end only ("**Paris**",
# "`42`") — NOT stripped mid-string, so "C#", "F#", "snake_case" are untouched.
_MARKDOWN_WRAP = re.compile(r"^(\*{1,2}|`+)+|(\*{1,2}|`+)+$")
# Trailing sentence punctuation only ("Paris." == "Paris"). Leading characters
# are never stripped, so a numeric sign ("-5", "+3") stays significant.
_TRAILING_PUNCT = re.compile(r"[.!?]+$")


def _normalize(answer: str) -> str:
    """Cosmetic-difference-insensitive key for agreement comparison."""
    text = " ".join(answer.split())  # collapse all whitespace runs
    text = _MARKDOWN_WRAP.sub("", text)
    text = _TRAILING_PUNCT.sub("", text)
    normalized = text.casefold()
    # Guard against degenerate collapse: if stripping ate the whole answer
    # (e.g. "..." or "?"), fall back to the original so two different
    # punctuation-only answers don't falsely agree.
    return normalized if normalized else " ".join(answer.split()).casefold()


def majority_vote(answers: list[str]) -> tuple[str, float]:
    if not answers:
        raise ValueError("no answers to vote on")

    total = len(answers)
    normalized: list[str] = []
    first_original: dict[str, str] = {}
    for answer in answers:
        norm = _normalize(answer)
        normalized.append(norm)
        if norm not in first_original:
            first_original[norm] = answer

    counts: Counter[str] = Counter(normalized)
    max_count = max(counts.values())

    # Stable tie-breaking: pick the normalized group that appears first in the
    # original input order.
    winner_norm: str | None = None
    for norm in normalized:
        if counts[norm] == max_count:
            winner_norm = norm
            break

    assert winner_norm is not None
    original = first_original[winner_norm]
    fraction = max_count / total
    return original, fraction
