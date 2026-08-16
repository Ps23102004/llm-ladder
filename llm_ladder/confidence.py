from __future__ import annotations

from collections import Counter
from typing import Sequence


def majority_vote(answers: list[str]) -> tuple[str, float]:
    if not answers:
        raise ValueError("no answers to vote on")

    total = len(answers)
    normalized: list[str] = []
    first_original: dict[str, str] = {}
    for answer in answers:
        norm = answer.strip().lower()
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
