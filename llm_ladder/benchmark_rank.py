from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankedEntry:
    model: str
    score: float
    rank: int


def rank_models(scores: dict[str, float]) -> list[RankedEntry]:
    """Ordinal rank: best score gets rank N (N = number of models), worst
    gets rank 1. Ties are broken by input (insertion) order — the
    first-inserted of a tied pair gets the higher rank, matching
    confidence.py's existing "first appearing wins" tie-break convention.
    """
    n = len(scores)
    if n == 0:
        return []
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [RankedEntry(model=model, score=score, rank=n - i) for i, (model, score) in enumerate(ordered)]
