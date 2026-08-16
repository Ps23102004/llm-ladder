from __future__ import annotations

import time
from dataclasses import dataclass

from llm_ladder.config import ChainConfig
from llm_ladder.ledger import Ledger
from llm_ladder.ollama_client import chat, OllamaConnectionError, OllamaModelNotFoundError
from llm_ladder.confidence import majority_vote


@dataclass
class CascadeResult:
    answer: str
    confidence: float
    tier_index: int
    model: str


def run_cascade(prompt: str, chain: str, chain_config: ChainConfig, ledger: Ledger | None = None) -> CascadeResult:
    """Run a prompt through a chain's tiers, recording each attempt to the ledger.

    Raises ValueError if the chain has no tiers. Raises OllamaConnectionError only
    if every sample in a tier fails; a tier proceeds on partial success.
    """
    if ledger is None:
        ledger = Ledger()

    for i, tier in enumerate(chain_config.tiers):
        start_time = time.perf_counter()
        results: list[str] = []
        last_error: OllamaConnectionError | None = None
        for _ in range(tier.samples):
            try:
                resp = chat(tier.model, prompt)
                results.append(resp.get("message", {}).get("content", ""))
            except OllamaModelNotFoundError:
                # Not transient — retrying more samples won't make the model appear.
                raise
            except OllamaConnectionError as exc:
                last_error = exc
        if not results:
            raise last_error or OllamaConnectionError(
                f"tier {i} ({tier.model}) has samples={tier.samples}, no attempts were made"
            )
        duration = time.perf_counter() - start_time

        answer, confidence = majority_vote(results)
        is_last = i == len(chain_config.tiers) - 1
        full_batch = len(results) == tier.samples

        ledger.record(
            chain=chain,
            tier_index=i,
            model=tier.model,
            confidence=confidence,
            samples=tier.samples,
            escalated=(i > 0),
            used_last_tier=is_last,
            duration_s=duration,
        )

        # A partial batch (some samples failed) never counts as confident enough
        # to stop early — a single surviving sample trivially "agrees with
        # itself" at confidence 1.0, which would otherwise overstate certainty.
        if (full_batch and confidence >= tier.threshold) or is_last:
            return CascadeResult(answer=answer, confidence=confidence, tier_index=i, model=tier.model)

    raise ValueError("chain has no tiers")
