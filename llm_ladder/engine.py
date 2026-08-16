from __future__ import annotations

import time
from dataclasses import dataclass

from llm_ladder.config import ChainConfig
from llm_ladder.ledger import Ledger
from llm_ladder.ollama_client import chat_n
from llm_ladder.confidence import majority_vote


@dataclass
class CascadeResult:
    answer: str
    confidence: float
    tier_index: int
    model: str


def run_cascade(prompt: str, chain: str, chain_config: ChainConfig, ledger: Ledger | None = None) -> CascadeResult:
    """Run a prompt through a chain's tiers, recording each attempt to the ledger.

    Raises ValueError if the chain has no tiers. Propagates OllamaConnectionError
    from chat_n on connection failure.
    """
    if ledger is None:
        ledger = Ledger()

    for i, tier in enumerate(chain_config.tiers):
        start_time = time.perf_counter()
        results = chat_n(tier.model, prompt, tier.samples)
        duration = time.perf_counter() - start_time

        answer, confidence = majority_vote(results)
        is_last = i == len(chain_config.tiers) - 1

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

        if confidence >= tier.threshold or is_last:
            return CascadeResult(answer=answer, confidence=confidence, tier_index=i, model=tier.model)

    raise ValueError("chain has no tiers")
