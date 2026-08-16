from unittest.mock import patch

import pytest

from llm_ladder.config import ChainConfig, TierConfig
from llm_ladder.engine import run_cascade
from llm_ladder.ledger import Ledger
from llm_ladder.ollama_client import OllamaConnectionError, OllamaModelNotFoundError


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


def _chain(samples: int, threshold: float = 0.66) -> ChainConfig:
    return ChainConfig(name="test", tiers=[TierConfig(model="m", samples=samples, threshold=threshold)])


def test_partial_success_does_not_inflate_confidence_past_threshold(tmp_path):
    # 3 requested samples: 1 succeeds, 2 raise. A single surviving sample
    # trivially "agrees with itself" (confidence 1.0) but must NOT be treated
    # as confident enough to stop early on a partial batch.
    chain = ChainConfig(
        name="test",
        tiers=[
            TierConfig(model="fast", samples=3, threshold=0.5),
            TierConfig(model="slow", samples=1, threshold=0.0),
        ],
    )
    calls = {"n": 0}

    def fake_chat(model, prompt, host=None):
        if model == "fast":
            calls["n"] += 1
            if calls["n"] == 1:
                return _chat_response("answer")
            raise OllamaConnectionError("transient")
        return _chat_response("final")

    with patch("llm_ladder.engine.chat", side_effect=fake_chat):
        ledger = Ledger(path=str(tmp_path / "ledger.jsonl"))
        result = run_cascade("prompt", "test", chain, ledger)

    # Must have escalated to tier 1 rather than trusting the partial tier-0 batch.
    assert result.tier_index == 1
    assert result.model == "slow"


def test_full_batch_at_threshold_stops_early(tmp_path):
    chain = _chain(samples=3, threshold=0.5)

    with patch("llm_ladder.engine.chat", return_value=_chat_response("answer")):
        ledger = Ledger(path=str(tmp_path / "ledger.jsonl"))
        result = run_cascade("prompt", "test", chain, ledger)

    assert result.tier_index == 0
    assert result.confidence == 1.0


def test_all_samples_fail_raises_last_error(tmp_path):
    chain = _chain(samples=2)

    with patch("llm_ladder.engine.chat", side_effect=OllamaConnectionError("down")):
        ledger = Ledger(path=str(tmp_path / "ledger.jsonl"))
        with pytest.raises(OllamaConnectionError):
            run_cascade("prompt", "test", chain, ledger)


def test_zero_samples_raises_clean_error_not_typeerror(tmp_path):
    chain = ChainConfig(name="test", tiers=[TierConfig(model="m", samples=0, threshold=0.0)])

    with patch("llm_ladder.engine.chat", return_value=_chat_response("x")):
        ledger = Ledger(path=str(tmp_path / "ledger.jsonl"))
        with pytest.raises(OllamaConnectionError):
            run_cascade("prompt", "test", chain, ledger)


def test_model_not_found_fails_fast_without_burning_remaining_samples(tmp_path):
    chain = _chain(samples=5)
    calls = {"n": 0}

    def fake_chat(model, prompt, host=None):
        calls["n"] += 1
        raise OllamaModelNotFoundError("no such model")

    with patch("llm_ladder.engine.chat", side_effect=fake_chat):
        ledger = Ledger(path=str(tmp_path / "ledger.jsonl"))
        with pytest.raises(OllamaModelNotFoundError):
            run_cascade("prompt", "test", chain, ledger)

    # Should not have retried across all 5 configured samples.
    assert calls["n"] == 1
