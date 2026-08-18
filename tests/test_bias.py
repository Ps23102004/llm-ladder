"""Tests for `llm_ladder.bias` — thin wrapper over digest's lens fan-out
engine. Model calls (discover_models/chat/run_cascade/check_memory_safety/
free_loaded_models) are patched at their import site inside llm_ladder.digest,
same as test_digest.py, because bias.run_bias delegates the actual fan-out
loop to digest._run_lens_core — bias.py itself never calls Ollama directly."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_ladder.bias import BiasError, BiasResult, run_bias


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


def _cascade_result(answer: str):
    return SimpleNamespace(answer=answer, tier_index=0, confidence=1.0, model="m")


def _model(name: str, size_bytes: int = 100):
    return SimpleNamespace(name=name, size_bytes=size_bytes)


# -- input validation ---------------------------------------------------------


def test_run_bias_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(BiasError, match="nope.txt"):
        run_bias(str(missing))


def test_run_bias_empty_file_raises(tmp_path):
    doc = tmp_path / "empty.txt"
    doc.write_text("   \n  ")
    with pytest.raises(BiasError, match="empty"):
        run_bias(str(doc))


def test_run_bias_url_input_rejected_without_new_dependency():
    with pytest.raises(BiasError, match="URL fetch isn't supported"):
        run_bias("https://example.com/article")
    with pytest.raises(BiasError, match="URL fetch isn't supported"):
        run_bias("http://example.com/article")


def test_run_bias_unknown_chain_raises(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("some article text")
    with pytest.raises(BiasError, match="chain 'nope' not found"):
        run_bias(str(doc), chain="nope")


def test_run_bias_invalid_chains_yaml_raises_bias_error(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("some article text")
    with patch("llm_ladder.bias.load_chains", side_effect=ValueError("bad yaml")):
        with pytest.raises(BiasError, match="chains.yaml"):
            run_bias(str(doc))


# -- fan-out / judge behavior (delegated to digest._run_lens_core) -----------


def test_run_bias_single_model_skips_judge_with_note(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("some article text")
    with (
        patch("llm_ladder.digest.discover_models", return_value=[_model("only")]),
        patch("llm_ladder.digest.free_loaded_models", return_value=None),
        patch("llm_ladder.digest.check_memory_safety", return_value=(True, "")),
        patch("llm_ladder.digest.chat", side_effect=lambda m, p: _chat_response(f"take {m}")),
    ):
        result = run_bias(str(doc))

    assert isinstance(result, BiasResult)
    assert result.source == str(doc)
    assert result.lens.note == "needs >=2 distinct models to compare"
    assert result.lens.takes[0].model == "only"
    assert result.lens.lens_verdict is None


def test_run_bias_two_models_run_judge_with_framing_prompt(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("some article text")
    judge_text = (
        "VERDICT: 2 of 2 models agree the piece leans sympathetic to the subject\n"
        "CONSENSUS:\n- both note the headline downplays the cost figure\n"
        "DISAGREEMENTS:\n- a calls the tone neutral, b calls it favorable\n"
    )
    seen_prompts = []

    def fake_chat(model, prompt):
        seen_prompts.append(prompt)
        return _chat_response(f"take {model}")

    with (
        patch("llm_ladder.digest.discover_models", return_value=[_model("a"), _model("b")]),
        patch("llm_ladder.digest.free_loaded_models", return_value=None),
        patch("llm_ladder.digest.check_memory_safety", return_value=(True, "")),
        patch("llm_ladder.digest.chat", side_effect=fake_chat),
        patch("llm_ladder.digest.run_cascade", return_value=_cascade_result(judge_text)) as cascade,
    ):
        result = run_bias(str(doc), chain="default")

    assert result.lens.note is None
    assert result.lens.lens_verdict == (
        "2 of 2 models agree the piece leans sympathetic to the subject"
    )
    assert result.lens.consensus == ["both note the headline downplays the cost figure"]
    assert result.lens.disagreements == ["a calls the tone neutral, b calls it favorable"]
    assert cascade.call_count == 1
    # Fan-out prompt is framing-focused, not the changelog "breaking change" prompt.
    assert all("framing" in p.lower() for p in seen_prompts)
    assert all("breaking" not in p.lower() for p in seen_prompts)
    # Judge cascade is tagged "bias", not digest's "digest" tag.
    cascade_args = cascade.call_args
    assert cascade_args.args[1] == "bias" or cascade_args.kwargs.get("chain") == "bias"


def test_run_bias_explicit_models_skip_discovery(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("some article text")
    with (
        patch("llm_ladder.digest.discover_models") as mock_discover,
        patch("llm_ladder.digest.free_loaded_models", return_value=None),
        patch("llm_ladder.digest.check_memory_safety", return_value=(True, "")),
        patch("llm_ladder.digest.chat", side_effect=lambda m, p: _chat_response(f"take {m}")),
        patch("llm_ladder.digest.run_cascade", return_value=_cascade_result("VERDICT: v\nCONSENSUS:\nDISAGREEMENTS:\n")),
    ):
        result = run_bias(str(doc), models=["x", "y"])

    mock_discover.assert_not_called()
    assert [t.model for t in result.lens.takes] == ["x", "y"]


def test_run_bias_memory_skip_leaves_model_out_of_takes(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("some article text")

    def safety(size):
        return (False, "needs 8 GB, 4 GB free") if size > 50 else (True, "")

    with (
        patch(
            "llm_ladder.digest.discover_models",
            return_value=[_model("big", size_bytes=99), _model("small", size_bytes=10)],
        ),
        patch("llm_ladder.digest.free_loaded_models", return_value=None),
        patch("llm_ladder.digest.time.sleep", return_value=None),
        patch("llm_ladder.digest.check_memory_safety", side_effect=safety),
        patch("llm_ladder.digest.chat", side_effect=lambda m, p: _chat_response(f"take {m}")),
    ):
        result = run_bias(str(doc))

    assert result.lens.skipped == [("big", "needs 8 GB, 4 GB free")]
    assert [t.model for t in result.lens.takes] == ["small"]
