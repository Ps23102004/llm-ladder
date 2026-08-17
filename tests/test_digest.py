"""Tests for `llm_ladder.digest` — pure helpers, state round-trip, and the
mocked-model paths (run_cascade / chat / discovery are patched at their
import sites inside llm_ladder.digest; no live Ollama anywhere)."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import llm_ladder.digest as digest
from llm_ladder.config import ChainConfig, TierConfig
from llm_ladder.digest import (
    DigestError,
    DigestOptions,
    DigestResult,
    LensResult,
    LensTake,
    ReleaseSummary,
    _parse_judge,
    _recent_range,
    _split_verdict,
    _subjects,
    _tag_pairs,
    _validate,
    load_state,
    run_digest,
    save_state,
    to_markdown,
)
from llm_ladder.ledger import Ledger


def _chat_response(content: str) -> dict:
    return {"message": {"content": content}}


def _cascade_result(answer: str, tier_index: int = 0, confidence: float = 1.0):
    return SimpleNamespace(
        answer=answer, tier_index=tier_index, confidence=confidence, model="m"
    )


def _chain() -> ChainConfig:
    return ChainConfig(
        name="default", tiers=[TierConfig(model="m", samples=1, threshold=0.0)]
    )


def _model(name: str, size_bytes: int = 100):
    return SimpleNamespace(name=name, size_bytes=size_bytes)


# -- _git wrappers: subjects capping ----------------------------------------


def test_subjects_below_cap_returns_all_lines(monkeypatch):
    monkeypatch.setattr(digest, "_git", lambda path, args: "one\ntwo\n\nthree\n")
    assert _subjects(".", "v1..v2") == ["one", "two", "three"]


def test_subjects_exactly_at_cap_has_no_trailer(monkeypatch):
    out = "\n".join(f"c{i}" for i in range(digest.MAX_SUBJECTS))
    monkeypatch.setattr(digest, "_git", lambda path, args: out)
    subjects = _subjects(".", "v1..v2")
    assert len(subjects) == digest.MAX_SUBJECTS
    assert not any("more commits" in s for s in subjects)


def test_subjects_over_cap_appends_and_n_more_trailer(monkeypatch):
    total = digest.MAX_SUBJECTS + 50
    out = "\n".join(f"c{i}" for i in range(total))
    monkeypatch.setattr(digest, "_git", lambda path, args: out)
    subjects = _subjects(".", "v1..v2")
    assert len(subjects) == digest.MAX_SUBJECTS + 1
    assert subjects[-1] == "(and 50 more commits)"
    assert subjects[:-1] == [f"c{i}" for i in range(digest.MAX_SUBJECTS)]


def test_subjects_passes_range_to_git(monkeypatch):
    seen = {}

    def fake_git(path, args):
        seen["args"] = args
        return "only commit"

    monkeypatch.setattr(digest, "_git", fake_git)
    _subjects("/repo", "v1..v2")
    assert seen["args"] == ["log", "--no-merges", "--pretty=%s", "v1..v2"]


def test_git_timeout_raises_digest_error(monkeypatch):
    import subprocess

    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="git", timeout=15)

    monkeypatch.setattr(digest.subprocess, "run", boom)
    with pytest.raises(DigestError, match="timed out"):
        digest._git(".", ["tag"])


# -- _tag_pairs --------------------------------------------------------------


def test_tag_pairs_fewer_tags_than_releases_uses_what_exists():
    # 3 tags, releases=3: only 2 pairs form.
    assert _tag_pairs(["v3", "v2", "v1"], 3) == [("v1", "v2"), ("v2", "v3")]


def test_tag_pairs_full_count_oldest_first():
    pairs = _tag_pairs(["v4", "v3", "v2", "v1"], 3)
    assert pairs == [("v1", "v2"), ("v2", "v3"), ("v3", "v4")]


def test_tag_pairs_single_release_takes_newest_pair():
    assert _tag_pairs(["v3", "v2", "v1"], 1) == [("v2", "v3")]


def test_tag_pairs_one_or_zero_tags_yield_no_pairs():
    assert _tag_pairs(["v1"], 3) == []
    assert _tag_pairs([], 3) == []


def test_tag_pairs_is_capped_not_extended():
    # 5 tags but releases=2 -> only the 2 newest releases (3 newest tags) used.
    assert _tag_pairs(["v5", "v4", "v3", "v2", "v1"], 2) == [
        ("v3", "v4"),
        ("v4", "v5"),
    ]


# -- _split_verdict / _parse_judge -------------------------------------------


def test_split_verdict_extracts_first_line():
    verdict, body = _split_verdict("VERDICT: big release\n- bullet\n- bullet 2")
    assert verdict == "big release"
    assert body == "- bullet\n- bullet 2"


def test_split_verdict_without_verdict_line_returns_whole_text():
    text = "just a body\nsecond line"
    assert _split_verdict(text) == ("", text)


def test_split_verdict_empty_and_whitespace_only():
    assert _split_verdict("") == ("", "")
    assert _split_verdict("   \n  \n") == ("", "")


def test_split_verdict_is_case_insensitive_and_keeps_inner_colons():
    verdict, body = _split_verdict("verdict: shipped: at last\nbody")
    assert verdict == "shipped: at last"
    assert body == "body"


def test_parse_judge_full_shape():
    text = (
        "VERDICT: 2 of 3 models agree this is a breaking release\n"
        "CONSENSUS:\n"
        "- dropped the old API\n"
        "* auth rework\n"
        "DISAGREEMENTS:\n"
        "- a: routine, b/c: risky\n"
    )
    verdict, consensus, disagreements = _parse_judge(text)
    assert verdict == "2 of 3 models agree this is a breaking release"
    assert consensus == ["dropped the old API", "auth rework"]
    assert disagreements == ["a: routine, b/c: risky"]


def test_parse_judge_missing_sections_yield_empty_lists():
    verdict, consensus, disagreements = _parse_judge("VERDICT: only a verdict")
    assert verdict == "only a verdict"
    assert consensus == []
    assert disagreements == []


def test_parse_judge_bullets_before_any_heading_are_ignored():
    text = "VERDICT: v\n- stray\nCONSENSUS:\n- kept"
    _, consensus, _ = _parse_judge(text)
    assert consensus == ["kept"]


def test_parse_judge_accepts_heading_without_colon_and_singular():
    text = "VERDICT: v\nCONSENSUS\n- a\nDISAGREEMENT\n- b"
    _, consensus, disagreements = _parse_judge(text)
    assert consensus == ["a"]
    assert disagreements == ["b"]


# -- _validate ---------------------------------------------------------------


def test_validate_rejects_file_with_range():
    with pytest.raises(DigestError, match="--file is mutually exclusive"):
        _validate(DigestOptions(file_path="x.txt", range_="v1..v2"))


def test_validate_rejects_file_with_explicit_releases():
    with pytest.raises(DigestError, match="--file is mutually exclusive"):
        _validate(DigestOptions(file_path="x.txt", releases=5))


def test_validate_rejects_file_with_since_last():
    with pytest.raises(DigestError, match="--file is mutually exclusive"):
        _validate(DigestOptions(file_path="x.txt", since_last=True))


def test_validate_allows_file_with_default_releases():
    # releases=3 is the DigestOptions default, so it doesn't count as "given".
    assert DigestOptions(file_path="x.txt").releases == 3
    _validate(DigestOptions(file_path="x.txt"))


def test_validate_allows_file_alone_and_range_without_file():
    _validate(DigestOptions(file_path="x.txt", releases=3))
    _validate(DigestOptions(range_="v1..v2", releases=5, since_last=True))


# -- lens: degradation + memory-safety skip ---------------------------------


def _run_lens_with_roster(roster, tmp_path, safety=None, chat_side=None):
    """Run _run_lens with discovery/frees/memory-safety/model calls patched."""
    ledger = Ledger(path=str(tmp_path / "ledger.jsonl"))
    with (
        patch("llm_ladder.digest.discover_models", return_value=roster),
        patch("llm_ladder.digest.free_loaded_models", return_value=None),
        patch(
            "llm_ladder.digest.check_memory_safety",
            side_effect=safety or (lambda size: (True, "")),
        ),
        patch(
            "llm_ladder.digest.chat",
            side_effect=chat_side or (lambda model, prompt: _chat_response(f"take {model}")),
        ),
    ):
        result = digest._run_lens(
            "material",
            DigestOptions(lens=True),
            _chain(),
            ledger,
            lambda p: None,
        )
    return result


def test_run_lens_single_model_skips_judge_with_note(tmp_path):
    result = _run_lens_with_roster([_model("only")], tmp_path)
    assert result.takes == [LensTake(model="only", take="take only")]
    assert result.lens_verdict is None
    assert result.note == "needs >=2 distinct models to compare"
    assert result.consensus == [] and result.disagreements == []


def test_run_lens_memory_skip_leaves_model_out_of_takes(tmp_path):
    def safety(size):
        return (False, "needs 8 GB, 4 GB free") if size > 50 else (True, "")

    chats = []

    def fake_chat(model, prompt):
        chats.append(model)
        return _chat_response(f"take {model}")

    result = _run_lens_with_roster(
        [_model("big", size_bytes=99), _model("small", size_bytes=10)],
        tmp_path,
        safety=safety,
        chat_side=fake_chat,
    )
    assert result.skipped == [("big", "needs 8 GB, 4 GB free")]
    assert [t.model for t in result.takes] == ["small"]
    assert chats == ["small"]


def test_run_lens_degrades_when_memory_rejection_leaves_one_answer(tmp_path):
    # Two models discovered, one memory-rejected: <2 takes -> judge skipped.
    def safety(size):
        return (False, "too big") if size > 50 else (True, "")

    result = _run_lens_with_roster(
        [_model("huge", 100), _model("tiny", 10)], tmp_path, safety=safety
    )
    assert len(result.takes) == 1
    assert result.note == "needs >=2 distinct models to compare"


def test_run_lens_two_answers_run_judge(tmp_path):
    judge_text = (
        "VERDICT: 2 of 2 models agree it is a big release\n"
        "CONSENSUS:\n- large change\n"
        "DISAGREEMENTS:\n- a says routine, b says breaking\n"
    )
    with (
        patch("llm_ladder.digest.discover_models", return_value=[_model("a"), _model("b")]),
        patch("llm_ladder.digest.free_loaded_models", return_value=None),
        patch("llm_ladder.digest.check_memory_safety", return_value=(True, "")),
        patch("llm_ladder.digest.chat", side_effect=lambda m, p: _chat_response(f"take {m}")),
        patch(
            "llm_ladder.digest.run_cascade",
            return_value=_cascade_result(judge_text),
        ) as cascade,
    ):
        progress = []
        result = digest._run_lens(
            "material",
            DigestOptions(lens=True),
            _chain(),
            Ledger(path=str(tmp_path / "ledger.jsonl")),
            progress.append,
        )

    assert result.note is None
    assert result.lens_verdict == "2 of 2 models agree it is a big release"
    assert result.consensus == ["large change"]
    assert result.disagreements == ["a says routine, b says breaking"]
    # Exactly one cascade call (the judge), after 2 lens phases.
    assert cascade.call_count == 1
    assert [p["phase"] for p in progress] == ["lens", "lens", "judge"]


# -- state round-trip --------------------------------------------------------


def test_save_then_load_state_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "STATE_PATH", str(tmp_path / "digest_state.json"))
    save_state("/some/repo", "v3")
    save_state("/other/repo", "v1")
    state = load_state()
    assert state["/some/repo"] == "v3"
    assert state["/other/repo"] == "v1"


def test_load_state_missing_or_malformed_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(digest, "STATE_PATH", str(tmp_path / "missing.json"))
    assert load_state() == {}

    bad = tmp_path / "bad.json"
    bad.write_text("not json {")
    monkeypatch.setattr(digest, "STATE_PATH", str(bad))
    assert load_state() == {}

    non_dict = tmp_path / "list.json"
    non_dict.write_text('["a"]')
    monkeypatch.setattr(digest, "STATE_PATH", str(non_dict))
    assert load_state() == {}


# -- run_digest end-to-end (all model calls mocked) --------------------------


def test_run_digest_file_mode_single_chunk_no_lens(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("the material")
    with patch(
        "llm_ladder.digest.run_cascade",
        return_value=_cascade_result("VERDICT: the doc matters\n- detail"),
    ) as cascade:
        result = run_digest(DigestOptions(file_path=str(doc), md_path=None))

    assert cascade.call_count == 1
    assert result.verdict == "the doc matters"
    assert len(result.releases) == 1
    assert result.releases[0].tag == str(doc)
    assert result.releases[0].prev == ""
    assert result.releases[0].summary == "- detail"
    assert result.lens is None
    assert result.repo == "."


def test_run_digest_unknown_chain_raises_digest_error(tmp_path):
    with pytest.raises(DigestError, match="chain 'nope' not found"):
        run_digest(DigestOptions(chain="nope", range_="v1..v2"))


def test_run_digest_file_with_range_rejected_before_any_work(tmp_path):
    with pytest.raises(DigestError, match="--file is mutually exclusive"):
        run_digest(DigestOptions(file_path="x.txt", range_="v1..v2"))


# -- to_markdown -------------------------------------------------------------


def _digest_result(lens: LensResult | None) -> DigestResult:
    return DigestResult(
        verdict="a big release",
        releases=[
            ReleaseSummary(
                tag="v2", prev="v1", summary="added thing", tier_index=0, confidence=1.0
            ),
            ReleaseSummary(
                tag="v3", prev="v2", summary="broke thing", tier_index=1, confidence=0.5
            ),
        ],
        lens=lens,
        repo=".",
        generated_at=1.0,
    )


def test_to_markdown_without_lens():
    md = to_markdown(_digest_result(None))
    assert md.startswith("## a big release\n")
    assert "### v2 (since v1)\nadded thing" in md
    assert "### v3 (since v2)\nbroke thing" in md
    assert "Model lens" not in md
    assert md.endswith("\n")


def test_to_markdown_with_full_lens():
    lens = LensResult(
        takes=[LensTake(model="a", take="x"), LensTake(model="b", take="y")],
        skipped=[("big", "too large")],
        consensus=["it is big"],
        disagreements=["a: routine, b: breaking"],
        lens_verdict="2 of 2 models agree it is big",
        note=None,
    )
    md = to_markdown(_digest_result(lens))
    assert "## Model lens" in md
    assert "**2 of 2 models agree it is big**" in md
    assert "### Consensus\n- it is big" in md
    assert "### Disagreements\n- a: routine, b: breaking" in md
    assert "_Skipped/failed: big (too large)_" in md


def test_to_markdown_with_degraded_lens_and_no_verdict():
    lens = LensResult(
        takes=[LensTake(model="a", error="conn refused")],
        note="needs >=2 distinct models to compare",
    )
    md = to_markdown(_digest_result(lens))
    assert "## Model lens" in md
    assert "_needs >=2 distinct models to compare_" in md
    assert "Consensus" not in md
    assert "Disagreements" not in md
    assert "_Skipped/failed: a (error: conn refused)_" in md


def test_to_markdown_empty_verdict_gets_placeholder():
    result = DigestResult(
        verdict="", releases=[], lens=None, repo=".", generated_at=1.0
    )
    assert to_markdown(result).startswith("## (no verdict)")


# -- regression tests for the adversarial-review fixes -----------------------


def test_validate_rejects_zero_releases():
    with pytest.raises(DigestError, match="releases"):
        _validate(DigestOptions(releases=0))


def test_validate_rejects_negative_releases():
    with pytest.raises(DigestError, match="releases"):
        _validate(DigestOptions(releases=-5))


def test_run_digest_since_last_covers_all_new_tags_not_capped(tmp_path):
    monkeypatch_state = str(tmp_path / "digest_state.json")
    with patch.object(digest, "STATE_PATH", monkeypatch_state):
        save_state(".", "v1")
        with (
            patch("llm_ladder.digest._tags", return_value=["v5", "v4", "v3", "v2", "v1"]),
            patch("llm_ladder.digest._shortstat", return_value="1 file changed"),
            patch("llm_ladder.digest._subjects", return_value=["a commit"]),
            patch(
                "llm_ladder.digest.run_cascade",
                return_value=_cascade_result("VERDICT: v"),
            ) as cascade,
        ):
            result = run_digest(DigestOptions(repo=".", since_last=True))

        # 4 map calls (v1->v2, v2->v3, v3->v4, v4->v5) + 1 rollup, not capped
        # to the default releases=3.
        assert cascade.call_count == 5
        assert [(r.prev, r.tag) for r in result.releases] == [
            ("v1", "v2"), ("v2", "v3"), ("v3", "v4"), ("v4", "v5"),
        ]
        assert load_state()[os.path.abspath(".")] == "v5"


def test_run_lens_degrades_when_discover_models_raises(tmp_path):
    ledger = Ledger(path=str(tmp_path / "ledger.jsonl"))
    with patch("llm_ladder.digest.discover_models", side_effect=RuntimeError("ollama binary missing")):
        result = digest._run_lens(
            "material", DigestOptions(lens=True), _chain(), ledger, lambda p: None
        )
    assert result.note == "needs >=2 distinct models to compare"
    assert result.lens_verdict is None
    assert result.takes == []
    assert result.skipped == []


def test_lens_roster_explicit_models_skips_discovery():
    mock_discover = MagicMock()
    with patch("llm_ladder.digest.discover_models", mock_discover):
        roster = digest._lens_roster(DigestOptions(models=["a", "b"]))
    mock_discover.assert_not_called()
    assert roster == [("a", 0), ("b", 0)]


def test_run_digest_nonexistent_file_raises_digest_error(tmp_path):
    missing = tmp_path / "nope.md"
    with pytest.raises(DigestError, match="nope.md"):
        run_digest(DigestOptions(file_path=str(missing)))


def test_run_digest_invalid_chains_yaml_raises_digest_error():
    with patch("llm_ladder.digest.load_chains", side_effect=ValueError("bad yaml")):
        with pytest.raises(DigestError, match="chains.yaml"):
            run_digest(DigestOptions(range_="v1..v2"))


def test_recent_range_fewer_than_two_commits_raises(monkeypatch):
    monkeypatch.setattr(digest, "_git", lambda path, args: "onlyone\n")
    with pytest.raises(DigestError, match="fewer than 2 commits"):
        _recent_range(".")


def test_recent_range_returns_oldest_to_head(monkeypatch):
    monkeypatch.setattr(digest, "_git", lambda path, args: "new\nmid\nold\n")
    assert _recent_range(".") == "old..HEAD"


def test_split_verdict_tolerates_bold_emphasis():
    verdict, body = _split_verdict("**VERDICT:** shipped it\nbody")
    assert verdict == "shipped it"
    assert body == "body"


def test_parse_judge_tolerates_markdown_headings():
    text = "VERDICT: v\n**CONSENSUS:**\n- a\n### DISAGREEMENTS\n- b"
    _, consensus, disagreements = _parse_judge(text)
    assert consensus == ["a"]
    assert disagreements == ["b"]


def test_parse_judge_inline_heading_content_becomes_bullet():
    text = "VERDICT: v\nCONSENSUS: everyone agrees"
    _, consensus, _ = _parse_judge(text)
    assert consensus == ["everyone agrees"]
