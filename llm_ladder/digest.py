"""`ladder digest` — summarize a repo's history, then show where models disagree.

One story, two passes over the SAME material:

1. Changelog digest (map-reduce): one ``run_cascade`` call per adjacent tag
   pair, then one roll-up cascade call that also emits the top-line VERDICT.
2. Bias lens (opt-in): every discovered model gets the same material via a
   direct ``ollama_client.chat`` call (raw per-model take is the signal, so
   no cascade here), then one cascade judge call reconciles them.

``run_digest`` is the single entry point; the CLI calls it synchronously and
the server wraps it in ``jobs.start_job`` — same function, same args.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from llm_ladder.benchmark_discovery import (
    check_memory_safety,
    discover_models,
    free_loaded_models,
)
from llm_ladder.config import ChainConfig, default_chains_path, load_chains
from llm_ladder.engine import run_cascade
from llm_ladder.ledger import Ledger
from llm_ladder.ollama_client import OllamaConnectionError, chat

STATE_PATH = os.path.expanduser("~/.llm-ladder/digest_state.json")

# `git log` output is capped so a 4000-commit range can't blow the context
# window of a tier-0 model.
MAX_SUBJECTS = 150

# Ledger `chain` tag for every model call this module makes, cascade or direct.
DIGEST_CHAIN_TAG = "digest"

ProgressFn = Callable[[dict], None]


class DigestError(ValueError):
    """Raised for bad options and for any failing git subprocess."""


# -- options / results ------------------------------------------------------


@dataclass
class DigestOptions:
    """Everything `ladder digest` needs; built by the CLI or the server route.

    `file_path` is doc-mode: digest a file's text instead of git history, and
    it is mutually exclusive with `range_`/`releases`/`since_last`.
    `models` is None -> discover the roster live; a list -> use exactly those.
    """

    repo: str = "."
    releases: int = 3
    range_: str | None = None
    file_path: str | None = None
    lens: bool = False
    models: list[str] | None = None
    chain: str = "default"
    since_last: bool = False
    md_path: str | None = None
    json_out: bool = False


@dataclass
class ReleaseSummary:
    """One tag-pair's bullets, plus which cascade tier produced them."""

    tag: str
    prev: str
    summary: str
    tier_index: int
    confidence: float


@dataclass
class LensTake:
    """One model's raw take, or the error string if its call failed.

    Exactly one of `take` / `error` is set.
    """

    model: str
    take: str | None = None
    error: str | None = None


@dataclass
class LensResult:
    """Fan-out takes plus the judge's reconciliation.

    `lens_verdict` is None when the judge was skipped (<2 models answered);
    in that case `note` explains why.
    """

    takes: list[LensTake] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (model, reason)
    consensus: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    lens_verdict: str | None = None
    note: str | None = None


@dataclass
class DigestResult:
    """The whole run. `lens` is None when --lens wasn't requested.

    `generated_at` is a unix float set INSIDE run_digest, matching how
    Ledger.record and run_benchmark stamp their own rows (this repo has no
    injected clock anywhere; the producer stamps, the caller doesn't). Tests
    that need determinism assert on the other fields or monkeypatch time.time.
    """

    verdict: str
    releases: list[ReleaseSummary]
    lens: LensResult | None
    repo: str
    generated_at: float


# -- git extraction ---------------------------------------------------------


def _git(path: str, args: list[str]) -> str:
    """Run `git -C <path> <args>` with timeout=15; raise DigestError on failure.

    Shared by the three wrappers below so the "git isn't installed / not a
    repo / bad range" failure mode is handled in exactly one place.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", path, *args],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        raise DigestError("git is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise DigestError(f"git {' '.join(args)} timed out after 15s")
    if proc.returncode != 0:
        raise DigestError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _tags(path: str) -> list[str]:
    """`git tag --sort=-creatordate` -> tag names, newest first, [] if none."""
    out = _git(path, ["tag", "--sort=-creatordate"])
    return [line for line in out.splitlines() if line.strip()]


def _subjects(path: str, range_: str) -> list[str]:
    """`git log --no-merges --pretty=%s <range>` -> subject lines.

    Capped at MAX_SUBJECTS entries; when truncated, a final
    "(and N more commits)" line is appended so the model knows it saw a slice.
    """
    out = _git(path, ["log", "--no-merges", "--pretty=%s", range_])
    lines = [line for line in out.splitlines() if line.strip()]
    if len(lines) <= MAX_SUBJECTS:
        return lines
    remainder = len(lines) - MAX_SUBJECTS
    return lines[:MAX_SUBJECTS] + [f"(and {remainder} more commits)"]


def _shortstat(path: str, range_: str) -> str:
    """`git diff --shortstat <range>` -> e.g. '12 files changed, 340 insertions(+)'."""
    out = _git(path, ["diff", "--shortstat", range_]).strip()
    return out or "no file changes recorded"


def _recent_range(path: str, limit: int = 50) -> str:
    """`<oldest commit within the last `limit`>..HEAD`, safe for short histories.

    `HEAD~N` errors out when a repo has fewer than N commits; walking the
    actual rev-list avoids guessing at commit count first.
    """
    out = _git(path, ["rev-list", f"--max-count={limit}", "HEAD"])
    shas = [line for line in out.splitlines() if line.strip()]
    if len(shas) < 2:
        # A single-commit (or empty) repo has no meaningful range to diff —
        # asking a model to summarize "no changes" just invites it to
        # hallucinate a verdict, so fail clearly instead.
        raise DigestError("repo has fewer than 2 commits, nothing to digest")
    oldest = shas[-1]
    return f"{oldest}..HEAD"


# -- --since-last state -----------------------------------------------------


def load_state() -> dict[str, str]:
    """Read ~/.llm-ladder/digest_state.json: {abs repo path: last digested tag}.

    Returns {} if the file is missing or malformed (same forgiving contract as
    model_config.load_config).
    """
    import json

    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_state(repo: str, tag: str) -> None:
    """Record `tag` as the last digested tag for abspath(repo).

    Atomic, copying model_config.save_config: dump to
    f"{STATE_PATH}.{os.getpid()}.{uuid4().hex[:8]}.tmp" then os.replace onto
    STATE_PATH, unlinking the temp file if the write fails.
    """
    import json

    state = load_state()
    state[os.path.abspath(repo)] = tag

    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp_path = f"{STATE_PATH}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_PATH)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# -- prompts ----------------------------------------------------------------


def _release_prompt(tag: str, prev: str, shortstat: str, subjects: list[str]) -> str:
    """Prompt for one tag pair: 2-4 plain-English bullets, no commit-speak.

    Must instruct: prefix any breaking change with "BREAKING:".
    """
    subject_block = "\n".join(f"- {s}" for s in subjects) or "(no commits)"
    return (
        f"You are writing release notes for users of this software.\n"
        f"Release {tag} (previous: {prev}). {shortstat}.\n\n"
        f"Commit subjects:\n{subject_block}\n\n"
        "Write 2-4 plain-English bullets: what changed and why it matters to a "
        "user. Prefix any bullet that looks like a breaking change with "
        "'BREAKING:'. No commit hashes, no git jargon. Do not invert the "
        "direction of a change — if a commit subject says X was replaced by "
        "Y, or switched to Z, say it that way round, not reversed. When "
        "the direction is ambiguous, quote the relevant commit subject "
        "directly rather than paraphrasing it."
    )


def _rollup_prompt(summaries: list[ReleaseSummary]) -> str:
    """Prompt over all per-release summaries.

    Must force the first output line to be exactly "VERDICT: <one sentence>".
    """
    sections = "\n\n".join(
        f"{s.tag} (since {s.prev}):\n{s.summary}" for s in summaries
    )
    return (
        "Combine these per-release summaries into one digest, newest first.\n"
        "Start with exactly one line: 'VERDICT: <one sentence — the single "
        "most important thing across these releases>'. Then a short section "
        "per release. Preserve each summary's stated direction of change "
        f"exactly — do not invert or restate it backwards.\n\n{sections}"
    )


def _single_chunk_prompt(material: str) -> str:
    """Prompt for the no-tags / --range / --file path: one chunk, one call.

    Same forced "VERDICT: <one sentence>" first line as the roll-up.
    """
    return (
        "Summarize the following into a plain-English digest.\n"
        "Start with exactly one line: 'VERDICT: <one sentence — the single "
        "most important thing here>'. Then 2-6 bullets of detail. Prefix any "
        "bullet that looks like a breaking change with 'BREAKING:'. Do not "
        "invert the direction of a change — if the material says X was "
        "replaced by Y, say it that way round, not reversed. When the "
        f"direction is ambiguous, quote the source text directly rather than "
        f"paraphrasing it.\n\n{material}"
    )


def _lens_prompt(material: str) -> str:
    """Prompt handed verbatim to every lens model — same material, one voice each."""
    return (
        "Read the following and give YOUR OWN interpretation in 3-5 sentences: "
        "the most significant change, who is affected, and the overall framing "
        f"(routine / risky / breaking / exciting). Be opinionated.\n\n{material}"
    )


def _judge_prompt(takes: list[LensTake]) -> str:
    """Prompt over the labeled takes, one "[model-name]: <take>" line each.

    Must force: "VERDICT: X of N models agree that <claim>", then a CONSENSUS
    bullet list, then DISAGREEMENT bullets naming which models are on each side.
    """
    answered = [t for t in takes if t.take is not None]
    labeled = "\n\n".join(f"[{t.model}]: {t.take}" for t in answered)
    return (
        f"Here are {len(answered)} models' independent interpretations of the "
        "same material. Compare them.\n\n"
        f"{labeled}\n\n"
        "Start with exactly one line: 'VERDICT: X of N models agree that "
        "<claim>'. Then a 'CONSENSUS:' heading with '-' bullets for what they "
        "agree on. Then a 'DISAGREEMENTS:' heading with '-' bullets naming "
        "which models are on each side of a real interpretive conflict. "
        "Ignore mere wording differences — only report substantive disagreement."
    )


def _strip_markdown_noise(line: str) -> str:
    """Drop '#'/'*' emphasis/heading characters models like to sprinkle on
    forced-format lines ("**VERDICT:**", "### CONSENSUS") before matching."""
    return line.replace("#", "").replace("*", "").strip()


def _split_verdict(text: str) -> tuple[str, str]:
    """Split a forced "VERDICT: ..." first line off the rest.

    Returns (verdict, remaining_body). If the model ignored the format, the
    verdict is "" and the whole text comes back as the body — never raise;
    a missing verdict line is cosmetic, not a failure. Tolerant of markdown
    emphasis around the label (models write "**VERDICT:**", "# VERDICT:", ...).
    """
    lines = text.strip().splitlines()
    if not lines:
        return "", ""
    first = _strip_markdown_noise(lines[0])
    if first.upper().startswith("VERDICT:"):
        verdict = first.split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
        return verdict, body
    return "", text.strip()


def _parse_judge(text: str) -> tuple[str, list[str], list[str]]:
    """Parse the judge reply -> (verdict, consensus bullets, disagreement bullets).

    Sections are found by the CONSENSUS / DISAGREEMENT headings (tolerant of
    markdown emphasis and inline content after the colon); bullets are the
    '-'/'*'-prefixed lines under each. Unparseable sections yield [].
    """
    verdict, body = _split_verdict(text)

    consensus_lines: list[str] = []
    disagreement_lines: list[str] = []
    section: str | None = None
    for line in body.splitlines():
        cleaned = _strip_markdown_noise(line)
        label, _, rest = cleaned.partition(":")
        heading = label.strip().upper()
        if heading in ("CONSENSUS", "DISAGREEMENTS", "DISAGREEMENT"):
            section = "consensus" if heading == "CONSENSUS" else "disagreements"
            inline = rest.strip()
            if inline:
                (consensus_lines if section == "consensus" else disagreement_lines).append(inline)
            continue
        stripped = line.strip()
        if section == "consensus" and stripped.startswith(("-", "*")):
            consensus_lines.append(stripped.lstrip("-*").strip())
        elif section == "disagreements" and stripped.startswith(("-", "*")):
            disagreement_lines.append(stripped.lstrip("-*").strip())

    return verdict, consensus_lines, disagreement_lines


# -- changelog digest -------------------------------------------------------


def _tag_pairs(tags: list[str], releases: int) -> list[tuple[str, str]]:
    """(older, newer) pairs for the last `releases` releases, newest-first input.

    Needs releases+1 tags to form `releases` pairs; uses however many are
    actually available.
    """
    usable = tags[: releases + 1]
    pairs = []
    for i in range(len(usable) - 1):
        newer, older = usable[i], usable[i + 1]
        pairs.append((older, newer))
    pairs.reverse()  # oldest pair first, so summaries read chronologically
    return pairs


def _digest_releases(
    opts: DigestOptions,
    chain_config: ChainConfig,
    ledger: Ledger,
    report_progress: ProgressFn,
    pairs: list[tuple[str, str]],
) -> tuple[str, list[ReleaseSummary]]:
    """Map-reduce over the given (older, newer) tag pairs.

    Map: one run_cascade(prompt, DIGEST_CHAIN_TAG, chain_config, ledger) per
    adjacent pair, reporting {"phase": "map", "tag", "index", "total"}.
    Reduce: one more cascade call on _rollup_prompt, reporting
    {"phase": "rollup"}, whose first line is split into the verdict.
    Returns (verdict, summaries).
    """
    summaries: list[ReleaseSummary] = []
    total = len(pairs)
    for index, (prev, tag) in enumerate(pairs, start=1):
        report_progress({"phase": "map", "tag": tag, "index": index, "total": total})
        range_ = f"{prev}..{tag}"
        shortstat = _shortstat(opts.repo, range_)
        subjects = _subjects(opts.repo, range_)
        prompt = _release_prompt(tag, prev, shortstat, subjects)
        result = run_cascade(prompt, DIGEST_CHAIN_TAG, chain_config, ledger)
        summaries.append(
            ReleaseSummary(
                tag=tag,
                prev=prev,
                summary=result.answer.strip(),
                tier_index=result.tier_index,
                confidence=result.confidence,
            )
        )

    report_progress({"phase": "rollup"})
    rollup_result = run_cascade(
        _rollup_prompt(summaries), DIGEST_CHAIN_TAG, chain_config, ledger
    )
    verdict, _body = _split_verdict(rollup_result.answer)
    return verdict, summaries


def _digest_single(
    material: str,
    label: str,
    opts: DigestOptions,
    chain_config: ChainConfig,
    ledger: Ledger,
    report_progress: ProgressFn,
) -> tuple[str, list[ReleaseSummary]]:
    """One chunk, one cascade call — used for --file, --range, and no-tags repos.

    Reports {"phase": "map", "tag": <range or file path>, "index": 1,
    "total": 1}; skips the roll-up step. Returns (verdict, [one ReleaseSummary]).
    """
    report_progress({"phase": "map", "tag": label, "index": 1, "total": 1})
    result = run_cascade(
        _single_chunk_prompt(material), DIGEST_CHAIN_TAG, chain_config, ledger
    )
    verdict, body = _split_verdict(result.answer)
    summary = ReleaseSummary(
        tag=label,
        prev="",
        summary=body,
        tier_index=result.tier_index,
        confidence=result.confidence,
    )
    return verdict, [summary]


# -- bias lens --------------------------------------------------------------


def _lens_roster(opts: DigestOptions) -> list[tuple[str, int]]:
    """Roster as [(model_name, size_bytes)].

    opts.models -> those names, size 0 (check_memory_safety becomes a no-op
    for them; the caller already told us exactly what to run, no need to shell
    out to `ollama list` to confirm it). Otherwise every model
    discover_models() returns. Never hardcodes a model list.

    discover_models() only runs when it's actually needed (opts.models is
    None); if it fails (ollama missing/hung), that failure is returned as an
    empty roster rather than raised — a --lens run shouldn't crash and lose
    an already-completed changelog digest over the lens step alone.
    """
    if opts.models:
        return [(name, 0) for name in opts.models]
    try:
        discovered = discover_models()
    except RuntimeError:
        return []
    return [(m.name, m.size_bytes) for m in discovered]


def _run_lens(
    material: str,
    opts: DigestOptions,
    chain_config: ChainConfig,
    ledger: Ledger,
    report_progress: ProgressFn,
) -> LensResult:
    """Fan the same material out to every roster model, then judge the takes.

    Per model, in order: free_loaded_models() -> check_memory_safety(size) ->
    if not ok, append (model, reason) to `skipped` and continue; else time and
    ollama_client.chat(model, prompt), append a LensTake, and ledger.record(
    chain=DIGEST_CHAIN_TAG, tier_index=0, model=model, confidence=1.0,
    samples=1, escalated=False, used_last_tier=True, duration_s=elapsed).
    A failing call records LensTake(model, error=str(exc)) and continues.
    Reports {"phase": "lens", "model", "index", "total"} before each call.

    Then: if fewer than 2 takes have a non-None `take`, skip the judge and set
    note="needs >=2 distinct models to compare"; else free_loaded_models()
    again (so the judge chain's tier-0 doesn't stack against the last lens
    model), report {"phase": "judge"}, and run one cascade over _judge_prompt.
    """
    roster = _lens_roster(opts)
    result = LensResult()
    prompt = _lens_prompt(material)
    total = len(roster)

    for index, (model, size_bytes) in enumerate(roster, start=1):
        report_progress({"phase": "lens", "model": model, "index": index, "total": total})
        free_loaded_models()
        ok, reason = check_memory_safety(size_bytes)
        if not ok:
            result.skipped.append((model, reason))
            continue
        start = time.perf_counter()
        try:
            resp = chat(model, prompt)
        except OllamaConnectionError as exc:
            result.takes.append(LensTake(model=model, error=str(exc)))
            continue
        elapsed = time.perf_counter() - start
        take_text = resp.get("message", {}).get("content", "").strip()
        result.takes.append(LensTake(model=model, take=take_text))
        ledger.record(
            chain=DIGEST_CHAIN_TAG,
            tier_index=0,
            model=model,
            confidence=1.0,
            samples=1,
            escalated=False,
            used_last_tier=True,
            duration_s=elapsed,
        )

    answered = [t for t in result.takes if t.take is not None]
    if len(answered) < 2:
        result.note = "needs >=2 distinct models to compare"
        return result

    free_loaded_models()
    report_progress({"phase": "judge"})
    judge_result = run_cascade(
        _judge_prompt(result.takes), DIGEST_CHAIN_TAG, chain_config, ledger
    )
    verdict, consensus, disagreements = _parse_judge(judge_result.answer)
    result.lens_verdict = verdict
    result.consensus = consensus
    result.disagreements = disagreements
    return result


# -- entry point ------------------------------------------------------------


def _validate(opts: DigestOptions) -> None:
    """Reject --file combined with --range/--releases/--since-last, and any
    non-positive --releases (0 digests nothing but still asks a model for a
    verdict; negative values slice the tag list backwards into a runaway
    digest of the whole history).

    Raises DigestError; `releases` counts as "given" only when it differs from
    the DigestOptions default.
    """
    if opts.releases < 1:
        raise DigestError(f"--releases must be at least 1, got {opts.releases}")
    if opts.file_path is None:
        return
    releases_given = opts.releases != DigestOptions().releases
    if opts.range_ is not None or releases_given or opts.since_last:
        raise DigestError(
            "--file is mutually exclusive with --range, --releases, and --since-last"
        )


def run_digest(opts: DigestOptions, report_progress: ProgressFn = lambda p: None) -> DigestResult:
    """Digest a repo (or a file) and optionally run the bias lens over it.

    The single entry point for both the CLI (called synchronously) and the
    server (wrapped in jobs.start_job). `report_progress` receives one of:
        {"phase": "map",    "tag": str, "index": int, "total": int}  # 1-based
        {"phase": "rollup"}
        {"phase": "lens",   "model": str, "index": int, "total": int}
        {"phase": "judge"}

    Flow: _validate -> load chains (config.load_chains(default_chains_path()),
    KeyError on an unknown opts.chain becomes a DigestError) -> build material
    (file text | opts.range_ | tag pairs; when opts.since_last, load_state()
    narrows the oldest bound to the stored tag and "nothing new since <tag>"
    is a DigestError) -> _digest_releases or _digest_single -> _run_lens when
    opts.lens -> save_state(repo, newest tag) on success, tag mode only ->
    stamp generated_at with time.time().

    Raises DigestError for bad options and git failures; OllamaConnectionError
    propagates from the cascade untouched (the CLI already maps it).
    """
    _validate(opts)

    try:
        chains = load_chains(default_chains_path())
    except ValueError as exc:
        raise DigestError(f"chains.yaml is invalid: {exc}")
    try:
        chain_config = chains[opts.chain]
    except KeyError:
        raise DigestError(f"chain '{opts.chain}' not found")

    ledger = Ledger()
    newest_tag_for_state: str | None = None

    if opts.file_path is not None:
        try:
            with open(opts.file_path, encoding="utf-8") as f:
                material = f.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise DigestError(f"could not read {opts.file_path}: {exc}")
        verdict, releases = _digest_single(
            material, opts.file_path, opts, chain_config, ledger, report_progress
        )
    elif opts.range_ is not None:
        material_parts = [
            _shortstat(opts.repo, opts.range_),
            "\n".join(_subjects(opts.repo, opts.range_)),
        ]
        verdict, releases = _digest_single(
            "\n\n".join(material_parts), opts.range_, opts, chain_config, ledger, report_progress
        )
    else:
        tags = _tags(opts.repo)
        releases_count = opts.releases

        if opts.since_last:
            state = load_state()
            last_tag = state.get(os.path.abspath(opts.repo))
            if last_tag is not None:
                if last_tag not in tags:
                    raise DigestError(
                        f"stored tag '{last_tag}' no longer exists in this repo"
                    )
                cutoff = tags.index(last_tag)
                if cutoff == 0:
                    raise DigestError(f"nothing new since {last_tag}")
                tags = tags[:cutoff] + [last_tag]
                # Digest every new tag, not just the newest `opts.releases` of
                # them — capping here would silently skip older new releases
                # while still recording the newest one as digested below.
                releases_count = len(tags) - 1

        if len(tags) >= 2:
            pairs = _tag_pairs(tags, releases_count)
            verdict, releases = _digest_releases(
                opts, chain_config, ledger, report_progress, pairs
            )
            newest_tag_for_state = tags[0]
        else:
            no_tag_range = _recent_range(opts.repo)
            material_parts = [
                _shortstat(opts.repo, no_tag_range),
                "\n".join(_subjects(opts.repo, no_tag_range)),
            ]
            verdict, releases = _digest_single(
                "\n\n".join(material_parts),
                no_tag_range,
                opts,
                chain_config,
                ledger,
                report_progress,
            )

    lens_result: LensResult | None = None
    if opts.lens:
        lens_material = "\n\n".join(
            f"{r.tag}: {r.summary}" for r in releases
        )
        lens_result = _run_lens(lens_material, opts, chain_config, ledger, report_progress)

    if newest_tag_for_state is not None:
        save_state(opts.repo, newest_tag_for_state)

    return DigestResult(
        verdict=verdict,
        releases=releases,
        lens=lens_result,
        repo=opts.repo,
        generated_at=time.time(),
    )


# -- export -----------------------------------------------------------------


def to_markdown(result: DigestResult) -> str:
    """Render a DigestResult as GitHub-flavored markdown.

    Layout: `## <verdict>`, then `### <tag> (since <prev>)` + body per release,
    then, when a lens ran, `## Model lens` with the lens verdict, a Consensus
    list, a Disagreements list naming models, and a trailing line for any
    skipped/errored models.
    """
    lines = [f"## {result.verdict or '(no verdict)'}", ""]

    for r in result.releases:
        heading = f"### {r.tag}" + (f" (since {r.prev})" if r.prev else "")
        lines.append(heading)
        lines.append(r.summary)
        lines.append("")

    if result.lens is not None:
        lens = result.lens
        lines.append("## Model lens")
        lines.append("")
        if lens.lens_verdict is not None:
            lines.append(f"**{lens.lens_verdict}**")
            lines.append("")
        if lens.note:
            lines.append(f"_{lens.note}_")
            lines.append("")
        if lens.consensus:
            lines.append("### Consensus")
            for c in lens.consensus:
                lines.append(f"- {c}")
            lines.append("")
        if lens.disagreements:
            lines.append("### Disagreements")
            for d in lens.disagreements:
                lines.append(f"- {d}")
            lines.append("")
        errored = [t for t in lens.takes if t.error is not None]
        if lens.skipped or errored:
            parts = [f"{m} ({reason})" for m, reason in lens.skipped]
            parts += [f"{t.model} (error: {t.error})" for t in errored]
            lines.append(f"_Skipped/failed: {', '.join(parts)}_")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
