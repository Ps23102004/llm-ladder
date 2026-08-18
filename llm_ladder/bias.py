"""`ladder bias` — run one article/policy doc through several local models
and surface where they genuinely disagree on framing.

Thin wrapper over digest's lens engine (`digest._run_lens_core`): the exact
same fan-out-to-every-installed-model + memory-safety-retry + cascade-judge
loop that `ladder digest --lens` already uses, pointed at framing-focused
prompts instead of changelog ones. No new LLM-calling layer, no new roster
logic — dogfoods the cascade engine that's already there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from llm_ladder.config import ChainConfig, default_chains_path, load_chains
from llm_ladder.digest import (
    LensResult,
    LensTake,
    ProgressFn,
    _answered_takes,
    _run_lens_core,
)
from llm_ladder.ledger import Ledger

BIAS_CHAIN_TAG = "bias"


class BiasError(ValueError):
    """Raised for bad options and unreadable/empty input files."""


@dataclass
class BiasResult:
    """One document's fan-out takes plus the judge's framing reconciliation."""

    source: str
    lens: LensResult
    generated_at: float


def _bias_lens_prompt(material: str) -> str:
    """Prompt handed verbatim to every model — same material, one voice each."""
    return (
        "Read the following article or document and give YOUR OWN take on "
        "its framing in 3-5 sentences: what it emphasizes, what it downplays "
        "or leaves out, and what tone or slant (if any) comes through in the "
        "word choices. Be specific and opinionated — this is a perspective "
        f"diff, not a summary.\n\n{material}"
    )


def _bias_judge_prompt(takes: list[LensTake]) -> str:
    """Prompt over the labeled takes; forces the same VERDICT/CONSENSUS/
    DISAGREEMENTS shape digest._parse_judge already knows how to parse."""
    answered = _answered_takes(takes)
    labeled = "\n\n".join(f"[{t.model}]: {t.take}" for t in answered)
    return (
        f"Here are {len(answered)} models' independent readings of the same "
        "document's framing. Compare them.\n\n"
        f"{labeled}\n\n"
        "Start with exactly one line: 'VERDICT: X of N models agree that "
        "<claim>'. Then a 'CONSENSUS:' heading with '-' bullets for what they "
        "agree on (emphasis, omissions, tone). Then a 'DISAGREEMENTS:' "
        "heading with '-' bullets naming which models are on each side of a "
        "real disagreement about framing, emphasis, or omission. Ignore mere "
        "wording differences — only report substantive disagreement in how "
        "the material is framed."
    )


def run_bias(
    path: str,
    models: list[str] | None = None,
    chain: str = "default",
    report_progress: ProgressFn = lambda p: None,
) -> BiasResult:
    """Read `path` as plain text and run the bias lens over it.

    `models` is None -> auto-discover every installed Ollama model (same as
    `ladder digest --lens`); a list -> use exactly those. `chain` picks which
    chains.yaml chain the judge pass cascades through.

    Raises BiasError for a URL (fetch is intentionally unsupported — it would
    need a new HTML-extraction dependency), a missing/unreadable/empty file,
    or an unknown chain. OllamaConnectionError propagates from the judge
    cascade call untouched, same as run_digest.
    """
    if path.lower().startswith(("http://", "https://")):
        raise BiasError(
            "URL fetch isn't supported (would need a new HTML-extraction "
            "dependency) — pass a local text file instead"
        )

    try:
        with open(path, encoding="utf-8") as f:
            material = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise BiasError(f"could not read {path}: {exc}") from exc
    if not material.strip():
        raise BiasError(f"{path} is empty, nothing to analyze")

    try:
        chains = load_chains(default_chains_path())
    except ValueError as exc:
        raise BiasError(f"chains.yaml is invalid: {exc}") from exc
    try:
        chain_config: ChainConfig = chains[chain]
    except KeyError:
        raise BiasError(f"chain '{chain}' not found")

    ledger = Ledger()
    lens_result = _run_lens_core(
        models,
        _bias_lens_prompt(material),
        _bias_judge_prompt,
        BIAS_CHAIN_TAG,
        chain_config,
        ledger,
        report_progress,
    )
    return BiasResult(source=path, lens=lens_result, generated_at=time.time())
