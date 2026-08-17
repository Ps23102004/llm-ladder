from __future__ import annotations

import json
from dataclasses import dataclass

from llm_ladder.config import ChainConfig
from llm_ladder.engine import run_cascade
from llm_ladder.providers import ModelProvider


@dataclass
class TriageResult:
    issue_number: int
    category: str
    priority: str
    duplicate_of: int | None
    reasoning: str
    tier_index: int
    model: str
    confidence: float


def build_triage_prompt(issue, all_titles: list[str]) -> str:
    """Assemble the classification prompt for one issue.

    `issue` is duck-typed: anything with .number, .title and .body works.
    """
    lines = [
        "Triage this GitHub issue. Respond with ONLY a JSON object, no prose:",
        '{"category": "bug|feature|question|docs|other", "priority": "low|medium|high",'
        ' "duplicate_of": <issue number from the list below or null>, "reasoning": "<one sentence>"}',
        "",
        f"Issue #{issue.number}: {issue.title}",
        "",
        "Body:",
        issue.body or "(no body)",
        "",
        "Existing issues to check for duplicates:",
    ]
    # _title_numbers already emits '#N: title' for numbered entries; only
    # prefix '#' when the entry is a bare title with no number.
    lines.extend(
        entry if entry.lstrip().startswith("#") else f"#{entry}"
        for entry in _title_numbers(all_titles)
    )
    return "\n".join(lines)


def _title_numbers(all_titles: list[str]) -> list[str]:
    """Normalize all_titles entries into '#N: title' lines.

    Entries already in '#N: title' form pass through unchanged; bare titles
    are listed without a number so the model can still see them as context.
    """
    out: list[str] = []
    for title in all_titles:
        stripped = title.lstrip()
        if stripped.startswith("#"):
            out.append(stripped)
        else:
            out.append(title)
    return out


def _parse_triage_json(raw: str) -> dict:
    """Parse the model's reply as a JSON object, tolerating code fences.

    Raises json.JSONDecodeError on failure; the caller applies the fallback.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip an opening fence (with optional language tag) and closing fence.
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("expected a JSON object", text, 0)
    return data


def triage_issue(
    issue,
    all_titles: list[str],
    chain_name: str,
    chain_config: ChainConfig,
    provider: ModelProvider | None = None,
) -> TriageResult:
    """Classify one issue (category, priority, duplicate_of) with a model.

    When `provider` is given, a single tier-0 call is made at full confidence;
    otherwise the prompt runs through the cascade and inherits its
    tier_index/confidence/model.
    """
    prompt = build_triage_prompt(issue, all_titles)

    if provider is not None:
        raw = provider.chat(prompt, model=chain_config.tiers[0].model)
        tier_index = 0
        model = chain_config.tiers[0].model
        confidence = 1.0
    else:
        result = run_cascade(prompt, chain_name, chain_config)
        raw = result.answer
        tier_index = result.tier_index
        model = result.model
        confidence = result.confidence

    # Extract each field defensively: a missing/invalid value for one field
    # must not blank out the fields that parsed fine.
    try:
        data = _parse_triage_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    category = str(data["category"]) if data.get("category") is not None else "other"
    priority = str(data["priority"]) if data.get("priority") is not None else "medium"
    try:
        duplicate_of = data.get("duplicate_of")
        duplicate_of = int(duplicate_of) if duplicate_of is not None else None
    except (TypeError, ValueError):
        duplicate_of = None
    reasoning = str(data["reasoning"]) if data.get("reasoning") is not None else raw[:200]

    return TriageResult(
        issue_number=issue.number,
        category=category,
        priority=priority,
        duplicate_of=duplicate_of,
        reasoning=reasoning,
        tier_index=tier_index,
        model=model,
        confidence=confidence,
    )
