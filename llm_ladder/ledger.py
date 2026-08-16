from __future__ import annotations

import json
import os
import time
from typing import Any


class Ledger:
    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = os.path.expanduser("~/.llm-ladder/ledger.jsonl")
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def record(
        self,
        chain: str,
        tier_index: int,
        model: str,
        confidence: float,
        samples: int,
        escalated: bool,
        used_last_tier: bool,
        duration_s: float,
    ) -> None:
        entry: dict[str, Any] = {
            "chain": chain,
            "tier_index": tier_index,
            "model": model,
            "confidence": confidence,
            "samples": samples,
            "escalated": escalated,
            "used_last_tier": used_last_tier,
            "duration_s": duration_s,
            "timestamp": time.time(),
        }
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            raise RuntimeError(f"could not write to ledger at {self.path}: {exc}") from exc

    def summary(self) -> dict:
        summary: dict[str, Any] = {
            "total_runs": 0,
            "runs_by_tier": {},
            "total_duration_s": 0.0,
            "estimated_savings_pct": 0.0,
            "skipped_malformed": 0,
        }

        if not os.path.exists(self.path):
            return summary

        entries: list[dict[str, Any]] = []
        skipped_malformed = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped_malformed += 1
                    continue

        summary["skipped_malformed"] = skipped_malformed
        total_runs = len(entries)
        summary["total_runs"] = total_runs

        if total_runs == 0:
            return summary

        runs_by_tier: dict[int, int] = {}
        total_duration_s = 0.0
        saved_runs = 0
        for entry in entries:
            tier_index = entry.get("tier_index", 0)
            runs_by_tier[tier_index] = runs_by_tier.get(tier_index, 0) + 1
            total_duration_s += float(entry.get("duration_s", 0.0))
            if entry.get("used_last_tier", True) is False:
                saved_runs += 1

        summary["runs_by_tier"] = runs_by_tier
        summary["total_duration_s"] = total_duration_s
        summary["estimated_savings_pct"] = saved_runs / total_runs * 100.0

        return summary
