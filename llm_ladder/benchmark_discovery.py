from __future__ import annotations

import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class DiscoveredModel:
    name: str
    size_bytes: int


_UNIT_MULTIPLIERS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def _parse_size(size_str: str, unit: str) -> int | None:
    try:
        value = float(size_str)
    except ValueError:
        return None
    multiplier = _UNIT_MULTIPLIERS.get(unit.upper())
    if multiplier is None:
        return None
    return int(value * multiplier)


def _parse_ollama_list(output: str) -> list[DiscoveredModel]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    models = []
    for line in lines[1:]:  # skip header row
        parts = line.split()
        if len(parts) < 4:
            continue
        name, _id, size_str, unit = parts[0], parts[1], parts[2], parts[3]
        size_bytes = _parse_size(size_str, unit)
        if size_bytes is not None:
            models.append(DiscoveredModel(name=name, size_bytes=size_bytes))
    return models


def discover_models() -> list[DiscoveredModel]:
    """Shells out to `ollama list` and parses installed model name + size.
    No hardcoded model list — portable to whoever runs this."""
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"could not run 'ollama list': {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"'ollama list' failed: {proc.stderr.strip()}")
    return _parse_ollama_list(proc.stdout)


def check_memory_safety(model_size_bytes: int) -> tuple[bool, str]:
    """Returns (ok, reason). Skips any model whose on-disk size would exceed
    80% of currently-available memory."""
    available = psutil.virtual_memory().available
    limit = available * 0.8
    if model_size_bytes > limit:
        needed_gb = model_size_bytes / 1024**3
        available_gb = available / 1024**3
        return False, f"needs ~{needed_gb:.1f}GB, {available_gb:.1f}GB available — skipped"
    return True, "ok"


def stop_model(model: str) -> None:
    """Unloads a model from Ollama so its weights don't stack in memory
    across a benchmark run. Best-effort — never raises."""
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
