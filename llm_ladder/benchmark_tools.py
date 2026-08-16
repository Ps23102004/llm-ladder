from __future__ import annotations

import importlib.resources
import shutil
import subprocess
from dataclasses import dataclass

import yaml


@dataclass
class ToolRegistryEntry:
    name: str
    check_command: list[str]
    install_hint: str
    description: str


def load_tool_registry() -> list[ToolRegistryEntry]:
    resource = importlib.resources.files("llm_ladder").joinpath("tool_registry.yaml")
    with resource.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        ToolRegistryEntry(
            name=item["name"],
            check_command=item["check_command"],
            install_hint=item["install_hint"],
            description=item["description"],
        )
        for item in data["tools"]
    ]


def discover_installed_tools(
    registry: list[ToolRegistryEntry],
) -> tuple[list[ToolRegistryEntry], list[ToolRegistryEntry]]:
    """Returns (installed, missing) by checking PATH for each tool's binary."""
    installed, missing = [], []
    for entry in registry:
        binary = entry.check_command[0]
        if shutil.which(binary):
            installed.append(entry)
        else:
            missing.append(entry)
    return installed, missing


def run_tool_probe(entry: ToolRegistryEntry) -> tuple[bool, str]:
    """Executes the tool's hardcoded, zero-argument check command. The
    command is entirely fixed by the registry — no model-controlled input
    ever reaches this subprocess call."""
    try:
        proc = subprocess.run(entry.check_command, capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout or proc.stderr).strip()[:200]
