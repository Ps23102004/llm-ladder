from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from typing import Dict, List

import yaml


@dataclass
class TierConfig:
    model: str
    samples: int
    threshold: float


@dataclass
class ChainConfig:
    name: str
    tiers: List[TierConfig] = field(default_factory=list)


def default_chains_path() -> str:
    """
    Returns the path to the packaged chains.yaml file.
    Uses importlib.resources to locate the file within the package.
    """
    resource = importlib.resources.files(__package__).joinpath("chains.yaml")
    # In Python 3.9+, Traversable objects support __fspath__
    return str(resource)


def load_chains(path: str) -> Dict[str, ChainConfig]:
    """
    Loads chain configurations from a YAML file.
    
    Args:
        path: Path to the YAML file containing chain definitions.
        
    Returns:
        Dictionary mapping chain names to ChainConfig objects.

    Raises:
        ValueError: if the file cannot be read/parsed, or a tier omits "model".
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        raise ValueError(f"could not load chains file {path}: {exc}") from exc

    chains = {}
    if not data or "chains" not in data:
        return chains

    raw_chains = data["chains"]
    for name, tiers_data in raw_chains.items():
        if not isinstance(tiers_data, list):
            raise ValueError(
                f"malformed chains file {path}: chain '{name}' must be a list of tiers"
            )
        tiers = []
        for index, tier_data in enumerate(tiers_data):
            if not isinstance(tier_data, dict) or not tier_data.get("model"):
                raise ValueError(
                    f"malformed chains file {path}: chain '{name}' tier {index} "
                    "is missing required key 'model'"
                )
            samples = int(tier_data.get("samples", 1))
            if samples < 1:
                raise ValueError(
                    f"malformed chains file {path}: chain '{name}' tier {index} "
                    f"has samples={samples}, must be at least 1"
                )
            # Ensure fields are properly cast
            tiers.append(
                TierConfig(
                    model=str(tier_data["model"]),
                    samples=samples,
                    threshold=float(tier_data.get("threshold", 0.0)),
                )
            )

        chains[name] = ChainConfig(name=name, tiers=tiers)

    return chains
