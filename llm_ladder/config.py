from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
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
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    chains = {}
    if not data or "chains" not in data:
        return chains
        
    raw_chains = data["chains"]
    for name, tiers_data in raw_chains.items():
        tiers = []
        if isinstance(tiers_data, list):
            for tier_data in tiers_data:
                # Ensure fields are properly cast
                model = str(tier_data.get("model"))
                samples = int(tier_data.get("samples", 1))
                threshold = float(tier_data.get("threshold", 0.0))
                tiers.append(TierConfig(model=model, samples=samples, threshold=threshold))
        elif isinstance(tiers_data, dict):
            # Handle case where a chain might be a single tier dict, though schema implies list
            model = str(tiers_data.get("model"))
            samples = int(tiers_data.get("samples", 1))
            threshold = float(tiers_data.get("threshold", 0.0))
            tiers.append(TierConfig(model=model, samples=samples, threshold=threshold))
            
        chains[name] = ChainConfig(name=name, tiers=tiers)
        
    return chains
