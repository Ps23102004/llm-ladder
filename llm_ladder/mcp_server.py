from __future__ import annotations

import sys
import traceback

from mcp.server import MCPServer

from llm_ladder.config import load_chains, default_chains_path
from llm_ladder.engine import run_cascade
from llm_ladder.ollama_client import OllamaConnectionError

mcp = MCPServer("llm-ladder")


@mcp.tool()
def ladder_run(prompt: str, chain: str = "default") -> dict:
    """Run a prompt through llm-ladder's confidence-gated model cascade.

    Escalates from a fast local model to progressively larger ones only when
    self-consistency across samples is low, minimizing cost/latency for easy
    prompts while still trusting the largest tier's answer as a fallback.
    Requires a local Ollama instance with the chain's models pulled.
    """
    try:
        chains = load_chains(default_chains_path())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return {"error": str(exc)}

    if chain not in chains:
        return {"error": f"chain '{chain}' not found", "available_chains": list(chains.keys())}

    try:
        result = run_cascade(prompt, chain, chains[chain])
    except OllamaConnectionError as exc:
        return {"error": f"could not reach Ollama: {exc}"}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        # Last-resort fallback so a tool call never crashes the calling agent —
        # but log the full traceback so an unexpected bug is still visible in dev.
        traceback.print_exc(file=sys.stderr)
        return {"error": str(exc)}

    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "tier_index": result.tier_index,
        "model": result.model,
    }


@mcp.tool()
def ladder_chains() -> dict:
    """List the configured model chains and their tier order."""
    try:
        chains = load_chains(default_chains_path())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return {"error": str(exc)}
    return {
        name: [tier.model for tier in config.tiers]
        for name, config in chains.items()
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
