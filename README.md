# llm-ladder

**llm-ladder** is a cascading confidence-gated local LLM router designed to optimize local inference costs. Instead of routing every prompt to a large, slow model, it attempts to resolve queries with smaller, faster models first. It only escalates to bigger, slower models when the smaller ones "disagree with themselves" (i.e., fall below a confidence threshold). This approach significantly cuts local inference cost and latency for high-volume or simple tasks, reserving heavy compute for complex edge cases.

## Installation

```bash
pip install -e .
```

## Prerequisites

1. [Ollama](https://ollama.com/) must be running locally.
2. The models in the default chain must be available in Ollama: `gemma4:e4b-mlx` (tier 1), `qwen3.8:27b-mlx` (tier 2), `ornith:35b-q4_K_M` (tier 3). Swap `chains.yaml` for whatever models you actually have pulled.

## Usage

### Running a Chain

Use the `run` command to process a prompt through the configured cascade.

```bash
# Basic usage (uses default chain)
ladder run "What is the capital of France?"

# Specify a custom chain
ladder run "Explain quantum entanglement" --chain advanced

# Get JSON output for programmatic processing
ladder run "Translate: Hello" --json
```

### Viewing Chains

List all available chains and their model tiers:

```bash
ladder chains
```

### Statistics

View aggregated performance statistics and estimated savings:

```bash
ladder stats
```

## Web

A self-contained, zero-dependency site in `web/` — no server needed:

- **`index.html`** — pitch page with a visual walkthrough of the confidence-gated escalation flow.
- **`stats.html`** — paste your `~/.llm-ladder/ledger.jsonl` contents and get a live dashboard: savings %, tier breakdown, recent runs. Parsing happens entirely in your browser.

## MCP

```bash
pip install -e ".[mcp]"
```

Exposes `ladder_run` and `ladder_chains` as MCP tools — attach the confidence-gated cascade to Claude Desktop, Claude Code, or any MCP client:

```json
{
  "mcpServers": {
    "llm-ladder": { "command": "ladder-mcp" }
  }
}
```

## Why This Exists

Local LLM inference involves a direct trade-off between model size and latency/cost. Large models (e.g., 70B parameters) provide higher accuracy but are extremely slow on consumer hardware. Smaller models (e.g., 7B parameters) are fast but prone to errors on complex tasks.

`llm-ladder` exploits the fact that most queries are "easy" and can be answered correctly by small models. By using self-consistency (sampling multiple answers and voting) as a confidence gate, we can dynamically select the smallest model that provides a confident answer. This reduces the average time-to-token and GPU/CPU utilization while maintaining acceptable accuracy for the majority of requests.
