# llm-ladder

**llm-ladder** is a cascading confidence-gated local LLM router designed to optimize local inference costs. Instead of routing every prompt to a large, slow model, it attempts to resolve queries with smaller, faster models first. It only escalates to bigger, slower models when the smaller ones "disagree with themselves" (i.e., fall below a confidence threshold). This approach significantly cuts local inference cost and latency for high-volume or simple tasks, reserving heavy compute for complex edge cases.

![The cascade landing page and the ledger stats dashboard showing savings by tier](assets/llm-ladder-demo.gif)

## Installation

```bash
pip install -e .
```

### Running Tests

```bash
pip install pytest
pytest
```

All model calls are mocked — the suite doesn't need Ollama running.

## Prerequisites

1. [Ollama](https://ollama.com/) must be running locally.
2. The models in the default chain must be available in Ollama: `gemma4:e4b-mlx` (tier 1), `qwen3.8:27b-mlx` (tier 2), `ornith:35b-q4_K_M` (tier 3). Swap `chains.yaml` for whatever models you actually have pulled.

## Usage

### Running a Chain

Use the `run` command to process a prompt through the configured cascade.

```bash
# Basic usage (uses the "default" chain from chains.yaml)
ladder run "What is the capital of France?"

# Specify a chain by name (add more chains to chains.yaml, then reference them here)
ladder run "Explain quantum entanglement" --chain default

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

### Benchmark

```bash
ladder benchmark [--quick] [--models m1,m2] [--skip-gpu]
```

Benchmarks every Ollama model installed on your machine: speed (tokens/sec,
time-to-first-token), hardware usage (RAM, CPU, and — on macOS with sudo —
GPU power/utilization), and output quality across 8 categories (reasoning,
code including edge cases, JSON extraction, tool usage, instruction
following, factual recall, RAG/retrieval-grounded QA). A memory-safety
filter skips any model that would eat more than 80% of currently-available
RAM rather than risk crashing your machine. Results append to
`~/.llm-ladder/benchmark.jsonl` and rank models ordinally per category — see
`web/benchmark.html` for the leaderboard view (same zero-dependency,
paste-or-load-your-own-file pattern as the ledger dashboard).

"Estimated load bandwidth" is a derived figure (model size ÷
time-to-first-token), not a verified hardware measurement — labeled as such
everywhere it appears.

### Digest

```bash
ladder digest [PATH] [--releases N] [--range v1..v2] [--file doc.txt] \
  [--lens] [--models m1,m2] [--since-last] [--md out.md] [--json]
```

Summarizes a local git repo's recent history — the last N tags (map-reduce
over each release, then a roll-up cascade call), an explicit commit range, a
plain text file, or the last 50 commits when there are no tags — into a
one-line VERDICT plus per-release bullets, prefixing anything that reads as a
breaking change with `BREAKING:`. Add `--lens` to fan the same material out
to every Ollama model you have installed (auto-discovered, never hardcoded)
and get a judge pass reconciling where they genuinely disagree in
interpretation, not just wording. `--since-last` tracks the last digested
tag per repo in `~/.llm-ladder/digest_state.json`, so `ladder digest
--since-last --md digest.md` is cron-safe out of the box. RAM-managed the
same way `ladder benchmark` is — models are unloaded and memory-checked
before each lens call so a 30B model never stacks against the last one.

## Web

A self-contained, zero-dependency site in `web/` — no server needed:

- **`index.html`** — pitch page with a visual walkthrough of the confidence-gated escalation flow.
- **`digest.html`** — run `ladder digest` from the browser (requires `ladder serve`): repo path, release count, and an optional perspective lens, with live progress polling.
- **`stats.html`** — load your `~/.llm-ladder/ledger.jsonl` (file picker, drag-and-drop, or paste) and get a live dashboard: savings %, tier breakdown, recent runs, with chain/tier/model filters. Parsing happens entirely in your browser.
- **`benchmark.html`** — load your `~/.llm-ladder/benchmark.jsonl` and get a ranked leaderboard across speed and all 8 quality categories, filterable per category.

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
