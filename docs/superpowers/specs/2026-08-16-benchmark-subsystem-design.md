# llm-ladder benchmark subsystem — design

Date: 2026-08-16
Status: approved, pending implementation plan

## Motivation

llm-ladder currently answers "how do I cascade a prompt through local models
cheaply." It doesn't answer "which of my local models is actually good, at
what, and can my machine even run them." This adds a `ladder benchmark`
subsystem that measures both **hardware performance** (speed, memory
footprint, CPU/GPU usage) and **output quality** (correctness across a fixed
task suite) for every Ollama model installed on the machine running it, then
ranks them.

Two ideas from an earlier portfolio-project shortlist (LocalRAG-Bench,
RepoTriage Agent) were reconsidered against "these are all part of the same
local-model ecosystem, why are they separate repos." Verdict:

- **LocalRAG-Bench** folds in cleanly as one more benchmark category
  (RAG/retrieval-grounded QA is a quality test, same shape as the rest).
- **RepoTriage Agent** does NOT fold in — it's an application (triages a
  repo's issues using an LLM), not a quality test. Out of scope for this
  spec. Flagged as a candidate future `ladder triage <repo>` command built
  on the existing cascade engine (`engine.py`), to be designed separately.

## Architecture

New module `llm_ladder/benchmark.py`, plus two packaged data files:

- `llm_ladder/benchmark_tasks.yaml` — the fixed task suite (prompts +
  answer keys), one list per category.
- `llm_ladder/tool_registry.yaml` — curated list of local CLI tools the
  Tool Usage category can detect (name, PATH check command, install hint,
  one-line description).

### Model discovery

`discover_models()` shells out to `ollama list`, parses name + on-disk
size for every installed model. No hardcoded model list — this is what
makes the benchmark portable to whoever else runs it.

### Memory safety (the "don't crash my RAM" requirement)

`check_memory_safety(model_size_bytes) -> (ok: bool, reason: str)`:
- Reads `psutil.virtual_memory().available`.
- Skips (does not attempt to run) any model whose on-disk size would
  exceed 80% of currently-available memory — logs a clear reason
  ("needs ~19GB, 14GB available — skipped").
- Models run strictly sequentially, never concurrently.
- After each model finishes, the harness calls `ollama stop <model>`
  before starting the next, so loaded weights don't stack in memory
  across the run.

### Hardware telemetry

`capture_hardware_snapshot() -> HardwareSnapshot`:
- RAM total/available/used, CPU% — via `psutil` (new dependency, no
  sudo, cross-platform).
- GPU utilization/power — via `sudo powermetrics --samplers gpu_power`
  on macOS only. Requires a one-time sudo prompt per run. Skipped
  entirely (field is `null`, clearly labeled "not available") on
  non-macOS or when `--skip-gpu` is passed.
- **Bandwidth is an estimate, not a real measurement.** True memory
  bandwidth needs Instruments-level tooling this project won't build.
  Instead: `load_bandwidth_estimate_gbps = model_size_bytes /
  time_to_first_token`, labeled in both the JSONL and the UI as
  "estimated load bandwidth," never presented as a verified figure.

### Speed benchmark

`run_speed_benchmark(model)` sends a fixed medium-length prompt via the
existing `ollama_client.py`, reads `eval_count`/`eval_duration` directly
from Ollama's own response (no hand-rolled token counting), computes
tokens/sec and time-to-first-token.

### Quality benchmark — 8 categories

Each category is a short, fixed, reproducible task list in
`benchmark_tasks.yaml`, graded by exact-match or regex — no LLM-judge, no
network dependency except where noted. All categories run against every
model that survives the memory-safety filter.

1. **Speed** — objective, always measured (see above).
2. **Reasoning** — small math/logic word problems, exact numeric-answer
   match.
3. **Code** — function-generation prompts INCLUDING edge-case/boundary
   inputs (empty input, boundary values, malformed input), graded by
   actually running the returned code against fixed test inputs in a
   subprocess and checking output.
4. **JSON Extraction** — mirrors llm-ladder's real cascading use case:
   extract structured fields from unstructured text, graded by
   schema-valid JSON + exact field match.
5. **Tool Usage** — two subcategories:
   - *Schema correctness*: fixed fake tool schema, no live effects,
     checks the model emits a correctly-formatted tool call.
   - *Toolbelt scan*: `discover_installed_tools()` checks PATH against
     `tool_registry.yaml`. For each tool actually present, the model is
     given that tool's schema and one safe, read-only task (e.g. "get
     this file's duration" for ffmpeg). The harness validates the
     model's tool-call args against a strict per-tool allowlist and
     executes a hardcoded-safe subprocess — **the model's raw output is
     never interpolated into a shell command**. For tools NOT installed,
     the report includes an install recommendation (source link) if
     that tool would plausibly help the categories being tested. Models
     that don't support Ollama's tool-calling API are marked
     "unsupported," not scored zero.
6. **Instruction-following** — format-constrained prompts (e.g. "answer
   in exactly 3 words"), graded by pattern match.
7. **Factual recall** — small set of stable facts (dates, authorship,
   physical constants — nothing that changes), exact/fuzzy match.
8. **RAG / retrieval-grounded QA** — a small fixed corpus (a few short
   local text snippets bundled with the package) plus questions
   answerable only from that corpus. Tests both retrieval (did it pull
   the right snippet) and groundedness (did the answer stick to what
   the snippet says, graded by keyword/fact presence, not exact match).

One live-network exception: if a future Web Research task is added inside
Tool Usage, it would call a real `web_search` tool (DuckDuckGo HTML
endpoint, stdlib-only HTML-to-text, no API key). **Not included in this
initial spec** — the toolbelt-scan subcategory above covers "does the
model use tools well" without requiring live network calls, keeping the
whole benchmark reproducible offline. Web Research can be a follow-up
category once the offline suite is proven out.

### Ranking

Ordinal, per the user's explicit spec: within each category, the top
performer among the N models actually benchmarked scores N, second place
scores N-1, down to 1. Overall rank is the average of per-category ranks
(also displayed as an ordinal position). Recomputed client-side in the
web page so filtering by category re-ranks correctly.

### CLI surface

```
ladder benchmark [--quick] [--models m1,m2] [--skip-gpu]
```
- `--quick`: speed benchmark only, skips the 8-category quality suite
  (fast iteration / smoke test).
- `--models`: restrict to a comma-separated list instead of
  auto-discovering everything installed.
- `--skip-gpu`: no sudo prompt, RAM/CPU telemetry only.

Results append to `~/.llm-ladder/benchmark.jsonl` (separate file from
the existing confidence-cascade `ledger.jsonl`). Prints a `rich` summary
table at the end (rich is already a dependency).

### Web leaderboard

New `web/benchmark.html`, following the exact zero-dependency,
paste-your-own-JSONL pattern already established by `stats.html`
(nothing leaves the browser, `connect-src 'none'` CSP). Pre-populated
with a real demo run captured on the maintainer's own M3 Pro, committed
to the repo — matches `stats.html`'s existing demo-ledger convention, so
the page isn't empty on first view. Category filter dropdowns reuse the
Chain/Tier/Model filter pattern just shipped in `stats.html`. Ordinal
rank badges per category, plus overall rank.

## Testing

Pure logic, unit-tested with mocked inputs:
- `check_memory_safety` — mocked `psutil.virtual_memory()` values.
- Every category's grader function.
- Ordinal ranking computation.
- Toolbelt-scan PATH detection (mocked `shutil.which`).

Live-smoke only, not unit-tested: actual Ollama calls, `powermetrics`
shell-out, `ollama list` parsing, subprocess execution of allowlisted
tool commands.

## Open items / honest limitations

- GPU telemetry is macOS-only and requires sudo. No Linux/Windows GPU
  path in this version.
- "Bandwidth" is a derived estimate, not a measured value — labeled as
  such everywhere it appears.
- Not every locally-installed model supports Ollama tool-calling; those
  models get "unsupported" instead of a fabricated score.
- RepoTriage Agent explicitly deferred, not part of this spec.
- Web Research (live network tool-use) explicitly deferred to a future
  category, to keep this suite fully offline-reproducible.
