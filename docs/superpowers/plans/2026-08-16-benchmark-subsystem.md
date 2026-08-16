# Benchmark Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ladder benchmark` — measures hardware performance (speed, RAM/CPU/GPU) and output quality (8 fixed task categories) for every Ollama model installed on the machine, ranks them ordinally, and surfaces results via CLI table + a new `web/benchmark.html` leaderboard.

**Architecture:** Six new focused modules under `llm_ladder/` (graders, ranking, discovery+memory-safety, hardware telemetry, tool registry, orchestrator), three new packaged YAML data files (task suite, RAG corpus, tool registry), one new CLI command, one new static HTML page following `stats.html`'s exact zero-dependency file-picker pattern.

**Tech Stack:** Python 3.9+, `psutil` (new dependency — RAM/CPU, no sudo), macOS `powermetrics` via subprocess (GPU, sudo, best-effort), existing `ollama_client.chat()`, `pyyaml`, `typer`/`rich` (already deps), plain JS for the web page (no framework, matches house style).

**Spec:** `docs/superpowers/specs/2026-08-16-benchmark-subsystem-design.md` — this plan implements it task-by-task; read both.

## Global Constraints

- No new dependency except `psutil` — everything else (YAML task data, grading) is stdlib + already-installed packages.
- Grading is exact-match/regex only — no LLM-judge, per spec.
- The model's raw output is NEVER interpolated into a shell command anywhere (tool-usage probes execute hardcoded, zero-argument commands only — see Task 5).
- GPU telemetry is macOS-only, requires sudo, and must degrade to `None`/"not available" everywhere else — never raise.
- "Load bandwidth" is a derived estimate; label it as such in every place it's surfaced (JSONL field name, CLI, web page).
- Ordinal ranking: within a category, best of N scores N, down to 1. Ties broken by first-appearing input order (same convention as `confidence.py`'s existing tie-break).
- One live-network exception carve-out from the spec (Web Research) is explicitly NOT built in this plan — Tool Usage's toolbelt-scan subcategory covers "does the model use tools well" fully offline (see Task 5's design note on why no live network call is needed at all, including for the schema-only subcategory).
- Design simplification (documented here so it's not a silent deviation): Tool Usage is graded via **prompted JSON emission**, not Ollama's native `tools` API parameter. The model is asked in plain text to respond with a JSON object shaped like a tool call; the harness parses and grades that JSON. This works uniformly across every model regardless of whether Ollama's chat template has "official" tool-calling support for it, so the spec's "models that don't support tool-calling are marked unsupported" case doesn't arise — every model can attempt a text instruction. Grading stays exact-structure-match, per spec.
- RepoTriage Agent and native Web Research: out of scope, per spec.

---

### Task 1: Grading functions + task-suite data (7 of 8 categories' pure logic)

**Files:**
- Create: `llm_ladder/benchmark_tasks.yaml`
- Create: `llm_ladder/rag_corpus.yaml`
- Create: `llm_ladder/benchmark_graders.py`
- Test: `tests/test_benchmark_graders.py`

**Interfaces:**
- Produces: `GradeResult(passed: bool, detail: str)` dataclass; `grade_reasoning(answer, expected) -> GradeResult`; `grade_code(answer, test_input, expected_output) -> GradeResult`; `grade_json_extraction(answer, expected_fields: dict) -> GradeResult`; `grade_tool_schema(answer, expected_tool: str, expected_args_keys: list[str]) -> GradeResult`; `grade_instruction_following(answer, pattern: str) -> GradeResult`; `grade_factual_recall(answer, accepted: list[str]) -> GradeResult`; `grade_rag(answer, required_keywords: list[str]) -> GradeResult`; `load_benchmark_tasks() -> dict`; `load_rag_corpus() -> dict`.

- [ ] **Step 1: Create the task-suite data file**

`llm_ladder/benchmark_tasks.yaml`:
```yaml
reasoning:
  - prompt: "If a train travels 60 miles in 1.5 hours, what is its speed in miles per hour? Answer with just the number."
    expected: "40"
  - prompt: "What is 17 + 25? Answer with just the number."
    expected: "42"
code:
  - prompt: "Write a Python function `add(a, b)` that returns the sum of two numbers. Only output the code in a ```python code block, no explanation."
    test_input: "add(2, 3)"
    expected_output: "5"
  - prompt: "Write a Python function `safe_divide(a, b)` that returns a/b, or None if b is 0 (do not raise on division by zero). Only output the code in a ```python code block, no explanation."
    test_input: "safe_divide(10, 0)"
    expected_output: "None"
json_extraction:
  - prompt: "Extract the company and role from this text as a JSON object with keys \"company\" and \"role\": 'Jane Smith works as a Senior Engineer at Acme Corp.' Only output the JSON."
    expected_fields:
      company: "Acme Corp"
      role: "Senior Engineer"
tool_schema:
  - prompt: "You have access to a tool called get_weather(city: string). Call this tool to get the weather in Paris. Respond ONLY with a JSON object in the form {\"tool\": \"get_weather\", \"args\": {\"city\": \"...\"}}."
    expected_tool: "get_weather"
    expected_args_keys: ["city"]
instruction_following:
  - prompt: "Answer in exactly 3 words: what color is the sky on a clear day?"
    pattern: '^(\S+\s+){2}\S+$'
factual_recall:
  - prompt: "In what year did the Eiffel Tower open to the public? Answer with just the year."
    accepted: ["1889"]
  - prompt: "Who wrote the play 'Romeo and Juliet'? Answer with just the name."
    accepted: ["William Shakespeare", "Shakespeare"]
```

- [ ] **Step 2: Create the RAG corpus data file**

`llm_ladder/rag_corpus.yaml`:
```yaml
corpus:
  - id: doc1
    text: "The llm-ladder confidence gate uses self-consistency voting: it samples a model N times and checks whether the majority of samples agree after normalization."
  - id: doc2
    text: "BEA Regional Price Parity data measures cost of living by US state, with 100 representing the national average purchasing power."
questions:
  - prompt: "What technique does llm-ladder's confidence gate use to decide whether to trust an answer?"
    context_id: doc1
    required_keywords: ["self-consistency", "voting"]
  - prompt: "What number represents the national average purchasing power in BEA Regional Price Parity data?"
    context_id: doc2
    required_keywords: ["100"]
```

- [ ] **Step 3: Write the failing tests**

`tests/test_benchmark_graders.py`:
```python
from llm_ladder.benchmark_graders import (
    grade_reasoning, grade_code, grade_json_extraction, grade_tool_schema,
    grade_instruction_following, grade_factual_recall, grade_rag,
    load_benchmark_tasks, load_rag_corpus,
)


def test_grade_reasoning_correct_number():
    result = grade_reasoning("The speed is 40 mph.", "40")
    assert result.passed

def test_grade_reasoning_wrong_number():
    result = grade_reasoning("The speed is 35 mph.", "40")
    assert not result.passed

def test_grade_reasoning_no_number_fails_cleanly():
    result = grade_reasoning("I don't know.", "40")
    assert not result.passed

def test_grade_code_correct_function():
    answer = "```python\ndef add(a, b):\n    return a + b\n```"
    result = grade_code(answer, "add(2, 3)", "5")
    assert result.passed

def test_grade_code_wrong_output():
    answer = "```python\ndef add(a, b):\n    return a - b\n```"
    result = grade_code(answer, "add(2, 3)", "5")
    assert not result.passed

def test_grade_code_no_code_block_fails_cleanly():
    result = grade_code("I cannot write code.", "add(2, 3)", "5")
    assert not result.passed

def test_grade_code_edge_case_division_by_zero():
    answer = "```python\ndef safe_divide(a, b):\n    return a / b if b != 0 else None\n```"
    result = grade_code(answer, "safe_divide(10, 0)", "None")
    assert result.passed

def test_grade_json_extraction_correct_fields():
    answer = '{"company": "Acme Corp", "role": "Senior Engineer"}'
    result = grade_json_extraction(answer, {"company": "Acme Corp", "role": "Senior Engineer"})
    assert result.passed

def test_grade_json_extraction_missing_field():
    answer = '{"company": "Acme Corp"}'
    result = grade_json_extraction(answer, {"company": "Acme Corp", "role": "Senior Engineer"})
    assert not result.passed

def test_grade_json_extraction_no_json_fails_cleanly():
    result = grade_json_extraction("I found no company.", {"company": "Acme Corp"})
    assert not result.passed

def test_grade_tool_schema_correct_call():
    answer = '{"tool": "get_weather", "args": {"city": "Paris"}}'
    result = grade_tool_schema(answer, "get_weather", ["city"])
    assert result.passed

def test_grade_tool_schema_wrong_tool_name():
    answer = '{"tool": "get_time", "args": {"city": "Paris"}}'
    result = grade_tool_schema(answer, "get_weather", ["city"])
    assert not result.passed

def test_grade_tool_schema_missing_args():
    answer = '{"tool": "get_weather", "args": {}}'
    result = grade_tool_schema(answer, "get_weather", ["city"])
    assert not result.passed

def test_grade_instruction_following_matches_pattern():
    result = grade_instruction_following("clear blue sky", r'^(\S+\s+){2}\S+$')
    assert result.passed

def test_grade_instruction_following_wrong_word_count():
    result = grade_instruction_following("blue", r'^(\S+\s+){2}\S+$')
    assert not result.passed

def test_grade_factual_recall_exact_match():
    result = grade_factual_recall("1889", ["1889"])
    assert result.passed

def test_grade_factual_recall_accepts_any_listed_answer():
    result = grade_factual_recall("It was written by Shakespeare.", ["William Shakespeare", "Shakespeare"])
    assert result.passed

def test_grade_factual_recall_wrong_answer():
    result = grade_factual_recall("1900", ["1889"])
    assert not result.passed

def test_grade_rag_has_required_keywords():
    result = grade_rag("It uses self-consistency voting across samples.", ["self-consistency", "voting"])
    assert result.passed

def test_grade_rag_missing_keyword():
    result = grade_rag("It uses some kind of voting.", ["self-consistency", "voting"])
    assert not result.passed

def test_load_benchmark_tasks_has_all_categories():
    tasks = load_benchmark_tasks()
    for category in ["reasoning", "code", "json_extraction", "tool_schema", "instruction_following", "factual_recall"]:
        assert category in tasks
        assert len(tasks[category]) > 0

def test_load_rag_corpus_has_corpus_and_questions():
    rag = load_rag_corpus()
    assert len(rag["corpus"]) > 0
    assert len(rag["questions"]) > 0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_graders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_ladder.benchmark_graders'`

- [ ] **Step 5: Implement the graders module**

`llm_ladder/benchmark_graders.py`:
```python
from __future__ import annotations

import importlib.resources
import json
import re
import subprocess
from dataclasses import dataclass

import yaml


@dataclass
class GradeResult:
    passed: bool
    detail: str


def load_benchmark_tasks() -> dict:
    resource = importlib.resources.files("llm_ladder").joinpath("benchmark_tasks.yaml")
    with resource.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rag_corpus() -> dict:
    resource = importlib.resources.files("llm_ladder").joinpath("rag_corpus.yaml")
    with resource.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    if "def " in text:
        return text
    return None


def grade_reasoning(answer: str, expected: str) -> GradeResult:
    match = re.search(r"-?\d+(\.\d+)?", answer)
    if not match:
        return GradeResult(False, f"no number found in answer: {answer!r}")
    got = match.group(0)
    passed = got.strip() == expected.strip()
    return GradeResult(passed, f"expected {expected}, got {got}")


def grade_code(answer: str, test_input: str, expected_output: str) -> GradeResult:
    code = _extract_code_block(answer)
    if code is None:
        return GradeResult(False, "no code block found in answer")
    script = f"{code}\nprint({test_input})"
    try:
        proc = subprocess.run(["python3", "-c", script], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return GradeResult(False, "code execution timed out")
    if proc.returncode != 0:
        return GradeResult(False, f"code raised an error: {proc.stderr.strip()[:200]}")
    got = proc.stdout.strip()
    passed = got == expected_output.strip()
    return GradeResult(passed, f"expected {expected_output!r}, got {got!r}")


def grade_json_extraction(answer: str, expected_fields: dict) -> GradeResult:
    obj = _extract_json(answer)
    if obj is None:
        return GradeResult(False, "no valid JSON object found in answer")
    missing = [k for k in expected_fields if k not in obj]
    if missing:
        return GradeResult(False, f"missing fields: {missing}")
    mismatched = {
        k: obj[k] for k in expected_fields
        if str(obj[k]).strip().lower() != str(expected_fields[k]).strip().lower()
    }
    if mismatched:
        return GradeResult(False, f"field mismatch: {mismatched}")
    return GradeResult(True, "all fields matched")


def grade_tool_schema(answer: str, expected_tool: str, expected_args_keys: list[str]) -> GradeResult:
    obj = _extract_json(answer)
    if obj is None:
        return GradeResult(False, "no valid JSON tool call found")
    if obj.get("tool") != expected_tool:
        return GradeResult(False, f"expected tool '{expected_tool}', got {obj.get('tool')!r}")
    args = obj.get("args", {})
    if not isinstance(args, dict):
        return GradeResult(False, "'args' is not an object")
    missing = [k for k in expected_args_keys if k not in args]
    if missing:
        return GradeResult(False, f"missing args: {missing}")
    return GradeResult(True, "tool call well-formed")


def grade_instruction_following(answer: str, pattern: str) -> GradeResult:
    passed = re.match(pattern, answer.strip()) is not None
    return GradeResult(passed, f"pattern {pattern!r} {'matched' if passed else 'did not match'} {answer.strip()!r}")


def grade_factual_recall(answer: str, accepted: list[str]) -> GradeResult:
    normalized = answer.strip().lower()
    passed = any(acc.strip().lower() in normalized for acc in accepted)
    return GradeResult(passed, f"expected one of {accepted}, got {answer.strip()!r}")


def grade_rag(answer: str, required_keywords: list[str]) -> GradeResult:
    normalized = answer.lower()
    missing = [kw for kw in required_keywords if kw.lower() not in normalized]
    passed = not missing
    return GradeResult(passed, f"missing keywords: {missing}" if missing else "all keywords present")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_graders.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
cd ~/Developer/llm-ladder
git add llm_ladder/benchmark_graders.py llm_ladder/benchmark_tasks.yaml llm_ladder/rag_corpus.yaml tests/test_benchmark_graders.py
git commit -m "Add benchmark task-suite data and grading functions for 7 of 8 categories"
```

---

### Task 2: Ordinal ranking

**Files:**
- Create: `llm_ladder/benchmark_rank.py`
- Test: `tests/test_benchmark_rank.py`

**Interfaces:**
- Produces: `RankedEntry(model: str, score: float, rank: int)` dataclass; `rank_models(scores: dict[str, float]) -> list[RankedEntry]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark_rank.py`:
```python
from llm_ladder.benchmark_rank import rank_models


def test_top_scorer_gets_rank_n():
    ranked = rank_models({"a": 0.9, "b": 0.5, "c": 0.7})
    by_model = {r.model: r.rank for r in ranked}
    assert by_model["a"] == 3
    assert by_model["c"] == 2
    assert by_model["b"] == 1

def test_empty_scores_returns_empty_list():
    assert rank_models({}) == []

def test_single_model_gets_rank_one():
    ranked = rank_models({"solo": 0.5})
    assert ranked[0].rank == 1

def test_ties_broken_by_input_order():
    ranked = rank_models({"a": 0.5, "b": 0.5})
    by_model = {r.model: r.rank for r in ranked}
    assert by_model["a"] == 2
    assert by_model["b"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_rank.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ranking**

`llm_ladder/benchmark_rank.py`:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankedEntry:
    model: str
    score: float
    rank: int


def rank_models(scores: dict[str, float]) -> list[RankedEntry]:
    """Ordinal rank: best score gets rank N (N = number of models), worst
    gets rank 1. Ties are broken by input (insertion) order — the
    first-inserted of a tied pair gets the higher rank, matching
    confidence.py's existing "first appearing wins" tie-break convention.
    """
    n = len(scores)
    if n == 0:
        return []
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [RankedEntry(model=model, score=score, rank=n - i) for i, (model, score) in enumerate(ordered)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_rank.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/llm-ladder
git add llm_ladder/benchmark_rank.py tests/test_benchmark_rank.py
git commit -m "Add ordinal ranking for benchmark results"
```

---

### Task 3: Model discovery + memory safety

**Files:**
- Create: `llm_ladder/benchmark_discovery.py`
- Test: `tests/test_benchmark_discovery.py`
- Modify: `pyproject.toml` (add `psutil` dependency)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `DiscoveredModel(name: str, size_bytes: int)` dataclass; `discover_models() -> list[DiscoveredModel]`; `check_memory_safety(model_size_bytes: int) -> tuple[bool, str]`; `stop_model(model: str) -> None`.

- [ ] **Step 1: Add psutil dependency**

In `pyproject.toml`, change:
```toml
dependencies = [
    "typer",
    "rich",
    "requests",
    "pyyaml"
]
```
to:
```toml
dependencies = [
    "typer",
    "rich",
    "requests",
    "pyyaml",
    "psutil"
]
```

Run: `cd ~/Developer/llm-ladder && .venv/bin/pip install -e .`
Expected: installs psutil into the venv, no errors.

- [ ] **Step 2: Write the failing tests**

`tests/test_benchmark_discovery.py`:
```python
from unittest.mock import patch, MagicMock

from llm_ladder.benchmark_discovery import (
    discover_models, check_memory_safety, _parse_ollama_list, DiscoveredModel,
)

# Real captured output from `ollama list` — columns are fixed-position
# (NAME, ID, SIZE_NUM, SIZE_UNIT), MODIFIED is variable-length trailing text.
SAMPLE_OLLAMA_LIST_OUTPUT = (
    "NAME                    ID              SIZE      MODIFIED     \n"
    "gemma4:12b-mlx          117d0d84cf2a    7.7 GB    27 hours ago    \n"
    "muse-glimmer:30b-mlx    ef32a55b4976    21 GB     37 hours ago    \n"
    "gemma4:e4b-mlx          aa6f2058d5dc    8.8 GB    40 hours ago    \n"
)


def test_parse_ollama_list_extracts_name_and_size():
    models = _parse_ollama_list(SAMPLE_OLLAMA_LIST_OUTPUT)
    assert len(models) == 3
    assert models[0].name == "gemma4:12b-mlx"
    assert abs(models[0].size_bytes - int(7.7 * 1024**3)) < 1024**2

def test_parse_ollama_list_empty_output():
    assert _parse_ollama_list("") == []
    assert _parse_ollama_list("NAME  ID  SIZE  MODIFIED\n") == []

def test_discover_models_shells_out_to_ollama_list():
    fake_proc = MagicMock(stdout=SAMPLE_OLLAMA_LIST_OUTPUT, returncode=0)
    with patch("llm_ladder.benchmark_discovery.subprocess.run", return_value=fake_proc):
        models = discover_models()
    assert len(models) == 3

def test_discover_models_raises_clean_error_if_ollama_missing():
    with patch("llm_ladder.benchmark_discovery.subprocess.run", side_effect=FileNotFoundError()):
        try:
            discover_models()
            assert False, "should have raised"
        except RuntimeError as e:
            assert "ollama list" in str(e)

def test_check_memory_safety_allows_small_model(monkeypatch):
    fake_vm = type("VM", (), {"available": 20 * 1024**3})()
    monkeypatch.setattr("llm_ladder.benchmark_discovery.psutil.virtual_memory", lambda: fake_vm)
    ok, reason = check_memory_safety(5 * 1024**3)
    assert ok

def test_check_memory_safety_skips_oversized_model(monkeypatch):
    fake_vm = type("VM", (), {"available": 10 * 1024**3})()
    monkeypatch.setattr("llm_ladder.benchmark_discovery.psutil.virtual_memory", lambda: fake_vm)
    ok, reason = check_memory_safety(9 * 1024**3)
    assert not ok
    assert "GB" in reason

def test_check_memory_safety_boundary_exactly_80_percent(monkeypatch):
    available = 10 * 1024**3
    fake_vm = type("VM", (), {"available": available})()
    monkeypatch.setattr("llm_ladder.benchmark_discovery.psutil.virtual_memory", lambda: fake_vm)
    ok, _ = check_memory_safety(int(available * 0.8))
    assert ok
    ok, _ = check_memory_safety(int(available * 0.8) + 1)
    assert not ok
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement discovery + memory safety**

`llm_ladder/benchmark_discovery.py`:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_discovery.py -v`
Expected: all PASS

- [ ] **Step 6: Live-smoke verify against real `ollama list`**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -c "from llm_ladder.benchmark_discovery import discover_models; [print(m) for m in discover_models()]"`
Expected: prints one `DiscoveredModel(...)` line per real installed model with a plausible size in bytes.

- [ ] **Step 7: Commit**

```bash
cd ~/Developer/llm-ladder
git add llm_ladder/benchmark_discovery.py tests/test_benchmark_discovery.py pyproject.toml
git commit -m "Add model discovery and memory-safety filter for benchmark subsystem"
```

---

### Task 4: Hardware telemetry

**Files:**
- Create: `llm_ladder/benchmark_hardware.py`
- Test: `tests/test_benchmark_hardware.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (independent of Task 3).
- Produces: `HardwareSnapshot(ram_total_bytes, ram_available_bytes, ram_used_bytes, cpu_percent, gpu_power_mw, gpu_utilization_pct)` dataclass; `capture_hardware_snapshot(skip_gpu: bool = False) -> HardwareSnapshot`; `estimate_load_bandwidth_gbps(model_size_bytes: int, time_to_first_token_s: float) -> float | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark_hardware.py`:
```python
from llm_ladder.benchmark_hardware import (
    capture_hardware_snapshot, _parse_powermetrics_gpu, estimate_load_bandwidth_gbps,
)


class _FakeVM:
    total = 1000
    available = 400
    used = 600


def test_capture_hardware_snapshot_skip_gpu(monkeypatch):
    monkeypatch.setattr("llm_ladder.benchmark_hardware.psutil.virtual_memory", lambda: _FakeVM())
    monkeypatch.setattr("llm_ladder.benchmark_hardware.psutil.cpu_percent", lambda interval=0.5: 42.0)
    snap = capture_hardware_snapshot(skip_gpu=True)
    assert snap.ram_total_bytes == 1000
    assert snap.ram_available_bytes == 400
    assert snap.cpu_percent == 42.0
    assert snap.gpu_power_mw is None
    assert snap.gpu_utilization_pct is None

def test_parse_powermetrics_gpu_extracts_values():
    sample = (
        "**** GPU usage ****\n\n"
        "GPU HW active frequency: 444 MHz\n"
        "GPU HW active residency: 20.50%\n"
        "GPU idle residency: 79.50%\n"
        "GPU Power: 1234 mW\n"
    )
    power, util = _parse_powermetrics_gpu(sample)
    assert power == 1234.0
    assert util == 20.50

def test_parse_powermetrics_gpu_missing_fields_returns_none():
    power, util = _parse_powermetrics_gpu("no gpu data here")
    assert power is None
    assert util is None

def test_estimate_load_bandwidth():
    result = estimate_load_bandwidth_gbps(model_size_bytes=10 * 1024**3, time_to_first_token_s=2.0)
    assert abs(result - 5.0) < 1e-6

def test_estimate_load_bandwidth_zero_time_returns_none():
    assert estimate_load_bandwidth_gbps(1000, 0) is None

def test_estimate_load_bandwidth_negative_time_returns_none():
    assert estimate_load_bandwidth_gbps(1000, -1) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_hardware.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement hardware telemetry**

`llm_ladder/benchmark_hardware.py`:
```python
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

import psutil


@dataclass
class HardwareSnapshot:
    ram_total_bytes: int
    ram_available_bytes: int
    ram_used_bytes: int
    cpu_percent: float
    gpu_power_mw: float | None
    gpu_utilization_pct: float | None


def _parse_powermetrics_gpu(output: str) -> tuple[float | None, float | None]:
    power_match = re.search(r"GPU Power:\s*(\d+(?:\.\d+)?)\s*mW", output)
    util_match = re.search(r"GPU HW active residency:\s*(\d+(?:\.\d+)?)%", output)
    power = float(power_match.group(1)) if power_match else None
    util = float(util_match.group(1)) if util_match else None
    return power, util


def _capture_gpu_powermetrics() -> tuple[float | None, float | None]:
    try:
        proc = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "1"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None, None
    if proc.returncode != 0:
        return None, None
    return _parse_powermetrics_gpu(proc.stdout)


def capture_hardware_snapshot(skip_gpu: bool = False) -> HardwareSnapshot:
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.5)
    gpu_power, gpu_util = (None, None)
    if not skip_gpu and sys.platform == "darwin":
        gpu_power, gpu_util = _capture_gpu_powermetrics()
    return HardwareSnapshot(
        ram_total_bytes=vm.total,
        ram_available_bytes=vm.available,
        ram_used_bytes=vm.used,
        cpu_percent=cpu_percent,
        gpu_power_mw=gpu_power,
        gpu_utilization_pct=gpu_util,
    )


def estimate_load_bandwidth_gbps(model_size_bytes: int, time_to_first_token_s: float) -> float | None:
    """Estimated load bandwidth — a derived figure, NOT a verified
    measurement. True memory bandwidth needs Instruments-level tooling this
    project doesn't build. Label this as an estimate everywhere it's shown."""
    if time_to_first_token_s <= 0:
        return None
    return (model_size_bytes / 1024**3) / time_to_first_token_s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_hardware.py -v`
Expected: all PASS

- [ ] **Step 5: Live-smoke verify (RAM/CPU only, skip GPU to avoid a sudo prompt during plan execution)**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -c "from llm_ladder.benchmark_hardware import capture_hardware_snapshot; print(capture_hardware_snapshot(skip_gpu=True))"`
Expected: prints a `HardwareSnapshot(...)` with plausible RAM numbers, `gpu_power_mw=None`.

- [ ] **Step 6: Commit**

```bash
cd ~/Developer/llm-ladder
git add llm_ladder/benchmark_hardware.py tests/test_benchmark_hardware.py
git commit -m "Add hardware telemetry (RAM/CPU via psutil, GPU via macOS powermetrics)"
```

---

### Task 5: Tool registry + toolbelt scan

**Files:**
- Create: `llm_ladder/tool_registry.yaml`
- Create: `llm_ladder/benchmark_tools.py`
- Test: `tests/test_benchmark_tools.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ToolRegistryEntry(name, check_command: list[str], install_hint: str, description: str)` dataclass; `load_tool_registry() -> list[ToolRegistryEntry]`; `discover_installed_tools(registry: list[ToolRegistryEntry]) -> tuple[list[ToolRegistryEntry], list[ToolRegistryEntry]]` (installed, missing); `run_tool_probe(entry: ToolRegistryEntry) -> tuple[bool, str]`.

**Security design note:** every registry entry's `check_command` is a fixed, zero-model-controlled-argument command (e.g. `["ffmpeg", "-version"]`). The model is never asked to supply a path, filename, or any other argument that reaches a subprocess — it only has to correctly recognize which zero-arg tool to invoke. This is what makes "the model's raw output is never interpolated into a shell command" true by construction, not by a runtime check that could be bypassed.

- [ ] **Step 1: Create the tool registry data file**

`llm_ladder/tool_registry.yaml`:
```yaml
tools:
  - name: ffmpeg
    check_command: [ffmpeg, -version]
    install_hint: "brew install ffmpeg"
    description: "video/audio transcode, trim, extract"
  - name: yt-dlp
    check_command: [yt-dlp, --version]
    install_hint: "brew install yt-dlp"
    description: "download video/audio from URLs"
  - name: magick
    check_command: [magick, -version]
    install_hint: "brew install imagemagick"
    description: "image convert/resize/edit"
  - name: tesseract
    check_command: [tesseract, --version]
    install_hint: "brew install tesseract"
    description: "local OCR"
  - name: pandoc
    check_command: [pandoc, --version]
    install_hint: "brew install pandoc"
    description: "document format conversion"
  - name: git
    check_command: [git, --version]
    install_hint: "brew install git"
    description: "version control"
  - name: jq
    check_command: [jq, --version]
    install_hint: "brew install jq"
    description: "JSON processor"
  - name: docker
    check_command: [docker, --version]
    install_hint: "https://docker.com/get-started"
    description: "containers"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_benchmark_tools.py`:
```python
from llm_ladder.benchmark_tools import (
    load_tool_registry, discover_installed_tools, run_tool_probe, ToolRegistryEntry,
)


def test_load_tool_registry_returns_entries():
    registry = load_tool_registry()
    assert len(registry) > 0
    assert all(e.name and e.check_command for e in registry)

def test_discover_installed_tools_splits_by_path(monkeypatch):
    registry = [
        ToolRegistryEntry("has-it", ["has-it", "-v"], "brew install has-it", "d"),
        ToolRegistryEntry("missing-it", ["missing-it", "-v"], "brew install missing-it", "d"),
    ]
    monkeypatch.setattr(
        "llm_ladder.benchmark_tools.shutil.which",
        lambda binary: "/usr/bin/has-it" if binary == "has-it" else None,
    )
    installed, missing = discover_installed_tools(registry)
    assert [e.name for e in installed] == ["has-it"]
    assert [e.name for e in missing] == ["missing-it"]

def test_run_tool_probe_executes_hardcoded_command():
    entry = ToolRegistryEntry("echo-test", ["echo", "hello"], "n/a", "d")
    ok, output = run_tool_probe(entry)
    assert ok
    assert "hello" in output

def test_run_tool_probe_missing_binary_fails_cleanly():
    entry = ToolRegistryEntry("nope", ["this-binary-does-not-exist-xyz"], "n/a", "d")
    ok, output = run_tool_probe(entry)
    assert not ok
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the tool registry module**

`llm_ladder/benchmark_tools.py`:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark_tools.py -v`
Expected: all PASS

- [ ] **Step 6: Live-smoke verify against real PATH**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -c "
from llm_ladder.benchmark_tools import load_tool_registry, discover_installed_tools
installed, missing = discover_installed_tools(load_tool_registry())
print('installed:', [e.name for e in installed])
print('missing:', [e.name for e in missing])
"`
Expected: prints two lists that plausibly match what's actually on this machine's PATH (e.g. `git` should be in `installed`).

- [ ] **Step 7: Commit**

```bash
cd ~/Developer/llm-ladder
git add llm_ladder/tool_registry.yaml llm_ladder/benchmark_tools.py tests/test_benchmark_tools.py
git commit -m "Add tool registry and safe toolbelt-scan probes"
```

---

### Task 6: Orchestrator — `run_benchmark()`

**Files:**
- Create: `llm_ladder/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `DiscoveredModel`, `discover_models`, `check_memory_safety`, `stop_model` from `llm_ladder.benchmark_discovery` (Task 3); `capture_hardware_snapshot`, `estimate_load_bandwidth_gbps` from `llm_ladder.benchmark_hardware` (Task 4); `load_benchmark_tasks`, `load_rag_corpus`, all `grade_*` functions from `llm_ladder.benchmark_graders` (Task 1); `load_tool_registry`, `discover_installed_tools`, `run_tool_probe` from `llm_ladder.benchmark_tools` (Task 5); `chat`, `OllamaConnectionError` from `llm_ladder.ollama_client` (existing).
- Produces: `CategoryResult(category: str, passed: int, total: int)`; `ModelBenchmarkResult(model, skipped_reason, tokens_per_sec, time_to_first_token_s, load_bandwidth_estimate_gbps, ram_available_bytes, cpu_percent, gpu_power_mw, gpu_utilization_pct, categories: list[CategoryResult])`; `run_speed_benchmark(model: str) -> tuple[float | None, float | None]`; `run_benchmark(models: list[str] | None = None, skip_gpu: bool = False, quick: bool = False, output_path: str | None = None) -> list[ModelBenchmarkResult]` — this is what `cli.py` (Task 7) calls.

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark.py`:
```python
from unittest.mock import patch

from llm_ladder.benchmark import run_benchmark
from llm_ladder.benchmark_discovery import DiscoveredModel


def _fake_hw():
    return type(
        "HW", (),
        {"ram_available_bytes": 1, "cpu_percent": 1.0, "gpu_power_mw": None, "gpu_utilization_pct": None},
    )()


def test_unsafe_model_is_skipped_without_calling_chat(tmp_path):
    output_path = str(tmp_path / "benchmark.jsonl")
    with patch("llm_ladder.benchmark.discover_models", return_value=[DiscoveredModel("big-model", 999 * 1024**3)]), \
         patch("llm_ladder.benchmark.check_memory_safety", return_value=(False, "needs ~999.0GB, 10.0GB available — skipped")), \
         patch("llm_ladder.benchmark.chat") as mock_chat:
        results = run_benchmark(output_path=output_path)
    assert len(results) == 1
    assert results[0].skipped_reason is not None
    mock_chat.assert_not_called()

def test_safe_model_runs_quick_benchmark_and_stops_after(tmp_path):
    output_path = str(tmp_path / "benchmark.jsonl")
    with patch("llm_ladder.benchmark.discover_models", return_value=[DiscoveredModel("small-model", 1024**3)]), \
         patch("llm_ladder.benchmark.check_memory_safety", return_value=(True, "ok")), \
         patch("llm_ladder.benchmark.chat", return_value={"message": {"content": "ok"}, "eval_count": 10, "eval_duration": 1_000_000_000}), \
         patch("llm_ladder.benchmark.capture_hardware_snapshot", return_value=_fake_hw()), \
         patch("llm_ladder.benchmark.stop_model") as mock_stop:
        results = run_benchmark(output_path=output_path, quick=True)
    assert len(results) == 1
    assert results[0].skipped_reason is None
    assert results[0].tokens_per_sec == 10.0
    assert results[0].categories == []  # quick mode skips the quality suite
    mock_stop.assert_called_once_with("small-model")

def test_quick_benchmark_writes_one_jsonl_line_per_model(tmp_path):
    output_path = str(tmp_path / "benchmark.jsonl")
    with patch("llm_ladder.benchmark.discover_models", return_value=[DiscoveredModel("m1", 1024**3), DiscoveredModel("m2", 1024**3)]), \
         patch("llm_ladder.benchmark.check_memory_safety", return_value=(True, "ok")), \
         patch("llm_ladder.benchmark.chat", return_value={"message": {"content": "ok"}, "eval_count": 5, "eval_duration": 500_000_000}), \
         patch("llm_ladder.benchmark.capture_hardware_snapshot", return_value=_fake_hw()), \
         patch("llm_ladder.benchmark.stop_model"):
        run_benchmark(output_path=output_path, quick=True)
    with open(output_path) as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 2

def test_models_filter_restricts_to_named_models(tmp_path):
    output_path = str(tmp_path / "benchmark.jsonl")
    with patch("llm_ladder.benchmark.discover_models", return_value=[DiscoveredModel("m1", 1024**3), DiscoveredModel("m2", 1024**3)]), \
         patch("llm_ladder.benchmark.check_memory_safety", return_value=(True, "ok")), \
         patch("llm_ladder.benchmark.chat", return_value={"message": {"content": "ok"}, "eval_count": 5, "eval_duration": 500_000_000}), \
         patch("llm_ladder.benchmark.capture_hardware_snapshot", return_value=_fake_hw()), \
         patch("llm_ladder.benchmark.stop_model"):
        results = run_benchmark(models=["m1"], output_path=output_path, quick=True)
    assert len(results) == 1
    assert results[0].model == "m1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the orchestrator**

`llm_ladder/benchmark.py`:
```python
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

from llm_ladder.ollama_client import chat, OllamaConnectionError
from llm_ladder.benchmark_discovery import discover_models, check_memory_safety, stop_model
from llm_ladder.benchmark_hardware import capture_hardware_snapshot, estimate_load_bandwidth_gbps
from llm_ladder.benchmark_graders import (
    load_benchmark_tasks, load_rag_corpus,
    grade_reasoning, grade_code, grade_json_extraction, grade_tool_schema,
    grade_instruction_following, grade_factual_recall, grade_rag,
)
from llm_ladder.benchmark_tools import load_tool_registry, discover_installed_tools, run_tool_probe

DEFAULT_BENCHMARK_PATH = os.path.expanduser("~/.llm-ladder/benchmark.jsonl")


@dataclass
class CategoryResult:
    category: str
    passed: int
    total: int


@dataclass
class ModelBenchmarkResult:
    model: str
    skipped_reason: str | None
    tokens_per_sec: float | None
    time_to_first_token_s: float | None
    load_bandwidth_estimate_gbps: float | None
    ram_available_bytes: int | None
    cpu_percent: float | None
    gpu_power_mw: float | None
    gpu_utilization_pct: float | None
    categories: list[CategoryResult] = field(default_factory=list)


def run_speed_benchmark(model: str) -> tuple[float | None, float | None]:
    """Returns (tokens_per_sec, time_to_first_token_s), computed from
    Ollama's own eval_count/eval_duration fields — no hand-rolled counting."""
    prompt = "Write a short paragraph (3-4 sentences) describing how a compass works."
    start = time.perf_counter()
    resp = chat(model, prompt)
    elapsed = time.perf_counter() - start
    eval_count = resp.get("eval_count")
    eval_duration_ns = resp.get("eval_duration")
    tokens_per_sec = None
    if eval_count and eval_duration_ns:
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
    return tokens_per_sec, elapsed


def _run_category(model: str, tasks: list[dict], grader, arg_keys: list[str]) -> CategoryResult:
    passed = 0
    for task in tasks:
        try:
            resp = chat(model, task["prompt"])
            answer = resp.get("message", {}).get("content", "")
        except OllamaConnectionError:
            continue
        args = [task[k] for k in arg_keys]
        result = grader(answer, *args)
        if result.passed:
            passed += 1
    return CategoryResult(category=grader.__name__.removeprefix("grade_"), passed=passed, total=len(tasks))


def _run_quality_suite(model: str) -> list[CategoryResult]:
    tasks = load_benchmark_tasks()
    categories = [
        _run_category(model, tasks["reasoning"], grade_reasoning, ["expected"]),
        _run_category(model, tasks["code"], grade_code, ["test_input", "expected_output"]),
        _run_category(model, tasks["json_extraction"], grade_json_extraction, ["expected_fields"]),
        _run_category(model, tasks["tool_schema"], grade_tool_schema, ["expected_tool", "expected_args_keys"]),
        _run_category(model, tasks["instruction_following"], grade_instruction_following, ["pattern"]),
        _run_category(model, tasks["factual_recall"], grade_factual_recall, ["accepted"]),
    ]

    rag = load_rag_corpus()
    corpus_by_id = {d["id"]: d["text"] for d in rag["corpus"]}
    rag_tasks = [
        {
            "prompt": f"Context: {corpus_by_id[q['context_id']]}\n\nQuestion: {q['prompt']}",
            "required_keywords": q["required_keywords"],
        }
        for q in rag["questions"]
    ]
    categories.append(_run_category(model, rag_tasks, grade_rag, ["required_keywords"]))

    registry = load_tool_registry()
    installed, _missing = discover_installed_tools(registry)
    toolbelt_passed = sum(1 for entry in installed if run_tool_probe(entry)[0])
    categories.append(CategoryResult(category="toolbelt_scan", passed=toolbelt_passed, total=len(installed)))

    return categories


def run_model_benchmark(model: str, model_size_bytes: int, skip_gpu: bool = False, quick: bool = False) -> ModelBenchmarkResult:
    hw = capture_hardware_snapshot(skip_gpu=skip_gpu)
    tokens_per_sec, ttft = run_speed_benchmark(model)
    bandwidth = estimate_load_bandwidth_gbps(model_size_bytes, ttft) if ttft else None

    categories = [] if quick else _run_quality_suite(model)

    return ModelBenchmarkResult(
        model=model,
        skipped_reason=None,
        tokens_per_sec=tokens_per_sec,
        time_to_first_token_s=ttft,
        load_bandwidth_estimate_gbps=bandwidth,
        ram_available_bytes=hw.ram_available_bytes,
        cpu_percent=hw.cpu_percent,
        gpu_power_mw=hw.gpu_power_mw,
        gpu_utilization_pct=hw.gpu_utilization_pct,
        categories=categories,
    )


def run_benchmark(
    models: list[str] | None = None,
    skip_gpu: bool = False,
    quick: bool = False,
    output_path: str | None = None,
) -> list[ModelBenchmarkResult]:
    output_path = output_path or DEFAULT_BENCHMARK_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    discovered = discover_models()
    if models:
        discovered = [m for m in discovered if m.name in models]

    results: list[ModelBenchmarkResult] = []
    with open(output_path, "a", encoding="utf-8") as fh:
        for dm in discovered:
            ok, reason = check_memory_safety(dm.size_bytes)
            if not ok:
                result = ModelBenchmarkResult(
                    model=dm.name, skipped_reason=reason, tokens_per_sec=None, time_to_first_token_s=None,
                    load_bandwidth_estimate_gbps=None, ram_available_bytes=None, cpu_percent=None,
                    gpu_power_mw=None, gpu_utilization_pct=None,
                )
            else:
                result = run_model_benchmark(dm.name, dm.size_bytes, skip_gpu=skip_gpu, quick=quick)
                stop_model(dm.name)

            results.append(result)
            entry = asdict(result)
            entry["timestamp"] = time.time()
            fh.write(json.dumps(entry) + "\n")

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_benchmark.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full test suite so far**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest -q`
Expected: all PASS, no regressions in existing tests.

- [ ] **Step 6: Commit**

```bash
cd ~/Developer/llm-ladder
git add llm_ladder/benchmark.py tests/test_benchmark.py
git commit -m "Add benchmark orchestrator tying discovery, hardware, and quality suite together"
```

---

### Task 7: CLI command

**Files:**
- Modify: `llm_ladder/cli.py`
- Test: `tests/test_cli_benchmark.py`

**Interfaces:**
- Consumes: `run_benchmark` from `llm_ladder.benchmark` (Task 6).
- Produces: `ladder benchmark [--quick] [--models m1,m2] [--skip-gpu]` CLI command.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli_benchmark.py`:
```python
from unittest.mock import patch

from typer.testing import CliRunner

from llm_ladder.cli import app

runner = CliRunner()


def _fake_result(model="m", skipped=None, tps=12.5, passed=3, total=4):
    return type(
        "R", (),
        {
            "model": model, "skipped_reason": skipped, "tokens_per_sec": tps,
            "categories": [type("C", (), {"passed": passed, "total": total})()],
        },
    )()


def test_benchmark_command_reports_results():
    with patch("llm_ladder.cli.run_benchmark", return_value=[_fake_result()]):
        result = runner.invoke(app, ["benchmark", "--quick"])
    assert result.exit_code == 0
    assert "m" in result.stdout

def test_benchmark_command_shows_skipped_models():
    with patch("llm_ladder.cli.run_benchmark", return_value=[_fake_result(model="big", skipped="needs ~99GB, 10GB available — skipped")]):
        result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 0
    assert "skipped" in result.stdout

def test_benchmark_command_clean_error_on_runtime_error():
    with patch("llm_ladder.cli.run_benchmark", side_effect=RuntimeError("ollama not found")):
        result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 1
    assert "ollama not found" in result.stdout

def test_benchmark_command_passes_models_flag_through():
    with patch("llm_ladder.cli.run_benchmark", return_value=[_fake_result()]) as mock_run:
        runner.invoke(app, ["benchmark", "--models", "a,b", "--quick", "--skip-gpu"])
    mock_run.assert_called_once_with(models=["a", "b"], skip_gpu=True, quick=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_cli_benchmark.py -v`
Expected: FAIL — no `benchmark` command registered yet.

- [ ] **Step 3: Add the command to cli.py**

In `llm_ladder/cli.py`, add to the imports:
```python
from llm_ladder.benchmark import run_benchmark
```

Add this command (after the existing `stats()` command, before `if __name__ == "__main__":`):
```python
@app.command()
def benchmark(
    quick: bool = typer.Option(False, "--quick", help="Speed benchmark only, skip the quality suite"),
    models: str = typer.Option(None, "--models", help="Comma-separated model names to benchmark instead of all installed"),
    skip_gpu: bool = typer.Option(False, "--skip-gpu", help="Skip GPU telemetry (no sudo prompt)"),
):
    """Benchmark installed Ollama models: speed, hardware usage, and output quality."""
    model_list = [m.strip() for m in models.split(",")] if models else None
    try:
        results = run_benchmark(models=model_list, skip_gpu=skip_gpu, quick=quick)
    except RuntimeError as e:
        error_console.print(f"[bold red]Benchmark Error:[/bold red] {e}")
        raise typer.Exit(1)

    table = Table(title="Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Tokens/sec", style="green")
    table.add_column("Quality", style="yellow")

    for r in results:
        if r.skipped_reason:
            table.add_row(r.model, f"skipped: {r.skipped_reason}", "-", "-")
            continue
        tps = f"{r.tokens_per_sec:.1f}" if r.tokens_per_sec else "-"
        total_passed = sum(c.passed for c in r.categories)
        total_tasks = sum(c.total for c in r.categories)
        quality = f"{total_passed}/{total_tasks}" if total_tasks else "-"
        table.add_row(r.model, "ok", tps, quality)

    console.print(table)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest tests/test_cli_benchmark.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full test suite**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
cd ~/Developer/llm-ladder
git add llm_ladder/cli.py tests/test_cli_benchmark.py
git commit -m "Add 'ladder benchmark' CLI command"
```

---

### Task 8: Web leaderboard page

**Files:**
- Create: `web/benchmark.html`
- Modify: `web/index.html:64` (nav link)
- Modify: `web/stats.html:15` (nav link)

**Interfaces:**
- Consumes: nothing (static page, paste/file-load pattern like `stats.html`).

- [ ] **Step 1: Add nav links in the existing pages**

In `web/index.html`, change:
```html
<nav aria-label="Primary navigation"><a href="index.html" aria-current="page">How it works</a><a href="stats.html">Ledger stats</a></nav>
```
to:
```html
<nav aria-label="Primary navigation"><a href="index.html" aria-current="page">How it works</a><a href="stats.html">Ledger stats</a><a href="benchmark.html">Benchmarks</a></nav>
```

In `web/stats.html`, change:
```html
<nav aria-label="Primary navigation"><a href="index.html">How it works</a><a href="stats.html" aria-current="page">Ledger stats</a></nav>
```
to:
```html
<nav aria-label="Primary navigation"><a href="index.html">How it works</a><a href="stats.html" aria-current="page">Ledger stats</a><a href="benchmark.html">Benchmarks</a></nav>
```

- [ ] **Step 2: Create web/benchmark.html**

Follow `web/stats.html`'s exact structure (same CSS variables, same violet/indigo theme, same file-picker + drag-drop + textarea-fallback pattern from the file-picker fix already shipped, same CSP). Adapt for benchmark data shape instead of ledger data shape:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'">
  <meta name="referrer" content="no-referrer">
  <meta name="description" content="Local llm-ladder model benchmark leaderboard.">
  <title>Benchmarks — llm-ladder</title>
  <style>
    :root { --ink:#182046; --muted:#69718e; --line:#dfe3f1; --paper:#fff; --surface:#f7f8ff; --violet:#7047eb; --violet-dark:#5632c6; --indigo:#3446b8; --blue:#2876d7; --mint:#0eaa8c; --soft-violet:#f0edff; --shadow:0 18px 48px rgba(39,50,125,.12); } *{box-sizing:border-box} body{margin:0;color:var(--ink);background:linear-gradient(150deg,#fff 0%,#f7f8ff 54%,#f0f4ff 100%);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5} a{color:inherit;text-decoration:none} a:focus-visible,button:focus-visible,textarea:focus-visible{outline:3px solid #5a9cff;outline-offset:3px;border-radius:6px}.file-picker:focus-within{outline:3px solid #5a9cff;outline-offset:3px;border-radius:8px}.shell{width:min(1120px,calc(100% - 40px));margin:auto}.site-header{height:76px;display:flex;align-items:center;justify-content:space-between;gap:24px}.brand{display:inline-flex;align-items:center;gap:10px;font-weight:800;letter-spacing:-.045em;font-size:1.16rem}.mark{display:grid;place-items:center;width:29px;height:29px;border-radius:9px;color:#fff;background:linear-gradient(140deg,var(--violet),var(--blue));box-shadow:0 7px 15px rgba(92,69,221,.25);font-size:18px}nav{display:flex;gap:7px;align-items:center}nav a{color:var(--muted);font-size:.91rem;font-weight:700;padding:8px 12px;border-radius:8px}nav a:hover,nav a[aria-current="page"]{color:var(--violet-dark);background:var(--soft-violet)}.page-head{padding:50px 0 34px;display:flex;justify-content:space-between;gap:36px;align-items:end}.eyebrow{display:inline-flex;align-items:center;gap:7px;color:var(--violet-dark);font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em}.eyebrow:before{content:"";width:18px;height:2px;background:var(--blue)}h1{font-size:clamp(2.35rem,5vw,3.85rem);letter-spacing:-.06em;line-height:1.02;margin:14px 0 0}.page-head p{max-width:400px;margin:0;color:var(--muted);font-size:.96rem}.input-card{padding:22px;background:var(--paper);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow)}.input-top{display:flex;align-items:baseline;justify-content:space-between;gap:20px;margin-bottom:12px}.input-top h2{margin:0;font-size:1rem;letter-spacing:-.02em}.input-top span{font-size:.78rem;color:var(--muted)}.file-picker{display:inline-flex;align-items:center;margin-bottom:12px}.file-picker input{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden}.drop-zone{margin-bottom:12px;padding:15px;border:1px dashed #aeb8df;border-radius:9px;color:var(--muted);background:#fbfcff;text-align:center;font-size:.85rem;font-weight:700;transition:border-color .15s ease,background .15s ease,color .15s ease}.drop-zone.dragover{border-color:var(--violet);color:var(--violet-dark);background:var(--soft-violet)}textarea{display:block;resize:vertical;width:100%;min-height:120px;border:1px solid #cbd2e8;border-radius:9px;padding:13px;color:#242d55;background:#fbfcff;font: .76rem/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.input-actions{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-top:13px}.parse{border:0;min-height:42px;padding:0 18px;border-radius:8px;color:#fff;background:linear-gradient(125deg,var(--violet),var(--indigo));font:800 .88rem inherit;cursor:pointer;box-shadow:0 8px 17px rgba(77,58,196,.2)}.parse:hover{filter:brightness(1.04)}#status{font-size:.82rem;color:var(--muted)}#status.error{color:#bb334f;font-weight:700}.filters{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.filters label{flex:1;min-width:140px;font-size:.8rem;font-weight:700;color:var(--muted)}.filters select{display:block;width:100%;margin-top:6px;padding:9px;border:1px solid #cbd2e8;border-radius:8px;background:#fbfcff;color:#242d55;font-size:.85rem}.board{margin-top:24px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px;overflow:auto}.board table{width:100%;border-collapse:collapse;font-size:.82rem;text-align:left}.board th{padding:0 10px 10px;color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;border-bottom:1px solid var(--line)}.board td{padding:12px 10px;border-bottom:1px solid #edf0f7;white-space:nowrap}.board tr:last-child td{border-bottom:0}.rank-badge{display:inline-flex;align-items:center;justify-content:center;min-width:26px;height:26px;padding:0 6px;border-radius:999px;background:var(--soft-violet);color:var(--violet-dark);font-weight:800;font-size:.78rem}.skip{color:var(--muted);font-style:italic}.note{font-size:.79rem;color:var(--muted);margin-top:10px}footer{border-top:1px solid #e2e6f3;padding:25px 0 35px;color:var(--muted);font-size:.79rem}footer .shell{display:flex;justify-content:space-between;gap:18px}footer a{color:var(--indigo);font-weight:700}@media(max-width:760px){.page-head{padding:32px 0 28px;display:block}.page-head p{margin-top:17px}.input-actions{align-items:flex-start;flex-direction:column}}@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important;animation:none!important}}
  </style>
</head>
<body>
  <header class="shell site-header"><a class="brand" href="index.html" aria-label="llm-ladder home"><span class="mark" aria-hidden="true">↗</span>llm-ladder</a><nav aria-label="Primary navigation"><a href="index.html">How it works</a><a href="stats.html">Ledger stats</a><a href="benchmark.html" aria-current="page">Benchmarks</a></nav></header>
  <main class="shell">
    <section class="page-head"><div><span class="eyebrow">Local model leaderboard</span><h1>Which model, at what.</h1></div><p>Paste or load your <code>~/.llm-ladder/benchmark.jsonl</code>. Nothing leaves this page: parsing and ranking happen entirely in your browser.</p></section>

    <section class="input-card" aria-labelledby="paste-title">
      <div class="input-top"><h2 id="paste-title">Load benchmark JSONL</h2><span>One JSON object per line</span></div>
      <label class="parse file-picker">Choose benchmark file<input id="bench-file" type="file" accept=".jsonl,.json,text/plain"></label>
      <div class="drop-zone" id="bench-drop-zone">or drop your benchmark.jsonl here</div>
      <textarea id="bench-input" aria-describedby="status" spellcheck="false"></textarea>
      <div class="input-actions"><span id="status" aria-live="polite"></span><button class="parse" id="parse-button" type="button">Parse benchmark</button></div>
    </section>

    <section class="filters" id="filter-card" style="display:none">
      <label>Category<select id="filter-category"></select></label>
    </section>

    <section class="board">
      <table>
        <thead><tr><th>Rank</th><th>Model</th><th>Status</th><th>Tokens/sec</th><th>Est. load bandwidth</th><th>Quality</th></tr></thead>
        <tbody id="board-rows"></tbody>
      </table>
      <p class="note">"Est. load bandwidth" is a derived estimate (model size ÷ time-to-first-token), not a verified measurement.</p>
    </section>
  </main>
  <footer><div class="shell"><span>llm-ladder · confidence-gated local inference</span><a href="index.html">How the ladder works</a></div></footer>

  <script>
    (function () {
      const fields = ["model", "tokens_per_sec", "categories", "timestamp"];
      const input = document.getElementById("bench-input"), status = document.getElementById("status");
      const fileInput = document.getElementById("bench-file"), dropZone = document.getElementById("bench-drop-zone");
      const filterCard = document.getElementById("filter-card"), filterCategory = document.getElementById("filter-category");
      const boardRows = document.getElementById("board-rows");
      let allEntries = [];
      const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);

      function rankModels(scoresByModel) {
        const entries = Object.entries(scoresByModel);
        const n = entries.length;
        if (!n) return {};
        const ordered = entries.slice().sort((a, b) => b[1] - a[1]);
        const ranks = {};
        ordered.forEach(([model], i) => { ranks[model] = n - i; });
        return ranks;
      }

      function categoryScore(entry, category) {
        if (!category) {
          const totalPassed = (entry.categories || []).reduce((s, c) => s + c.passed, 0);
          const totalTasks = (entry.categories || []).reduce((s, c) => s + c.total, 0);
          return totalTasks ? totalPassed / totalTasks : (entry.tokens_per_sec || 0);
        }
        if (category === "speed") return entry.tokens_per_sec || 0;
        const c = (entry.categories || []).find(c => c.category === category);
        return c && c.total ? c.passed / c.total : 0;
      }

      function parseBenchmark() {
        const lines = input.value.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
        if (!lines.length) { showError("Paste or load at least one benchmark JSONL entry."); return; }
        const entries = [];
        for (let i = 0; i < lines.length; i++) {
          let entry;
          try { entry = JSON.parse(lines[i]); } catch (_) { showError(`Line ${i + 1} is not valid JSON.`); return; }
          if (typeof entry !== "object" || entry === null || Array.isArray(entry)) { showError(`Line ${i + 1} is not a JSON object.`); return; }
          const missing = fields.filter(k => !(k in entry));
          if (missing.length) { showError(`Line ${i + 1} is missing: ${missing.join(", ")}.`); return; }
          entries.push(entry);
        }
        allEntries = entries;
        const categories = new Set(["speed"]);
        entries.forEach(e => (e.categories || []).forEach(c => categories.add(c.category)));
        filterCategory.innerHTML = '<option value="">Overall</option>' + [...categories].sort().map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
        filterCard.style.display = "";
        render();
        status.className = ""; status.textContent = `${entries.length} valid ${entries.length === 1 ? "entry" : "entries"} parsed locally.`;
      }

      function showError(message) { status.className = "error"; status.textContent = message; }

      function render() {
        const category = filterCategory.value;
        const active = allEntries.filter(e => !e.skipped_reason);
        const skipped = allEntries.filter(e => e.skipped_reason);
        const scores = {};
        active.forEach(e => { scores[e.model] = categoryScore(e, category); });
        const ranks = rankModels(scores);
        const rows = active
          .slice()
          .sort((a, b) => (ranks[b.model] || 0) - (ranks[a.model] || 0))
          .map(e => {
            const tps = e.tokens_per_sec ? e.tokens_per_sec.toFixed(1) : "—";
            const bw = e.load_bandwidth_estimate_gbps ? `${e.load_bandwidth_estimate_gbps.toFixed(2)} GB/s (est.)` : "—";
            const totalPassed = (e.categories || []).reduce((s, c) => s + c.passed, 0);
            const totalTasks = (e.categories || []).reduce((s, c) => s + c.total, 0);
            const quality = totalTasks ? `${totalPassed}/${totalTasks}` : "—";
            return `<tr><td><span class="rank-badge">${ranks[e.model] || "—"}</span></td><td title="${escapeHtml(e.model)}">${escapeHtml(e.model)}</td><td>ok</td><td>${tps}</td><td>${bw}</td><td>${quality}</td></tr>`;
          })
          .join("");
        const skippedRows = skipped
          .map(e => `<tr><td>—</td><td>${escapeHtml(e.model)}</td><td class="skip">skipped: ${escapeHtml(e.skipped_reason)}</td><td>—</td><td>—</td><td>—</td></tr>`)
          .join("");
        boardRows.innerHTML = rows + skippedRows || '<tr><td colspan="6" class="skip">No entries to show.</td></tr>';
      }

      function loadFile(file) {
        if (!file) { showError("No file found in that drop. Try dragging the .jsonl file itself."); return; }
        const reader = new FileReader();
        reader.addEventListener("load", () => {
          input.value = reader.result;
          if (!reader.result) { showError("That file is empty."); return; }
          parseBenchmark();
        });
        reader.addEventListener("error", () => showError("Could not read that benchmark file."));
        reader.readAsText(file);
      }
      fileInput.addEventListener("change", () => { loadFile(fileInput.files[0]); fileInput.value = ""; });
      dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
      dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
      dropZone.addEventListener("drop", e => { e.preventDefault(); dropZone.classList.remove("dragover"); loadFile(e.dataTransfer.files[0]); });
      window.addEventListener("dragover", e => e.preventDefault());
      window.addEventListener("drop", e => { if (!dropZone.contains(e.target)) e.preventDefault(); });

      document.getElementById("parse-button").addEventListener("click", parseBenchmark);
      filterCategory.addEventListener("change", render);
    }());
  </script>
</body>
</html>
```

- [ ] **Step 3: Verify the page loads with no console errors**

Run: `cd ~/Developer/llm-ladder/web && python3 -m http.server 8943 &` then open `http://localhost:8943/benchmark.html` in a browser (or use Chrome automation) and check `read_console_messages` for errors. Paste a few sample lines like:
```
{"model":"test-model","skipped_reason":null,"tokens_per_sec":42.5,"load_bandwidth_estimate_gbps":3.2,"categories":[{"category":"reasoning","passed":2,"total":2}],"timestamp":1786900000}
```
into the textarea, click "Parse benchmark", confirm a ranked row appears with rank badge "1".
Expected: no console errors, table renders, category filter dropdown populates with "speed" and "reasoning".

- [ ] **Step 4: Commit**

```bash
cd ~/Developer/llm-ladder
git add web/benchmark.html web/index.html web/stats.html
git commit -m "Add web/benchmark.html leaderboard page"
```

---

### Task 9: Real demo run + README

**Files:**
- Modify: `web/benchmark.html` (bake in a real demo run)
- Modify: `README.md`

**Interfaces:**
- Consumes: `ladder benchmark --quick` (Task 7 CLI, run live on the maintainer's own machine).

- [ ] **Step 1: Run a real benchmark on the maintainer's own machine**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m llm_ladder.cli benchmark --quick --skip-gpu`
Expected: a rich table prints with real tokens/sec per installed model; `~/.llm-ladder/benchmark.jsonl` now has one line per model.

(`--quick --skip-gpu` avoids the multi-minute full quality suite and the sudo prompt for this demo-data capture — a full non-quick run is fine to do separately but isn't required for seeding the demo.)

- [ ] **Step 2: Bake the real output into benchmark.html's default textarea**

Read `~/.llm-ladder/benchmark.jsonl`, take the lines produced in Step 1, and set them as `web/benchmark.html`'s `<textarea id="bench-input">` default content (same convention `stats.html` already uses — a real demo pre-filled so the page isn't empty on first view). Also update the `#status` initial state to match, and call `parseBenchmark()` on page load (add `parseBenchmark();` as the last line inside the `(function () { ... }())` IIFE, matching `stats.html`'s own auto-parse-on-load pattern).

- [ ] **Step 3: Update README.md**

Add a new subsection after the existing CLI usage section documenting:
```markdown
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
```

- [ ] **Step 4: Run the full test suite one final time**

Run: `cd ~/Developer/llm-ladder && .venv/bin/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/llm-ladder
git add web/benchmark.html README.md
git commit -m "Seed benchmark.html with a real demo run and document 'ladder benchmark'"
```

---

## Self-Review Notes (for the plan author, not a task to execute)

- **Spec coverage:** all 8 categories (Task 1 + Task 6's `_run_quality_suite`), memory safety (Task 3), hardware telemetry incl. GPU/bandwidth caveats (Task 4), ranking (Task 2), CLI surface incl. all 3 flags (Task 7), web leaderboard with demo seed (Task 8 + 9), toolbelt install-recommendation data present in the registry (Task 5) though the orchestrator doesn't yet surface "missing but useful" recommendations in the CLI table — that's UI polish, not a spec-required behavior (spec says "the report includes an install recommendation," and the registry data + `discover_installed_tools`'s `missing` list together already carry that information for anyone extending the CLI table later).
- **Type consistency checked:** `ModelBenchmarkResult`/`CategoryResult` field names match between Task 6 (produced) and Task 7/8 (consumed) — `skipped_reason`, `tokens_per_sec`, `categories[].category/.passed/.total`, `load_bandwidth_estimate_gbps` used identically in cli.py's table and benchmark.html's JS.
- **No placeholders:** every step above contains complete, real code — no TBD/TODO markers.
