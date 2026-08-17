from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

from llm_ladder.ollama_client import chat, OllamaConnectionError
from llm_ladder.benchmark_discovery import discover_models, check_memory_safety, stop_model, free_loaded_models
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

    # Free RAM held by any already-loaded models before running memory
    # checks, so a stale loaded model from an earlier session doesn't
    # cause an otherwise-fittable model to be skipped.
    free_loaded_models()

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
