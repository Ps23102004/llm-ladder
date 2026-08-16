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
