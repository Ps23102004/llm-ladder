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
    assert "ollama not found" in result.stderr

def test_benchmark_command_passes_models_flag_through():
    with patch("llm_ladder.cli.run_benchmark", return_value=[_fake_result()]) as mock_run:
        runner.invoke(app, ["benchmark", "--models", "a,b", "--quick", "--skip-gpu"])
    mock_run.assert_called_once_with(models=["a", "b"], skip_gpu=True, quick=True)
