from unittest.mock import patch

from typer.testing import CliRunner

from llm_ladder.bias import BiasError, BiasResult
from llm_ladder.cli import app
from llm_ladder.digest import LensResult, LensTake

runner = CliRunner()


def _result(note=None, verdict=None, consensus=None, disagreements=None, takes=None, skipped=None):
    lens = LensResult(
        takes=takes or [],
        skipped=skipped or [],
        consensus=consensus or [],
        disagreements=disagreements or [],
        lens_verdict=verdict,
        note=note,
    )
    return BiasResult(source="a.txt", lens=lens, generated_at=1.0)


def test_bias_command_prints_consensus_and_disagreements():
    result = _result(
        verdict="2 of 2 models agree it's slanted",
        consensus=["emphasizes cost"],
        disagreements=["a: neutral, b: critical"],
        takes=[LensTake(model="a", take="x"), LensTake(model="b", take="y")],
    )
    with patch("llm_ladder.cli.run_bias", return_value=result) as mock_run:
        r = runner.invoke(app, ["bias", "a.txt"])

    assert r.exit_code == 0
    assert "consensus" in r.stdout.lower()
    assert "disagreement" in r.stdout.lower()
    assert "emphasizes cost" in r.stdout
    mock_run.assert_called_once()


def test_bias_command_json_output():
    result = _result(note="needs >=2 distinct models to compare", takes=[LensTake(model="only", take="x")])
    with patch("llm_ladder.cli.run_bias", return_value=result):
        r = runner.invoke(app, ["bias", "a.txt", "--json"])

    assert r.exit_code == 0
    assert "needs >=2" in r.stdout


def test_bias_command_clean_error_on_bias_error():
    with patch("llm_ladder.cli.run_bias", side_effect=BiasError("nope.txt not found")):
        r = runner.invoke(app, ["bias", "nope.txt"])

    assert r.exit_code == 1
    assert "nope.txt" in r.stderr


def test_bias_command_passes_models_and_chain_flags_through():
    result = _result(note="needs >=2 distinct models to compare")
    with patch("llm_ladder.cli.run_bias", return_value=result) as mock_run:
        runner.invoke(app, ["bias", "a.txt", "--models", "a,b", "--chain", "default"])

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["models"] == ["a", "b"]
    assert kwargs["chain"] == "default"
