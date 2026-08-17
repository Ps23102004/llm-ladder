from unittest.mock import patch

from typer.testing import CliRunner

from llm_ladder.cli import app

runner = CliRunner()


def test_serve_command_calls_server_main():
    with patch("llm_ladder.server.main") as mock_main:
        result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    mock_main.assert_called_once_with()


def test_serve_command_clean_error_on_port_in_use():
    with patch("llm_ladder.server.main", side_effect=OSError("Address already in use")):
        result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1
    assert "Address already in use" in result.stderr
