import json

import pytest

from llm_ladder import model_config as mc


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "model_config.json"
    monkeypatch.setattr(mc, "CONFIG_PATH", str(path))
    return path


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_save_then_load_roundtrip(config_path):
    cfg = mc.ModelConfig(mode="local", model="llama3", base_url="http://localhost:11434")
    mc.save_config(cfg)
    loaded = mc.load_config()
    assert loaded == cfg


def test_save_includes_api_style(config_path):
    mc.save_config(mc.ModelConfig(mode="api_key", model="claude", api_style="anthropic"))
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data == {
        "mode": "api_key",
        "model": "claude",
        "base_url": None,
        "api_style": "anthropic",
    }


def test_save_never_writes_api_key_field(config_path):
    mc.save_config(mc.ModelConfig(mode="api_key", model="gpt-4o"))
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert not any("key" in field.lower() for field in data)


def test_load_missing_file_returns_default(config_path):
    assert mc.load_config() == mc.ModelConfig(mode="default", model="", base_url=None)


def test_load_invalid_json_returns_default(config_path):
    config_path.write_text("{not json", encoding="utf-8")
    assert mc.load_config().mode == "default"


def test_load_invalid_mode_returns_default(config_path):
    _write(config_path, {"mode": "bogus", "model": "x"})
    cfg = mc.load_config()
    assert cfg.mode == "default"
    assert cfg.model == ""


def test_load_invalid_api_style_falls_back_to_openai(config_path):
    _write(config_path, {"mode": "local", "model": "m", "api_style": "nope"})
    assert mc.load_config().api_style == "openai"


def test_load_missing_fields_get_defaults(config_path):
    _write(config_path, {"mode": "local"})
    cfg = mc.load_config()
    assert cfg.mode == "local"
    assert cfg.model == ""
    assert cfg.base_url is None
    assert cfg.api_style == "openai"


def test_save_invalid_mode_raises(config_path):
    with pytest.raises(ValueError) as exc:
        mc.save_config(mc.ModelConfig(mode="nope", model="m"))
    assert "nope" in str(exc.value)
    assert not config_path.exists()


def test_save_creates_missing_directory(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "dir" / "model_config.json"
    monkeypatch.setattr(mc, "CONFIG_PATH", str(path))
    mc.save_config(mc.ModelConfig(mode="default", model=""))
    assert path.exists()
    assert mc.load_config().mode == "default"
