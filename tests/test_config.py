import pytest
from llm_ladder.config import load_chains, default_chains_path

def test_loads_packaged_chains():
    chains = load_chains(default_chains_path())
    assert "default" in chains
    assert chains["default"].tiers[0].model

def test_missing_model_in_tier_raises(tmp_path):
    path = tmp_path / "chains.yaml"
    path.write_text("chains:\n  default:\n    - model: a\n    - samples: 3\n")
    with pytest.raises(ValueError) as exc:
        load_chains(str(path))
    msg = str(exc.value)
    assert "default" in msg
    assert "tier 1" in msg
    assert "model" in msg

def test_malformed_yaml_raises_with_path(tmp_path):
    path = tmp_path / "chains.yaml"
    path.write_text("chains: [unclosed\n")
    with pytest.raises(ValueError) as exc:
        load_chains(str(path))
    assert str(path) in str(exc.value)

def test_missing_file_raises_with_path(tmp_path):
    path = tmp_path / "nope.yaml"
    with pytest.raises(ValueError) as exc:
        load_chains(str(path))
    assert str(path) in str(exc.value)

def test_null_model_value_raises(tmp_path):
    path = tmp_path / "chains.yaml"
    path.write_text("chains:\n  default:\n    - model:\n      samples: 3\n")
    with pytest.raises(ValueError) as exc:
        load_chains(str(path))
    assert "model" in str(exc.value)

def test_null_chain_raises_clean_error(tmp_path):
    path = tmp_path / "chains.yaml"
    path.write_text("chains:\n  default:\n")
    with pytest.raises(ValueError) as exc:
        load_chains(str(path))
    assert "default" in str(exc.value)

def test_zero_samples_raises(tmp_path):
    path = tmp_path / "chains.yaml"
    path.write_text("chains:\n  default:\n    - model: a\n      samples: 0\n")
    with pytest.raises(ValueError) as exc:
        load_chains(str(path))
    assert "samples" in str(exc.value)
