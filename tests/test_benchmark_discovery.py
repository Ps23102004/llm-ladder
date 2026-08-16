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
