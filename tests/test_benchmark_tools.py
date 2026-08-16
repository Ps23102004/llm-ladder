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
