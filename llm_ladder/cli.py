import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_ladder.config import load_chains, default_chains_path, ChainConfig
from llm_ladder.ollama_client import OllamaConnectionError, OllamaModelNotFoundError
from llm_ladder.ledger import Ledger
from llm_ladder.engine import run_cascade
from llm_ladder.benchmark import run_benchmark
from llm_ladder.digest import DigestError, DigestOptions, run_digest, to_markdown

app = typer.Typer()
console = Console()
error_console = Console(stderr=True)


def _load_chains_or_exit() -> dict:
    try:
        return load_chains(default_chains_path())
    except ValueError as e:
        error_console.print(f"[bold red]Config Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt to send to the LLM"),
    chain: str = typer.Option("default", "--chain", help="Name of the chain to use"),
    json_out: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """Run a prompt through the cascade ladder."""
    chains = _load_chains_or_exit()

    if chain not in chains:
        error_console.print(f"Error: Chain '{chain}' not found.")
        raise typer.Exit(1)

    chain_config: ChainConfig = chains[chain]
    ledger = Ledger()

    try:
        result = run_cascade(prompt, chain, chain_config, ledger)

        if json_out:
            output = {
                "answer": result.answer,
                "confidence": result.confidence,
                "tier_index": result.tier_index,
                "model": result.model,
            }
            console.print(json.dumps(output, indent=2))
        else:
            title = f"Answer from Tier {result.tier_index} ({result.model})"
            panel = Panel(
                result.answer,
                title=title,
                subtitle=f"Confidence: {result.confidence:.2f}",
                border_style="green"
            )
            console.print(panel)

    except OllamaModelNotFoundError as e:
        error_console.print(f"[bold red]Model Not Found:[/bold red] {e}")
        raise typer.Exit(1)
    except OllamaConnectionError as e:
        error_console.print(f"[bold red]Ollama Connection Error:[/bold red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        error_console.print(f"[bold red]Value Error:[/bold red] {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        error_console.print(f"[bold red]Ledger Error:[/bold red] {e}")
        raise typer.Exit(1)

@app.command()
def chains():
    """List available chains and their tiers."""
    chains_config = _load_chains_or_exit()

    table = Table(title="Available Chains")
    table.add_column("Chain Name", style="cyan")
    table.add_column("Tiers (Model)", style="magenta")

    for name, config in chains_config.items():
        tier_models = " -> ".join([t.model for t in config.tiers])
        table.add_row(name, tier_models)

    console.print(table)

@app.command()
def stats():
    """Display ledger statistics."""
    try:
        ledger = Ledger()
        summary = ledger.summary()
    except OSError as e:
        error_console.print(f"[bold red]Ledger Error:[/bold red] could not read the ledger file: {e}")
        raise typer.Exit(1)

    table = Table(title="Ledger Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Runs", str(summary.get("total_runs", 0)))

    runs_by_tier = summary.get("runs_by_tier", {})
    for tier_index, count in sorted(runs_by_tier.items(), key=lambda x: int(x[0])):
        table.add_row(f"  Tier {tier_index}", str(count))

    table.add_row("Total Duration (s)", f"{summary.get('total_duration_s', 0):.2f}")
    table.add_row("Estimated Savings %", f"{summary.get('estimated_savings_pct', 0):.2f}")

    skipped = summary.get("skipped_malformed", 0)
    if skipped:
        console.print(f"[yellow]{skipped} malformed ledger {'entry' if skipped == 1 else 'entries'} skipped.[/yellow]")

    console.print(table)

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

@app.command()
def digest(
    path: str = typer.Argument(".", help="Local git repo to digest"),
    releases: int = typer.Option(3, "--releases", "-n", help="Number of recent tags to digest"),
    range_: str = typer.Option(None, "--range", help="Explicit commit range, e.g. 'v1..v2'"),
    file: str = typer.Option(None, "--file", help="Digest a text file instead of git history"),
    lens: bool = typer.Option(False, "--lens/--no-lens", help="Add a multi-model disagreement diff"),
    models: str = typer.Option(None, "--models", help="Comma-separated model names for --lens (default: auto-discover installed)"),
    chain: str = typer.Option("default", "--chain", help="Name of the chain to use"),
    since_last: bool = typer.Option(False, "--since-last", help="Only digest tags newer than this repo's last digest"),
    md: str = typer.Option(None, "--md", help="Also write a Markdown export to this path"),
    json_out: bool = typer.Option(False, "--json", help="Output result as JSON"),
):
    """Summarize a repo's history, then show where local models disagree about it."""
    model_list = [m.strip() for m in models.split(",")] if models else None
    opts = DigestOptions(
        repo=path,
        releases=releases,
        range_=range_,
        file_path=file,
        lens=lens,
        models=model_list,
        chain=chain,
        since_last=since_last,
        md_path=md,
        json_out=json_out,
    )

    def _report(progress: dict) -> None:
        phase = progress.get("phase")
        if phase == "map":
            status.update(f"digesting {progress['tag']} ({progress['index']}/{progress['total']})…")
        elif phase == "rollup":
            status.update("combining release summaries…")
        elif phase == "lens":
            status.update(f"lens: {progress['model']} ({progress['index']}/{progress['total']})…")
        elif phase == "judge":
            status.update("judging model disagreement…")

    try:
        with console.status("digesting…") as status:
            result = run_digest(opts, report_progress=_report)
    except DigestError as e:
        error_console.print(f"[bold red]Digest Error:[/bold red] {e}")
        raise typer.Exit(1)
    except OllamaModelNotFoundError as e:
        error_console.print(f"[bold red]Model Not Found:[/bold red] {e}")
        raise typer.Exit(1)
    except OllamaConnectionError as e:
        error_console.print(f"[bold red]Ollama Connection Error:[/bold red] {e}")
        raise typer.Exit(1)

    if md:
        with open(md, "w", encoding="utf-8") as f:
            f.write(to_markdown(result))

    if json_out:
        output = {
            "verdict": result.verdict,
            "repo": result.repo,
            "releases": [
                {"tag": r.tag, "prev": r.prev, "summary": r.summary, "tier_index": r.tier_index, "confidence": r.confidence}
                for r in result.releases
            ],
            "lens": None if result.lens is None else {
                "takes": [{"model": t.model, "take": t.take, "error": t.error} for t in result.lens.takes],
                "skipped": result.lens.skipped,
                "consensus": result.lens.consensus,
                "disagreements": result.lens.disagreements,
                "lens_verdict": result.lens.lens_verdict,
                "note": result.lens.note,
            },
            "generated_at": result.generated_at,
        }
        console.print(json.dumps(output, indent=2))
        return

    panel = Panel(
        result.verdict or "(no verdict)",
        title=f"Digest — {result.repo}",
        border_style="red",
    )
    console.print(panel)

    for r in result.releases:
        heading = r.tag + (f" (since {r.prev})" if r.prev else "")
        console.print(f"\n[bold]{heading}[/bold]")
        console.print(r.summary)

    if result.lens is not None:
        console.print("\n[bold]Model lens[/bold]")
        if result.lens.note:
            console.print(f"[yellow]{result.lens.note}[/yellow]")
        if result.lens.lens_verdict:
            console.print(f"[bold]{result.lens.lens_verdict}[/bold]")
        for c in result.lens.consensus:
            console.print(f"  [green]consensus:[/green] {c}")
        for d_ in result.lens.disagreements:
            console.print(f"  [red]disagreement:[/red] {d_}")
        for model, reason in result.lens.skipped:
            console.print(f"  [dim]skipped {model}: {reason}[/dim]")
        for t in result.lens.takes:
            if t.error:
                console.print(f"  [red]failed {t.model}:[/red] {t.error}")

@app.command()
def serve():
    """Launch the local RepoTriage Agent web server.

    Binds 127.0.0.1 only (see LLM_LADDER_PORT). No auth beyond that — any
    other process on this machine can reach it. Trusted-single-user-machine
    threat model; don't run on a shared host.
    """
    from llm_ladder.server import main  # local import: keeps `ladder run`/etc. fast to start

    try:
        main()
    except OSError as e:
        error_console.print(f"[bold red]Server Error:[/bold red] {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
