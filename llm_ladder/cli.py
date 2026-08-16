import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_ladder.config import load_chains, default_chains_path, ChainConfig
from llm_ladder.ollama_client import OllamaConnectionError, OllamaModelNotFoundError
from llm_ladder.ledger import Ledger
from llm_ladder.engine import run_cascade

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

if __name__ == "__main__":
    app()
