import time
import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_ladder.config import load_chains, default_chains_path, ChainConfig
from llm_ladder.ollama_client import chat_n, OllamaConnectionError
from llm_ladder.confidence import majority_vote
from llm_ladder.ledger import Ledger

app = typer.Typer()
console = Console()
error_console = Console(stderr=True)

@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt to send to the LLM"),
    chain: str = typer.Option("default", "--chain", help="Name of the chain to use"),
    json_out: bool = typer.Option(False, "--json", help="Output result as JSON")
):
    """Run a prompt through the cascade ladder."""
    chains = load_chains(default_chains_path())

    if chain not in chains:
        error_console.print(f"Error: Chain '{chain}' not found.")
        raise typer.Exit(1)

    chain_config: ChainConfig = chains[chain]
    ledger = Ledger()

    try:
        final_answer = ""
        final_confidence = 0.0
        final_tier_index = 0
        final_model = ""
        found = False

        for i, tier in enumerate(chain_config.tiers):
            start_time = time.perf_counter()
            results = chat_n(tier.model, prompt, tier.samples)
            duration = time.perf_counter() - start_time

            answer, confidence = majority_vote(results)
            is_last = (i == len(chain_config.tiers) - 1)

            ledger.record(
                chain=chain,
                tier_index=i,
                model=tier.model,
                confidence=confidence,
                samples=tier.samples,
                escalated=(i > 0),
                used_last_tier=is_last,
                duration_s=duration
            )

            if confidence >= tier.threshold or is_last:
                final_answer = answer
                final_confidence = confidence
                final_tier_index = i
                final_model = tier.model
                found = True
                break

        if not found:
            error_console.print("Error: No tiers found in chain configuration.")
            raise typer.Exit(1)

        if json_out:
            output = {
                "answer": final_answer,
                "confidence": final_confidence,
                "tier_index": final_tier_index,
                "model": final_model
            }
            console.print(json.dumps(output, indent=2))
        else:
            title = f"Answer from Tier {final_tier_index} ({final_model})"
            body = final_answer
            panel = Panel(
                body,
                title=title,
                subtitle=f"Confidence: {final_confidence:.2f}",
                border_style="green"
            )
            console.print(panel)

    except OllamaConnectionError as e:
        error_console.print(f"[bold red]Ollama Connection Error:[/bold red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        error_console.print(f"[bold red]Value Error:[/bold red] {e}")
        raise typer.Exit(1)

@app.command()
def chains():
    """List available chains and their tiers."""
    chains_config = load_chains(default_chains_path())

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
    ledger = Ledger()
    summary = ledger.summary()

    table = Table(title="Ledger Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Runs", str(summary.get("total_runs", 0)))

    runs_by_tier = summary.get("runs_by_tier", {})
    for tier_index, count in sorted(runs_by_tier.items(), key=lambda x: int(x[0])):
        table.add_row(f"  Tier {tier_index}", str(count))

    table.add_row("Total Duration (s)", f"{summary.get('total_duration_s', 0):.2f}")
    table.add_row("Estimated Savings %", f"{summary.get('estimated_savings_pct', 0):.2f}")

    console.print(table)

if __name__ == "__main__":
    app()
