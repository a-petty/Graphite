"""Report formatting for evaluation results — Rich console tables and JSON export."""

import json
import logging
from pathlib import Path
from typing import Optional

from cortex.evaluation.runner import EvalReport

logger = logging.getLogger(__name__)


class ReportFormatter:
    """Formats EvalReport for console display and JSON export."""

    @staticmethod
    def print_console(report: EvalReport, console=None) -> None:
        """Print a formatted evaluation report to the console using Rich."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        if console is None:
            console = Console()

        # Header
        header = (
            f"[bold white]Cortex Evaluation Report[/bold white]\n"
            f"[dim]Corpus: {report.document_count} documents | "
            f"{report.entity_count} entities | "
            f"{report.edge_count} edges | "
            f"{report.chunk_count} chunks\n"
            f"Mode: {report.mode} | "
            f"Duration: {report.duration_seconds:.1f}s[/dim]"
        )
        console.print(Panel(header, border_style="cyan", expand=False))

        # Metrics table
        table = Table(title="Metric Comparison")
        table.add_column("Metric", style="cyan", min_width=25)
        table.add_column("Cortex", justify="center", min_width=8)
        table.add_column("Baseline", justify="center", min_width=8)
        table.add_column("Winner", justify="center", min_width=8)

        cortex_wins = 0
        comparable_count = 0

        for metric in report.metrics:
            cortex_str = _format_score(metric.score, metric.name)
            baseline_str = "--"
            winner = "--"

            if metric.baseline_score is not None:
                baseline_str = _format_score(metric.baseline_score, metric.name)
                comparable_count += 1

                if metric.score > metric.baseline_score + 0.01:
                    winner = "[green]Cortex[/green]"
                    cortex_wins += 1
                elif metric.baseline_score > metric.score + 0.01:
                    winner = "[red]Baseline[/red]"
                else:
                    winner = "[yellow]Tie[/yellow]"

            table.add_row(metric.name, cortex_str, baseline_str, winner)

        console.print(table)

        # Summary
        if comparable_count > 0:
            console.print(
                f"\n[bold]Summary:[/bold] Cortex outperforms baseline on "
                f"{cortex_wins}/{comparable_count} comparable metrics"
            )

        # Errors
        all_errors = list(report.errors)
        for m in report.metrics:
            all_errors.extend(m.errors)

        if all_errors:
            console.print(f"\n[yellow]Warnings ({len(all_errors)}):[/yellow]")
            for err in all_errors[:10]:
                console.print(f"  [dim]- {err}[/dim]")
            if len(all_errors) > 10:
                console.print(f"  [dim]... and {len(all_errors) - 10} more[/dim]")

    @staticmethod
    def save_json(report: EvalReport, path: Path) -> None:
        """Save the evaluation report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Evaluation report saved to {path}")


def _format_score(score: float, metric_name: str) -> str:
    """Format a score for display."""
    if "Efficiency" in metric_name:
        return f"{score:.2f}/1k"
    return f"{score:.2f}"
