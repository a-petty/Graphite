import sys
import argparse
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install as install_rich_traceback

# Install rich traceback handler for cleaner debugging
install_rich_traceback(show_locals=False)

console = Console()

# -----------------------------------------------------------------------------
# Dependency Check: Rust Core
# -----------------------------------------------------------------------------
try:
    from cortex.agent import CortexAgent
except ImportError as e:
    if "semantic_engine" in str(e) or "No module named 'semantic_engine'" in str(e):
        console.print(Panel(
            "[bold red]CRITICAL ERROR: Rust Core Not Found[/bold red]\n\n"
            "The [yellow]semantic_engine[/yellow] extension is missing.\n"
            "Please build the project before running the CLI:\n\n"
            "    [green]$ maturin develop[/green]",
            border_style="red"
        ))
        sys.exit(1)
    else:
        raise e

# -----------------------------------------------------------------------------
# CLI Implementation
# -----------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    """Create the main parser and subparsers."""
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="Cortex: Knowledge Graph Memory System"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- Watch Command ---
    watch_parser = subparsers.add_parser("watch", help="Watch a directory for changes and keep the graph updated.")
    watch_parser.add_argument(
        "project_root", nargs="?", default=".", type=Path,
        help="Target repository path (default: current directory)"
    )

    # --- Query Command ---
    query_parser = subparsers.add_parser("query", help="Ask a question or give a command to the agent.")
    query_parser.add_argument(
        "prompt", type=str, help="The prompt to send to the agent."
    )
    query_parser.add_argument(
        "-p", "--project-root", type=Path, default=".",
        help="Target repository path (default: current directory)"
    )
    query_parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "mlx"],
        help="LLM provider to use (default: ollama)."
    )
    query_parser.add_argument(
        "--model", type=str, default="deepseek-r2-distill-qwen-32b",
        help="Model name or HuggingFace model ID."
    )

    # --- Ingest Command ---
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the knowledge graph.")
    ingest_parser.add_argument(
        "path", type=Path,
        help="File or directory to ingest"
    )
    ingest_parser.add_argument(
        "-p", "--project-root", type=Path, default=".",
        help="Project root for graph storage (default: current directory)"
    )
    ingest_parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "mlx"],
        help="LLM provider to use (default: ollama)."
    )
    ingest_parser.add_argument(
        "--model", type=str, default="llama3.3:70b",
        help="Model name for extraction (default: llama3.3:70b)."
    )

    # --- Chat Command ---
    chat_parser = subparsers.add_parser("chat", help="Start an interactive multi-turn conversation with the agent.")
    chat_parser.add_argument(
        "-p", "--project-root", type=Path, default=".",
        help="Target repository path (default: current directory)"
    )
    chat_parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "mlx"],
        help="LLM provider to use (default: ollama)."
    )
    chat_parser.add_argument(
        "--model", type=str, default="deepseek-r2-distill-qwen-32b",
        help="Model name or HuggingFace model ID."
    )
    chat_parser.add_argument(
        "--max-rounds", type=int, default=5,
        help="Maximum tool-use rounds per turn (default: 5)"
    )

    return parser

def validate_path(target_path: Path) -> Path:
    """Ensure the target path exists and is a directory."""
    resolved_path = target_path.resolve()
    if not resolved_path.exists():
        console.print(f"[bold red]Error:[/bold red] The path '{resolved_path}' does not exist.")
        sys.exit(1)
    if not resolved_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] The path '{resolved_path}' is not a directory.")
        sys.exit(1)
    return resolved_path

def configure_logging(verbose: bool):
    """Adjust logging verbosity based on flags."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("cortex").setLevel(log_level)
    if verbose:
        console.print("[dim]Verbose mode: [green]ON[/green][/dim]")

def handle_watch_command(args: argparse.Namespace):
    """Execute the file watching functionality."""
    target_path = validate_path(args.project_root)
    agent = CortexAgent(project_root=target_path)
    agent.initialize()
    agent.run()

def handle_query_command(args: argparse.Namespace):
    """Execute a single query against the agent."""
    target_path = validate_path(args.project_root)
    console.print(Panel(
        f"[bold white]Cortex Query[/bold white]\n"
        f"[dim]Target: {target_path}\n"
        f"Provider: {args.provider}\n"
        f"Model: {args.model}[/dim]",
        border_style="cyan",
        expand=False
    ))

    agent = CortexAgent(project_root=target_path, provider=args.provider, model_name=args.model)
    agent.initialize()
    agent.query(args.prompt)


def handle_ingest_command(args: argparse.Namespace):
    """Ingest documents into the knowledge graph."""
    from rich.table import Table
    from cortex.config import CortexConfig
    from cortex.ingestion.pipeline import IngestionPipeline
    from cortex.semantic_engine import PyKnowledgeGraph

    project_root = Path(args.project_root).resolve()
    ingest_path = Path(args.path).resolve()

    if not ingest_path.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{ingest_path}' does not exist.")
        sys.exit(1)

    # Create LLM client
    if args.provider == "ollama":
        from cortex.llm import OllamaClient
        llm_client = OllamaClient(model=args.model)
    else:
        from cortex.llm import MLXClient
        llm_client = MLXClient(model=args.model)

    # Load or create knowledge graph
    graph_dir = project_root / ".cortex"
    graph_dir.mkdir(exist_ok=True)
    graph_path = graph_dir / "graph.msgpack"

    if graph_path.exists():
        console.print(f"[dim]Loading existing graph from {graph_path}[/dim]")
        kg = PyKnowledgeGraph.load(str(graph_path))
    else:
        console.print("[dim]Creating new knowledge graph[/dim]")
        kg = PyKnowledgeGraph(str(project_root))

    config = CortexConfig(
        llm_model=args.model,
        llm_provider=args.provider,
        memory_root=project_root / "memory",
    )

    console.print(Panel(
        f"[bold white]Cortex Ingest[/bold white]\n"
        f"[dim]Path: {ingest_path}\n"
        f"Provider: {args.provider}\n"
        f"Model: {args.model}[/dim]",
        border_style="magenta",
        expand=False,
    ))

    pipeline = IngestionPipeline(
        knowledge_graph=kg,
        llm_client=llm_client,
        config=config,
    )

    if ingest_path.is_file():
        results = [pipeline.ingest_file(ingest_path)]
    else:
        results = pipeline.ingest_directory(ingest_path)

    # Display results
    table = Table(title="Ingestion Results")
    table.add_column("Document", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Chunks")
    table.add_column("Entities")
    table.add_column("Edges")
    table.add_column("Time")

    for r in results:
        status_style = {
            "complete": "[green]complete[/green]",
            "partial": "[yellow]partial[/yellow]",
            "failed": "[red]failed[/red]",
        }.get(r.status, r.status)

        doc_name = Path(r.source_document).name
        table.add_row(
            doc_name,
            status_style,
            str(r.chunks_tagged),
            str(r.entities_created + r.entities_linked),
            str(r.edges_created),
            f"{r.duration_seconds:.1f}s",
        )

    console.print(table)

    # Show errors if any
    for r in results:
        if r.errors:
            console.print(f"\n[yellow]Errors for {Path(r.source_document).name}:[/yellow]")
            for err in r.errors:
                console.print(f"  [red]- {err}[/red]")

    # Save graph
    pipeline.save_graph(str(graph_path))
    console.print(f"\n[green]Graph saved to {graph_path}[/green]")


def handle_chat_command(args: argparse.Namespace):
    """Start an interactive multi-turn chat session."""
    from rich.prompt import Prompt

    target_path = validate_path(args.project_root)
    console.print(Panel(
        f"[bold white]Cortex Chat[/bold white]\n"
        f"[dim]Target: {target_path}\n"
        f"Provider: {args.provider}\n"
        f"Model: {args.model}\n"
        f"Max tool rounds: {args.max_rounds}[/dim]",
        border_style="green",
        expand=False
    ))

    agent = CortexAgent(project_root=target_path, provider=args.provider, model_name=args.model)
    agent.initialize()

    console.print("[bold green]Chat session started.[/bold green] Type [bold]/reset[/bold] to clear history, [bold]exit[/bold] or [bold]quit[/bold] to end.\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Chat session ended.[/yellow]")
            break

        stripped = user_input.strip().lower()
        if stripped in ("exit", "quit"):
            console.print("[yellow]Chat session ended.[/yellow]")
            break
        if stripped == "/reset":
            agent.reset_conversation()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue
        if not user_input.strip():
            continue

        agent.chat(user_input, max_tool_rounds=args.max_rounds)
        console.print()  # blank line between turns


def main():
    """Main entry point for the 'cortex' command."""
    parser = create_parser()
    args = parser.parse_args()

    # Default to 'watch' if no command is given
    if args.command is None:
        args.command = "watch"
        # Since project_root is optional for watch, ensure it exists if not specified
        if not hasattr(args, 'project_root'):
            args.project_root = "."

    try:
        configure_logging(args.verbose)
        
        if args.command == "watch":
            handle_watch_command(args)
        elif args.command == "query":
            handle_query_command(args)
        elif args.command == "ingest":
            handle_ingest_command(args)
        elif args.command == "chat":
            handle_chat_command(args)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down agent...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {e}")
        if args.verbose:
            console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    main()