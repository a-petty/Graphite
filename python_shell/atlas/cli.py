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
    from atlas.agent import AtlasAgent
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
        prog="atlas",
        description="Atlas: Autonomous Coding Agent"
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
        "--model", type=str, default="deepseek-r2-distill-qwen-32b",
        help="The Ollama model to use for the query."
    )

    # --- Chat Command ---
    chat_parser = subparsers.add_parser("chat", help="Start an interactive multi-turn conversation with the agent.")
    chat_parser.add_argument(
        "-p", "--project-root", type=Path, default=".",
        help="Target repository path (default: current directory)"
    )
    chat_parser.add_argument(
        "--model", type=str, default="deepseek-r2-distill-qwen-32b",
        help="The Ollama model to use."
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
    logging.getLogger("atlas").setLevel(log_level)
    if verbose:
        console.print("[dim]Verbose mode: [green]ON[/green][/dim]")

def handle_watch_command(args: argparse.Namespace):
    """Execute the file watching functionality."""
    target_path = validate_path(args.project_root)
    agent = AtlasAgent(project_root=target_path)
    agent.initialize()
    agent.run()

def handle_query_command(args: argparse.Namespace):
    """Execute a single query against the agent."""
    target_path = validate_path(args.project_root)
    console.print(Panel(
        f"[bold white]Atlas Query[/bold white]\n"
        f"[dim]Target: {target_path}\n"
        f"Model: {args.model}[/dim]",
        border_style="cyan",
        expand=False
    ))

    # Instantiate agent with the real LLM client
    agent = AtlasAgent(project_root=target_path, use_real_llm=True, model_name=args.model)
    agent.initialize()
    agent.query(args.prompt)


def handle_chat_command(args: argparse.Namespace):
    """Start an interactive multi-turn chat session."""
    from rich.prompt import Prompt

    target_path = validate_path(args.project_root)
    console.print(Panel(
        f"[bold white]Atlas Chat[/bold white]\n"
        f"[dim]Target: {target_path}\n"
        f"Model: {args.model}\n"
        f"Max tool rounds: {args.max_rounds}[/dim]",
        border_style="green",
        expand=False
    ))

    agent = AtlasAgent(project_root=target_path, use_real_llm=True, model_name=args.model)
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
    """Main entry point for the 'atlas' command."""
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