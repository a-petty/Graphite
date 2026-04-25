import sys
import argparse
import logging
from datetime import datetime
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
    from graphite.agent import GraphiteAgent
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
        prog="graphite",
        description="Graphite: Knowledge Graph Memory System"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
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
        choices=["ollama", "mlx", "openai", "anthropic"],
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
        help="Project root (used to tag ingested content; default: current directory)"
    )
    ingest_parser.add_argument(
        "--graph-root", type=Path, default=None,
        help="Directory that holds .graphite/graph.msgpack. Defaults to ~."
    )
    ingest_parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "mlx", "openai", "anthropic"],
        help="LLM provider to use (default: ollama)."
    )
    ingest_parser.add_argument(
        "--model", type=str, default="llama3.3:70b",
        help="Model name for extraction (default: llama3.3:70b)."
    )

    # --- Eval Command ---
    eval_parser = subparsers.add_parser("eval", help="Run evaluation benchmarks against the knowledge graph.")
    eval_parser.add_argument(
        "corpus_dir", type=Path,
        help="Directory containing test corpus with eval_queries.json"
    )
    eval_parser.add_argument(
        "-p", "--project-root", type=Path, default=".",
        help="Project root for graph storage (default: current directory)"
    )
    eval_parser.add_argument(
        "--full", action="store_true",
        help="Run full pipeline mode (includes LLM-dependent metrics)"
    )
    eval_parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "mlx", "openai", "anthropic"],
        help="LLM provider for full mode (default: ollama)."
    )
    eval_parser.add_argument(
        "--model", type=str, default="llama3.3:70b",
        help="Model name for full mode (default: llama3.3:70b)."
    )
    eval_parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Save JSON report to this path (default: .graphite/eval_report.json)"
    )
    eval_parser.add_argument(
        "-k", type=int, default=5,
        help="Top-k for retrieval precision metric (default: 5)"
    )

    # --- Ingest Sessions Command ---
    sessions_parser = subparsers.add_parser(
        "ingest-sessions",
        help="Ingest Claude Code conversation transcripts into the knowledge graph."
    )
    sessions_parser.add_argument(
        "-p", "--project-root", type=Path, default=".",
        help="Project root (used to tag ingested content; default: current directory)"
    )
    sessions_parser.add_argument(
        "--graph-root", type=Path, default=None,
        help="Directory that holds .graphite/graph.msgpack. Defaults to ~."
    )
    sessions_parser.add_argument(
        "--claude-dir", type=Path, default=None,
        help="Path to .claude directory (default: ~/.claude)"
    )
    sessions_parser.add_argument(
        "--project", type=str, default=None,
        help="Only ingest sessions from projects matching this name substring"
    )
    sessions_parser.add_argument(
        "--since", type=str, default=None,
        help="Only ingest sessions modified after this date (YYYY-MM-DD)"
    )
    sessions_parser.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "mlx", "openai", "anthropic"],
        help="LLM provider to use (default: ollama)."
    )
    sessions_parser.add_argument(
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
        choices=["ollama", "mlx", "openai", "anthropic"],
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

    # --- Daemon Commands ---
    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Manage the graphited daemon (install, start, stop, status).",
    )
    daemon_sub = daemon_parser.add_subparsers(
        dest="daemon_action",
        help="Daemon action",
    )
    daemon_sub.add_parser("install", help="Install the launchd agent and start the daemon.")
    daemon_sub.add_parser("uninstall", help="Stop the daemon and remove the launchd agent.")
    daemon_sub.add_parser("start", help="Start (or restart) the daemon via launchctl.")
    daemon_sub.add_parser("stop", help="Stop the daemon (launchctl bootout).")
    daemon_sub.add_parser("restart", help="Stop and start the daemon.")
    daemon_sub.add_parser("status", help="Report daemon PID, socket state, and graph stats.")

    # --- Reconcile ---
    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help="Replay any archived sessions missing from the graph.",
    )
    reconcile_parser.add_argument(
        "--archive-dir", type=Path, default=None,
        help="Override archive directory (default: ~/.graphite/archive/sessions).",
    )

    # --- Remember (quick capture) ---
    remember_parser = subparsers.add_parser(
        "remember",
        help="Push a fact into long-term memory via the spool.",
    )
    remember_parser.add_argument(
        "text", type=str,
        help="Text to remember. Quote it: graphite remember \"Sarah prefers async standups\"",
    )
    remember_parser.add_argument(
        "--source", type=str, default=None,
        help="Stable source URI for dedup (default: synthesized).",
    )
    remember_parser.add_argument(
        "--category", type=str, default="Episodic",
        choices=["Episodic", "Semantic", "Procedural"],
        help="Memory category (default: Episodic).",
    )
    remember_parser.add_argument(
        "--project", type=str, default=None,
        help="Project tag.",
    )
    remember_parser.add_argument(
        "--entity-hint", action="append", default=None, dest="entity_hints",
        help="Entity name hint (repeat to pass multiple).",
    )

    # --- Spool ---
    spool_parser = subparsers.add_parser(
        "spool",
        help="Inspect and manage the remember() spool.",
    )
    spool_sub = spool_parser.add_subparsers(dest="spool_action", help="Spool action")
    spool_sub.add_parser("status", help="Show fragment counts + recent batches.")
    spool_flush = spool_sub.add_parser(
        "flush", help="Drain pending fragments through the LLM pipeline.",
    )
    spool_flush.add_argument(
        "--source-filter", type=str, default=None,
        help="Drain only fragments matching this source_id.",
    )
    spool_flush.add_argument(
        "--limit", type=int, default=1000,
        help="Cap on fragments per drain (default: 1000).",
    )
    spool_sub.add_parser(
        "retry-failed",
        help="Move all failed fragments back to pending so the next batch retries.",
    )
    spool_cleanup = spool_sub.add_parser(
        "cleanup", help="Purge extracted fragments older than --retain-days.",
    )
    spool_cleanup.add_argument(
        "--retain-days", type=int, default=30,
        help="Keep extracted fragments this many days (default: 30).",
    )

    # --- MCP wiring (other clients) ---
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Wire Graphite's MCP server into another agent harness's config.",
    )
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_action", help="MCP action")

    mcp_install_p = mcp_sub.add_parser("install", help="Install the Graphite MCP server entry.")
    mcp_install_p.add_argument(
        "--target", choices=("openclaw",), default="openclaw",
        help="Which agent harness to wire (default: openclaw).",
    )
    mcp_install_p.add_argument(
        "--with-plugin", action="store_true",
        help="Also enable the OpenClaw capture plugin (write capability via agent_end).",
    )
    mcp_install_p.add_argument(
        "--plugin-source", type=Path, default=None,
        help="Path to the plugin source directory (default: openclaw_plugin/ in this repo).",
    )
    mcp_install_p.add_argument(
        "--auto-link", action="store_true",
        help="Run `openclaw plugins install -l <path>` automatically (requires openclaw on PATH).",
    )

    for action in ("uninstall", "status"):
        sub = mcp_sub.add_parser(action, help=f"{action} the Graphite MCP server entry.")
        sub.add_argument(
            "--target", choices=("openclaw",), default="openclaw",
            help="Which agent harness to (un)wire (default: openclaw).",
        )

    # --- Claude Code hooks ---
    hooks_parser = subparsers.add_parser(
        "hooks",
        help="Wire Graphite's session-capture hooks into ~/.claude/settings.json.",
    )
    hooks_sub = hooks_parser.add_subparsers(
        dest="hooks_action",
        help="Hooks action",
    )
    hooks_sub.add_parser(
        "install",
        help="Add (or refresh) Graphite hook entries in ~/.claude/settings.json.",
    )
    hooks_sub.add_parser(
        "uninstall",
        help="Remove Graphite hook entries (other tools' hooks are preserved).",
    )
    hooks_sub.add_parser(
        "status",
        help="Show which hooks are wired and how many sessions are archived.",
    )

    return parser


def _create_llm_client(provider: str, model: str):
    """Create an LLM client for the given provider."""
    if provider == "mlx":
        from graphite.llm import MLXClient
        return MLXClient(model=model)
    elif provider == "openai":
        from graphite.llm import OpenAIClient
        return OpenAIClient(model=model)
    elif provider == "anthropic":
        from graphite.llm import AnthropicClient
        return AnthropicClient(model=model)
    else:
        from graphite.llm import OllamaClient
        return OllamaClient(model=model)


def _resolve_graph_root(args: argparse.Namespace, config) -> Path:
    """Resolve the directory under which ``.graphite/graph.msgpack`` is stored.

    Resolution order: ``--graph-root`` CLI arg > config.graph_root > Path.home().
    """
    explicit = getattr(args, "graph_root", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(config.graph_root).expanduser().resolve()


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
    logging.getLogger("graphite").setLevel(log_level)
    if verbose:
        console.print("[dim]Verbose mode: [green]ON[/green][/dim]")

def handle_query_command(args: argparse.Namespace):
    """Execute a single query against the agent."""
    target_path = validate_path(args.project_root)
    console.print(Panel(
        f"[bold white]Graphite Query[/bold white]\n"
        f"[dim]Target: {target_path}\n"
        f"Provider: {args.provider}\n"
        f"Model: {args.model}[/dim]",
        border_style="cyan",
        expand=False
    ))

    agent = GraphiteAgent(project_root=target_path, provider=args.provider, model_name=args.model)
    agent.initialize()
    agent.query(args.prompt)


def handle_ingest_command(args: argparse.Namespace):
    """Ingest documents into the knowledge graph."""
    from rich.table import Table
    from graphite.client import DaemonBackedGraph, DaemonUnavailable, GraphiteClient
    from graphite.config import GraphiteConfig
    from graphite.ingestion.pipeline import IngestionPipeline

    project_root = Path(args.project_root).resolve()
    ingest_path = Path(args.path).resolve()

    if not ingest_path.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{ingest_path}' does not exist.")
        sys.exit(1)

    # Create LLM client
    llm_client = _create_llm_client(args.provider, args.model)

    config = GraphiteConfig(
        llm_model=args.model,
        llm_provider=args.provider,
        memory_root=project_root / "memory",
    )
    graph_root = _resolve_graph_root(args, config)

    client = GraphiteClient()
    try:
        client.ping()
    except DaemonUnavailable as e:
        console.print(
            f"[bold red]Daemon not running:[/bold red] {e}\n"
            f"[dim]Start it with [bold]graphited[/bold] (or "
            f"[bold]graphite daemon start[/bold] once PR 8 ships).[/dim]"
        )
        sys.exit(1)

    kg = DaemonBackedGraph(client)
    graph_path = graph_root / ".graphite" / "graph.msgpack"

    console.print(Panel(
        f"[bold white]Graphite Ingest[/bold white]\n"
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

    # Save via the daemon (owns persistence).
    client.force_save()
    console.print(f"\n[green]Graph saved to {graph_path}[/green]")


def handle_eval_command(args: argparse.Namespace):
    """Run evaluation benchmarks against the knowledge graph."""
    from graphite.config import GraphiteConfig
    from graphite.evaluation.runner import EvalRunner
    from graphite.evaluation.report import ReportFormatter

    project_root = Path(args.project_root).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()

    if not corpus_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Corpus directory '{corpus_dir}' does not exist.")
        sys.exit(1)

    mode = "full_pipeline" if args.full else "graph_only"

    llm_client = None
    if args.full:
        llm_client = _create_llm_client(args.provider, args.model)

    config = GraphiteConfig(
        llm_model=args.model,
        llm_provider=args.provider,
        memory_root=project_root / "memory",
    )

    console.print(Panel(
        f"[bold white]Graphite Evaluation[/bold white]\n"
        f"[dim]Corpus: {corpus_dir}\n"
        f"Mode: {mode}\n"
        f"K: {args.k}[/dim]",
        border_style="blue",
        expand=False,
    ))

    runner = EvalRunner(
        corpus_dir=corpus_dir,
        project_root=project_root,
        config=config,
        mode=mode,
        llm_client=llm_client,
        k=args.k,
    )

    report = runner.run()

    ReportFormatter.print_console(report, console)

    output_path = args.output or (project_root / ".graphite" / "eval_report.json")
    ReportFormatter.save_json(report, output_path)
    console.print(f"\n[green]Report saved to {output_path}[/green]")


def handle_ingest_sessions_command(args: argparse.Namespace):
    """Ingest Claude Code conversation transcripts."""
    from rich.table import Table
    from graphite.client import DaemonBackedGraph, DaemonUnavailable, GraphiteClient
    from graphite.config import GraphiteConfig
    from graphite.ingestion.pipeline import IngestionPipeline

    project_root = Path(args.project_root).resolve()

    # Create LLM client
    llm_client = _create_llm_client(args.provider, args.model)

    config = GraphiteConfig(
        llm_model=args.model,
        llm_provider=args.provider,
        memory_root=project_root / "memory",
    )
    if args.claude_dir:
        config.claude_data_dir = Path(args.claude_dir)

    graph_root = _resolve_graph_root(args, config)
    graph_path = graph_root / ".graphite" / "graph.msgpack"

    client = GraphiteClient()
    try:
        client.ping()
    except DaemonUnavailable as e:
        console.print(
            f"[bold red]Daemon not running:[/bold red] {e}\n"
            f"[dim]Start it with [bold]graphited[/bold].[/dim]"
        )
        sys.exit(1)

    kg = DaemonBackedGraph(client)

    console.print(Panel(
        f"[bold white]Graphite Ingest Sessions[/bold white]\n"
        f"[dim]Claude dir: {config.claude_data_dir}\n"
        f"Project filter: {args.project or 'all'}\n"
        f"Since: {args.since or 'all time'}\n"
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

    results = pipeline.ingest_all_sessions(
        claude_dir=config.claude_data_dir,
        project_filter=args.project,
        since=args.since,
    )

    # Display results
    table = Table(title="Session Ingestion Results")
    table.add_column("Session", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Chunks")
    table.add_column("Filler")
    table.add_column("Entities")
    table.add_column("Edges")
    table.add_column("Time")

    completed = skipped = failed = 0
    for r in results:
        status_style = {
            "complete": "[green]complete[/green]",
            "partial": "[yellow]partial[/yellow]",
            "failed": "[red]failed[/red]",
        }.get(r.status, r.status)

        # Session name from source_document URI
        session_name = r.source_document.rsplit("/", 1)[-1][:12] + "..."
        table.add_row(
            session_name,
            status_style,
            str(r.chunks_tagged),
            str(r.chunks_filler),
            str(r.entities_created + r.entities_linked),
            str(r.edges_created),
            f"{r.duration_seconds:.1f}s",
        )

        if r.status == "complete" and r.chunks_tagged > 0:
            completed += 1
        elif r.status == "complete" and r.chunks_tagged == 0:
            skipped += 1
        else:
            failed += 1

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {completed} ingested, {skipped} skipped (unchanged), {failed} failed"
    )

    # Show errors if any
    for r in results:
        if r.errors:
            session_name = r.source_document.rsplit("/", 1)[-1][:12]
            console.print(f"\n[yellow]Errors for {session_name}:[/yellow]")
            for err in r.errors:
                console.print(f"  [red]- {err}[/red]")

    # Save via the daemon.
    client.force_save()
    console.print(f"\n[green]Graph saved to {graph_path}[/green]")


def handle_chat_command(args: argparse.Namespace):
    """Start an interactive multi-turn chat session."""
    from rich.prompt import Prompt

    target_path = validate_path(args.project_root)
    console.print(Panel(
        f"[bold white]Graphite Chat[/bold white]\n"
        f"[dim]Target: {target_path}\n"
        f"Provider: {args.provider}\n"
        f"Model: {args.model}\n"
        f"Max tool rounds: {args.max_rounds}[/dim]",
        border_style="green",
        expand=False
    ))

    agent = GraphiteAgent(project_root=target_path, provider=args.provider, model_name=args.model)
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


def handle_remember_command(args: argparse.Namespace):
    """Push a single fact into the spool from the command line."""
    from graphite.client import DaemonUnavailable, GraphiteClient

    text = (args.text or "").strip()
    if not text:
        console.print("[bold red]Error:[/bold red] empty text.")
        sys.exit(1)

    try:
        with GraphiteClient() as c:
            result = c.remember(
                text=text,
                source_id=args.source,
                category=args.category,
                project=args.project,
                entity_hints=args.entity_hints,
            )
    except DaemonUnavailable as e:
        console.print(
            f"[bold red]Daemon not running:[/bold red] {e}\n"
            f"[dim]Start it with [bold]graphite daemon start[/bold].[/dim]"
        )
        sys.exit(1)

    console.print(
        f"[green]Remembered[/green] (fragment_id={result['fragment_id']}, "
        f"source_id={result['source_id']}, pending={result['pending_count']})."
    )


def handle_spool_command(args: argparse.Namespace):
    """Dispatch ``graphite spool <action>`` against the daemon."""
    from rich.table import Table
    from graphite.client import DaemonUnavailable, GraphiteClient

    action = getattr(args, "spool_action", None)
    if not action:
        console.print(
            "[yellow]No spool action given.[/yellow] "
            "Try: [bold]status | flush | retry-failed | cleanup[/bold]"
        )
        sys.exit(1)

    try:
        client = GraphiteClient()
        client.connect()
    except DaemonUnavailable as e:
        console.print(
            f"[bold red]Daemon not running:[/bold red] {e}\n"
            f"[dim]Start it with [bold]graphite daemon start[/bold].[/dim]"
        )
        sys.exit(1)

    try:
        if action == "status":
            s = client.spool_status()
            counts = s.get("counts", {})
            border = "yellow" if counts.get("failed", 0) > 0 else "green"
            console.print(Panel(
                f"[bold]Spool[/bold]\n"
                f"  Pending:    {counts.get('pending', 0)}\n"
                f"  Extracting: {counts.get('extracting', 0)}\n"
                f"  Extracted:  {counts.get('extracted', 0)}\n"
                f"  Failed:     {counts.get('failed', 0)}\n"
                f"  Total:      {counts.get('total', 0)}",
                border_style=border,
                expand=False,
            ))

            recent = s.get("recent_batches", []) or []
            if recent:
                table = Table(title="Recent batches")
                table.add_column("batch_id", style="cyan")
                table.add_column("fragments", justify="right")
                table.add_column("sources")
                table.add_column("when")
                for b in recent:
                    when = _format_unix(b.get("extracted_at"))
                    sources = ", ".join((b.get("sources") or [])[:3])
                    if len(b.get("sources") or []) > 3:
                        sources += " ..."
                    table.add_row(
                        b.get("batch_id", "?"),
                        str(b.get("fragment_count", 0)),
                        sources,
                        when,
                    )
                console.print(table)

            failed_sample = s.get("failed_sample", []) or []
            if failed_sample:
                table = Table(title=f"Failed fragments (first {len(failed_sample)})")
                table.add_column("id", justify="right")
                table.add_column("source")
                table.add_column("error", style="red")
                for f in failed_sample:
                    err = (f.get("error") or "")[:80]
                    table.add_row(str(f.get("id")), f.get("source_id", ""), err)
                console.print(table)

        elif action == "flush":
            result = client.flush_spool(
                source_filter=args.source_filter,
                limit=args.limit,
            )
            console.print(
                f"[green]Flush enqueued[/green] (job_id={result.get('job_id')}, "
                f"queue_position={result.get('queue_position')})."
            )
        elif action == "retry-failed":
            result = client.spool_retry_failed()
            n = result.get("reset", 0)
            console.print(f"[green]Reset[/green] {n} failed fragment{'s' if n != 1 else ''} to pending.")
        elif action == "cleanup":
            result = client.spool_cleanup(retain_days=args.retain_days)
            console.print(
                f"[green]Cleanup[/green] removed {result.get('removed', 0)} "
                f"fragments older than {result.get('retain_days')} days."
            )
        else:
            console.print(f"[red]Unknown spool action: {action}[/red]")
            sys.exit(1)
    finally:
        client.close()


def _format_unix(ts) -> str:
    """Format a Unix timestamp as a short local-time string, or '—' if None."""
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError, OverflowError):
        return "—"


def handle_reconcile_command(args: argparse.Namespace):
    """Trigger a backlog reconciliation against the graphited daemon.
    Drains both the session archive and the overflow directory."""
    from graphite.client import DaemonUnavailable, GraphiteClient

    archive_dir = str(args.archive_dir) if args.archive_dir else None

    try:
        with GraphiteClient() as c:
            archive_summary = c.reconcile_archive(archive_dir=archive_dir)
            overflow_summary = c.reconcile_overflow()
    except DaemonUnavailable as e:
        console.print(
            f"[bold red]Daemon not running:[/bold red] {e}\n"
            f"[dim]Start it with [bold]graphite daemon start[/bold].[/dim]"
        )
        sys.exit(1)

    activity = (
        archive_summary["enqueued"] + archive_summary["already_indexed"]
        + overflow_summary["replayed"] + overflow_summary["already_indexed"]
    )
    border = "green" if activity > 0 else "yellow"
    console.print(Panel(
        f"[bold]Reconcile summary[/bold]\n"
        f"  Session archive:   {archive_summary['archive_dir']}\n"
        f"    Scanned:           {archive_summary['scanned']}\n"
        f"    Already indexed:   {archive_summary['already_indexed']}\n"
        f"    Enqueued:          {archive_summary['enqueued']}\n"
        f"    Skipped (no LLM):  {archive_summary['skipped_no_llm']}\n"
        f"    Skipped (parse):   {archive_summary['skipped_unparseable']}\n"
        f"\n"
        f"  Overflow:          {overflow_summary['overflow_dir']}\n"
        f"    Scanned:           {overflow_summary['scanned']}\n"
        f"    Replayed:          {overflow_summary['replayed']}\n"
        f"    Already indexed:   {overflow_summary['already_indexed']}\n"
        f"    Failed:            {overflow_summary['failed']}\n"
        f"    Skipped (parse):   {overflow_summary['skipped_unparseable']}",
        border_style=border,
        expand=False,
    ))


def handle_mcp_command(args: argparse.Namespace):
    """Dispatch ``graphite mcp <action> --target <name>``."""
    from graphite import mcp_install

    action = getattr(args, "mcp_action", None)
    target = getattr(args, "target", "openclaw")
    if not action:
        console.print(
            "[yellow]No mcp action given.[/yellow] "
            "Try: [bold]install | uninstall | status[/bold] [--target openclaw]"
        )
        sys.exit(1)

    try:
        if action == "install":
            console.print(mcp_install.install(
                target=target,
                with_plugin=getattr(args, "with_plugin", False),
                plugin_source=getattr(args, "plugin_source", None),
                auto_link=getattr(args, "auto_link", False),
            ))
        elif action == "uninstall":
            console.print(mcp_install.uninstall(target=target))
        elif action == "status":
            s = mcp_install.status(target=target)
            fully_wired = s.server_installed and s.plugin_installed
            border = "green" if fully_wired else ("yellow" if s.server_installed else "red")
            console.print(Panel(
                f"[bold]Graphite MCP wiring — target: {s.target}[/bold]\n"
                f"  Config:           {s.config_path}  ({'present' if s.config_exists else 'missing'})\n"
                f"  MCP server:       {s.server_installed}\n"
                f"  Capture plugin:   {s.plugin_installed}\n"
                + (f"  Command:          {s.command}\n" if s.command else "")
                + f"\n{s.message}",
                border_style=border,
                expand=False,
            ))
        else:
            console.print(f"[red]Unknown mcp action: {action}[/red]")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[bold red]Cannot proceed:[/bold red] {e}")
        sys.exit(1)


def handle_hooks_command(args: argparse.Namespace):
    """Dispatch ``graphite hooks <action>`` to the settings.json merger."""
    from graphite import hooks_control

    action = getattr(args, "hooks_action", None)
    if not action:
        console.print(
            "[yellow]No hooks action given.[/yellow] "
            "Try: [bold]install | uninstall | status[/bold]"
        )
        sys.exit(1)

    try:
        if action == "install":
            console.print(hooks_control.install())
        elif action == "uninstall":
            console.print(hooks_control.uninstall())
        elif action == "status":
            s = hooks_control.status()
            border = "green" if (
                len(s.installed_events) == len(hooks_control.EVENTS) and s.daemon_reachable
            ) else ("yellow" if s.installed_events else "red")
            installed_summary = ", ".join(s.installed_events) if s.installed_events else "none"
            console.print(Panel(
                f"[bold]Graphite Claude Code hooks[/bold]\n"
                f"  Settings file: {s.settings_path}  ({'present' if s.settings_exists else 'missing'})\n"
                f"  Installed events: {installed_summary}\n"
                f"  Archived sessions: {s.archived_sessions}\n"
                f"  Daemon reachable: {s.daemon_reachable}\n\n"
                f"{s.message}",
                border_style=border,
                expand=False,
            ))
        else:
            console.print(f"[red]Unknown hooks action: {action}[/red]")
            sys.exit(1)
    except ValueError as e:
        # Refused to clobber malformed settings.json — surface the message
        # clearly instead of dumping a traceback.
        console.print(f"[bold red]Cannot proceed:[/bold red] {e}")
        sys.exit(1)


def handle_daemon_command(args: argparse.Namespace):
    """Dispatch ``graphite daemon <action>`` to the launchd-backed control plane."""
    from graphite import daemon_control

    action = getattr(args, "daemon_action", None)
    if not action:
        console.print(
            "[yellow]No daemon action given.[/yellow] "
            "Try: [bold]install | uninstall | start | stop | restart | status[/bold]"
        )
        sys.exit(1)

    if action == "install":
        console.print(daemon_control.install())
    elif action == "uninstall":
        console.print(daemon_control.uninstall())
    elif action == "start":
        console.print(daemon_control.start())
    elif action == "stop":
        console.print(daemon_control.stop())
    elif action == "restart":
        console.print(daemon_control.restart())
    elif action == "status":
        s = daemon_control.status()
        style = "green" if s.reachable else ("yellow" if s.installed else "red")

        live_lines: list[str] = []
        if s.reachable:
            from graphite.client import GraphiteClient
            try:
                with GraphiteClient(timeout_s=2.0) as c:
                    sp = c.spool_status()
                    qs = c.ingest_queue_status()
                counts = sp.get("counts", {})
                live_lines.extend([
                    "",
                    f"  Spool: pending={counts.get('pending', 0)} "
                    f"extracting={counts.get('extracting', 0)} "
                    f"extracted={counts.get('extracted', 0)} "
                    f"failed={counts.get('failed', 0)}",
                    f"  Ingest queue depth: {qs.get('depth', 0)}",
                    f"  LLM: {'configured' if qs.get('llm_configured') else 'NOT CONFIGURED'}"
                    + (f" ({qs.get('llm_provider')}/{qs.get('llm_model')})"
                       if qs.get('llm_configured') else ""),
                ])
            except Exception as e:
                live_lines.append(f"  (live status unavailable: {e})")

        console.print(Panel(
            f"[bold]graphited[/bold]\n"
            f"  Installed: {s.installed}\n"
            f"  Running:   {s.running} (pid: {s.pid})\n"
            f"  Socket:    {s.socket_present}  Reachable: {s.reachable}"
            + ("\n" + "\n".join(live_lines) if live_lines else "")
            + f"\n\n{s.message}",
            border_style=style,
            expand=False,
        ))
    else:
        console.print(f"[red]Unknown daemon action: {action}[/red]")
        sys.exit(1)


def main():
    """Main entry point for the 'graphite' command."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        configure_logging(args.verbose)

        if args.command == "query":
            handle_query_command(args)
        elif args.command == "ingest":
            handle_ingest_command(args)
        elif args.command == "eval":
            handle_eval_command(args)
        elif args.command == "ingest-sessions":
            handle_ingest_sessions_command(args)
        elif args.command == "chat":
            handle_chat_command(args)
        elif args.command == "daemon":
            handle_daemon_command(args)
        elif args.command == "hooks":
            handle_hooks_command(args)
        elif args.command == "mcp":
            handle_mcp_command(args)
        elif args.command == "reconcile":
            handle_reconcile_command(args)
        elif args.command == "remember":
            handle_remember_command(args)
        elif args.command == "spool":
            handle_spool_command(args)

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