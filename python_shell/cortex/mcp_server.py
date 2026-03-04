"""
Cortex MCP Server — Exposes Cortex's semantic graph intelligence as MCP tools.

Provides tools for graph-aware code intelligence: architecture maps,
dependency analysis, semantic search, and context assembly.

CPG-based tools (call graphs, file symbols) were removed in Phase 0b
as part of the Atlas → Cortex transition. They will be replaced by
knowledge graph tools in Phase 4.

Usage:
    cortex-mcp --project-root /path/to/repo
    cortex-mcp --project-root /path/to/repo --verbose
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional

import anyio

# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: The 'mcp' package is not installed.\n"
        "Install it with: pip install -e '.[mcp]'",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from cortex.semantic_engine import (
        RepoGraph,
        scan_repository,
    )
except ImportError as e:
    print(
        "ERROR: Cortex semantic engine not found. Build with: maturin develop\n"
        f"  Details: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging — all output to stderr (required for stdio transport)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("cortex.mcp")

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Cortex",
    instructions="Semantic graph intelligence for codebases — architecture maps, "
    "dependency analysis, call graphs, and optimized context assembly.",
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_graph: Optional[RepoGraph] = None
_project_root: Optional[Path] = None
_graph_initialized: bool = False
_embedding_manager = None  # Lazy: cortex.embeddings.EmbeddingManager
_context_manager = None    # Lazy: cortex.context.ContextManager
_tool_lock: Optional[asyncio.Lock] = None

IGNORED_DIRS = {"node_modules", "target", ".git", "__pycache__", "dist", "build", ".venv", "venv"}
MAX_RESULT_CHARS = 60_000  # Safety cap for MCP tool results

# Re-ranking weights for combining embedding similarity with PageRank.
SIMILARITY_WEIGHT = 0.80
PAGERANK_WEIGHT = 0.20


def _get_lock() -> asyncio.Lock:
    """Get or create the global tool lock (must be called from async context)."""
    global _tool_lock
    if _tool_lock is None:
        _tool_lock = asyncio.Lock()
    return _tool_lock


def _load_ignore_dirs(project_root: Path) -> list:
    """Load ignored directory names from IGNORED_DIRS and .cortexignore file."""
    dirs = list(IGNORED_DIRS)
    ignore_file = project_root / ".cortexignore"
    if ignore_file.exists():
        for line in ignore_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dirs.append(line)
        log.info("Loaded %d ignore patterns from .cortexignore", len(dirs) - len(IGNORED_DIRS))
    return dirs


def _load_source_roots(project_root: Path):
    """Load explicit source roots from .cortex.toml if present."""
    toml_file = project_root / ".cortex.toml"
    if not toml_file.exists():
        return None
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            log.warning(".cortex.toml found but neither tomllib nor tomli available. Ignoring.")
            return None
    try:
        with open(toml_file, "rb") as f:
            config = tomllib.load(f)
        roots = config.get("project", {}).get("source_roots")
        if roots and isinstance(roots, list):
            log.info("Loaded source roots from .cortex.toml: %s", roots)
            return roots
    except Exception as e:
        log.warning("Failed to parse .cortex.toml: %s", e)
    return None


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------
def _initialize_graph(project_root: Path) -> None:
    """Scan repository, build file-level graph, compute PageRank."""
    global _graph, _project_root, _graph_initialized

    _project_root = project_root.resolve()
    log.info("Initializing graph for %s", _project_root)

    ignored = _load_ignore_dirs(_project_root)
    source_roots = _load_source_roots(_project_root)
    _graph = RepoGraph(str(_project_root), ignored_dirs=ignored, source_roots=source_roots)

    files = scan_repository(str(_project_root), ignored_dirs=ignored)
    log.info("Scanned %d files", len(files))

    _graph.build_complete(files)
    _graph.ensure_pagerank_up_to_date()

    stats = _graph.get_statistics()
    log.info(
        "Graph ready: %d files, %d edges, %d symbols",
        stats.node_count,
        stats.edge_count,
        stats.total_definitions,
    )
    _graph_initialized = True


def _ensure_graph() -> None:
    """Lazily initialize the graph on first tool call."""
    if _graph_initialized:
        return
    if _project_root is None:
        raise RuntimeError("Project root not set")
    _initialize_graph(_project_root)


def _ensure_embeddings() -> None:
    """Lazily initialize EmbeddingManager and ContextManager."""
    global _embedding_manager, _context_manager
    if _embedding_manager is not None:
        return
    log.info("Initializing embedding manager (first semantic search call)...")
    from cortex.embeddings import EmbeddingManager
    from cortex.context import ContextManager

    _embedding_manager = EmbeddingManager(repo_graph=_graph, project_root=_project_root)
    _context_manager = ContextManager(_graph, _embedding_manager, max_tokens=100_000)
    log.info("Embedding manager ready")


def _normalize_path(file_path: str) -> str:
    """Accept absolute or project-relative path, return canonical absolute."""
    p = Path(file_path)
    if not p.is_absolute():
        p = _project_root / p
    canonical = p.resolve()
    try:
        canonical.relative_to(_project_root)
    except ValueError:
        raise ValueError(f"Path {canonical} is outside project root {_project_root}")
    return str(canonical)


def _to_relative(abs_path: str) -> str:
    """Convert an absolute path back to project-relative for output."""
    try:
        return str(Path(abs_path).relative_to(_project_root))
    except ValueError:
        return abs_path


def _is_trivial_init(path: Path) -> bool:
    """Check if a file is a trivial __init__.py with minimal content."""
    if path.name != "__init__.py":
        return False
    try:
        content = path.read_text().strip()
        meaningful = "\n".join(
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        return len(meaningful) < 50
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@mcp.tool()
async def cortex_status() -> str:
    """Get graph statistics and readiness status. Call this first to verify Cortex is ready."""
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                stats = _graph.get_statistics()
                root_modules = stats.known_root_modules[:20]
                lines = [
                    "Cortex Semantic Graph Status",
                    f"  Project root: {_project_root}",
                    f"  Files indexed: {stats.node_count}",
                    f"  Dependency edges: {stats.edge_count}",
                    f"    Import edges: {stats.import_edges}",
                    f"    Symbol usage edges: {stats.symbol_edges}",
                    f"  Symbol definitions: {stats.total_definitions}",
                    f"  Module index size: {stats.module_index_size}",
                    f"  Source roots: {stats.source_roots}",
                    f"  Known root modules: {root_modules}",
                    f"  Import resolution: {stats.attempted_imports} attempted, {stats.failed_imports} failed",
                    f"  Unresolved imports: {stats.unresolved_import_count}",
                    f"  Embeddings loaded: {_embedding_manager is not None}",
                ]
                if stats.unresolved_import_count > 0:
                    unresolved = _graph.get_unresolved_imports(5)
                    lines.append("  Top unresolved targets:")
                    for target, count in unresolved:
                        lines.append(f"    {target} (wanted by {count} file(s))")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def get_repository_map(max_files: int = 50) -> str:
    """Get a PageRank-ordered architecture overview of the repository.

    Returns the most architecturally important files with their dependencies
    and symbols, ordered by importance score. Use this to understand the
    overall structure before diving into specific files.

    Args:
        max_files: Maximum number of files to include (default 50).
    """
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                return _graph.generate_map(max_files)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def get_dependencies(file_path: str) -> str:
    """Get outgoing dependencies for a file (what this file imports/uses).

    Args:
        file_path: Absolute or project-relative path to the file.
    """
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                normalized = _normalize_path(file_path)
                if not _graph.has_file(normalized):
                    return f"File not found in graph: {_to_relative(normalized)}. It may not be indexed (wrong extension, syntax errors, or in an ignored directory)."
                deps = _graph.get_dependencies(normalized)
                if not deps:
                    return f"No outgoing dependencies found for {_to_relative(normalized)} (file is in graph but has no imports)"
                lines = [f"Dependencies of {_to_relative(normalized)}:"]
                for dep_path, edge_kind in deps:
                    lines.append(f"  {_to_relative(dep_path)} ({edge_kind})")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def get_dependents(file_path: str) -> str:
    """Get incoming dependents for a file (what depends on this file — blast radius).

    Args:
        file_path: Absolute or project-relative path to the file.
    """
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                normalized = _normalize_path(file_path)
                if not _graph.has_file(normalized):
                    return f"File not found in graph: {_to_relative(normalized)}. It may not be indexed (wrong extension, syntax errors, or in an ignored directory)."
                deps = _graph.get_dependents(normalized)
                if not deps:
                    return f"No incoming dependents found for {_to_relative(normalized)} (file is in graph but nothing imports it)"
                lines = [f"Dependents of {_to_relative(normalized)} (files that depend on this):"]
                for dep_path, edge_kind in deps:
                    lines.append(f"  {_to_relative(dep_path)} ({edge_kind})")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def get_top_ranked_files(limit: int = 20) -> str:
    """Get the most architecturally important files ranked by PageRank.

    Higher-ranked files are more central to the codebase — they are imported
    by many other files and define widely-used symbols.

    Args:
        limit: Number of files to return (default 20).
    """
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                ranked = _graph.get_top_ranked_files(limit)
                if not ranked:
                    return "No ranked files available"
                lines = ["Top files by architectural importance (PageRank):"]
                for i, (path, rank) in enumerate(ranked, 1):
                    lines.append(f"  {i:3d}. {_to_relative(path)} (score: {rank:.4f})")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def find_relevant_files(query: str, top_n: int = 10) -> str:
    """Find files most relevant to a natural language query using semantic search.

    Uses vector embeddings to find files whose content is semantically similar
    to the query. Good for finding code related to a concept or feature.

    Args:
        query: Natural language description of what you're looking for.
        top_n: Number of results to return (default 10).
    """
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                _ensure_embeddings()
                stats = _graph.get_statistics()
                all_files = [Path(p) for p, _ in _graph.get_top_ranked_files(stats.node_count)]
                scored = _embedding_manager.find_relevant_files_scored(query, all_files, top_n=top_n * 3 + 10)
                scored = [(p, sim) for p, sim in scored if not _is_trivial_init(p)]
                if not scored:
                    return f"No relevant files found for: {query}"

                pagerank_map = dict(_graph.get_top_ranked_files(stats.node_count))
                max_pr = max(pagerank_map.values()) if pagerank_map else 1.0
                reranked = []
                for path, similarity in scored:
                    pr = pagerank_map.get(str(path), 0.0)
                    normalized_pr = pr / max_pr if max_pr > 0 else 0.0
                    combined = SIMILARITY_WEIGHT * similarity + PAGERANK_WEIGHT * normalized_pr
                    reranked.append((path, combined))
                reranked.sort(key=lambda x: x[1], reverse=True)
                reranked = reranked[:top_n]

                lines = [f"Files relevant to '{query}':"]
                for i, (path, _score) in enumerate(reranked, 1):
                    lines.append(f"  {i}. {_to_relative(str(path))}")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def assemble_context(query: str) -> str:
    """Assemble optimized code context for a query using Anchor & Expand.

    This is Cortex's core intelligence: it finds relevant files via semantic
    search (anchor), then expands through the dependency graph to pull in
    related code. Returns a three-tier context: repository map, full file
    content for key files, and skeletons for architectural context.

    Args:
        query: The coding task or question to assemble context for.
    """
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                _ensure_embeddings()
                result = _context_manager.assemble_context(query, files_in_scope=[])
                if len(result) > MAX_RESULT_CHARS:
                    return result[:MAX_RESULT_CHARS] + "\n\n[... truncated to fit MCP limit ...]"
                return result
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def get_file_skeleton(file_path: str) -> str:
    """Get function/class signatures without implementation bodies.

    Useful for understanding a file's API surface without reading the full source.

    Args:
        file_path: Absolute or project-relative path to the file.
    """
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                normalized = _normalize_path(file_path)
                skeleton = _graph.get_skeleton(normalized)
                if not skeleton or not skeleton.strip():
                    return f"No skeleton available for {_to_relative(normalized)}"
                return f"Skeleton of {_to_relative(normalized)}:\n\n{skeleton}"
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def cortex_refresh() -> str:
    """Re-scan the repository and rebuild the graph from scratch.

    Use this after significant file system changes (branch switches, large
    merges, etc.) to ensure the graph is up to date.
    """
    async with _get_lock():
        try:
            def _run():
                global _embedding_manager, _context_manager, _graph_initialized
                if _project_root is None:
                    return "ERROR: Project root not set"
                _embedding_manager = None
                _context_manager = None
                _graph_initialized = False
                _initialize_graph(_project_root)
                stats = _graph.get_statistics()
                return (
                    f"Graph refreshed: {stats.node_count} files, "
                    f"{stats.edge_count} edges, {stats.total_definitions} symbols"
                )
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="cortex-mcp",
        description="Cortex MCP Server — semantic graph intelligence for Claude Code",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to the repository to analyze (default: current directory)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("cortex").setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    global _project_root

    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        print(f"ERROR: {project_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    _project_root = project_root
    log.info("Cortex MCP server starting (project: %s, graph builds on first tool call)", project_root)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
