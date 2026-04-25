"""
Graphite MCP Server — Knowledge graph memory tools for LLMs.

Provides 16 tools for knowledge graph interaction: entity lookup,
co-occurrence analysis, semantic search, context assembly, document
ingestion, and graph management.

The server loads a persisted graph from .graphite/graph.msgpack on first
tool call. Ingestion (which requires LLM calls) is only triggered
explicitly via graphite_ingest or graphite_reingest.

Usage:
    graphite-mcp --project-root /path/to/project
    graphite-mcp --project-root /path/to/project --verbose
"""

import sys
import asyncio
import argparse
import atexit
import contextlib
import json
import logging
import signal
from datetime import datetime, timezone
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
    from graphite.semantic_engine import PyKnowledgeGraph
except ImportError as e:
    print(
        "ERROR: Graphite semantic engine not found. Build with: maturin develop\n"
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
log = logging.getLogger("graphite.mcp")

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Graphite",
    instructions="Knowledge graph memory system for LLMs — entity lookup, "
    "co-occurrence analysis, semantic search, evidence retrieval, "
    "and document ingestion.",
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_kg: Optional[PyKnowledgeGraph] = None
_embedding_manager = None  # Lazy: graphite.embeddings.EmbeddingManager
_context_manager = None    # Lazy: graphite.context.MemoryContextManager
_pipeline = None           # Lazy: graphite.ingestion.pipeline.IngestionPipeline
_agent_assembler = None    # Lazy: graphite.agent_context.AgentContextAssembler
_config = None             # Lazy: graphite.config.GraphiteConfig
_project_root: Optional[Path] = None
# Directory that holds .graphite/graph.msgpack. Populated by _ensure_config()
# from GraphiteConfig.graph_root (default ~) or a --graph-root CLI override.
_graph_root: Optional[Path] = None
_graph_root_override: Optional[Path] = None
_graph_initialized: bool = False
_graph_dirty: bool = False
_tool_lock: Optional[asyncio.Lock] = None  # Deprecated: use _write_lock via _get_lock()

# Phase 3: Readers-Writer Lock — allows concurrent reads, exclusive writes
_write_lock: Optional[asyncio.Lock] = None
_readers_lock: Optional[asyncio.Lock] = None
_readers_count: int = 0

# Phase 3: Async post-ingest reflection task
_reflection_task: Optional[asyncio.Task] = None

# Phase 2: Smart cache invalidation — track dirty entity IDs
_dirty_entity_ids: set = set()

# Phase 2: Debounced async save
_last_save_time: float = 0.0
_save_interval: float = 30.0  # minimum seconds between saves
_save_task: Optional[asyncio.Task] = None

MAX_RESULT_CHARS = 60_000  # Safety cap for MCP tool results


# ---------------------------------------------------------------------------
# Graceful shutdown: ensure dirty graph is saved on process exit
# ---------------------------------------------------------------------------
def _force_save() -> None:
    """Save graph and embeddings immediately regardless of debounce. For shutdown."""
    global _graph_dirty, _save_task
    # Cancel any pending debounced save
    if _save_task is not None and not _save_task.done():
        _save_task.cancel()
        _save_task = None
    # Save graph
    if _graph_dirty and _kg is not None and _graph_root is not None:
        try:
            _kg.save(str(_graph_root))
            _graph_dirty = False
            log.info("Graph force-saved on shutdown")
        except Exception as e:
            log.error("Force-save failed: %s", e)
    # Save embeddings
    if _embedding_manager is not None:
        try:
            _embedding_manager.save_entity_embeddings()
        except Exception as e:
            log.error("Embedding force-save failed: %s", e)


def _shutdown_handler(signum=None, frame=None):
    """Signal handler: auto-save on SIGTERM/SIGINT, then exit."""
    sig_name = signal.Signals(signum).name if signum is not None else "EXIT"
    log.info("Received %s — auto-saving before exit", sig_name)
    _force_save()
    sys.exit(0)


# Register atexit handler for normal process termination
atexit.register(_force_save)

# Register signal handlers for graceful shutdown on SIGTERM / SIGINT
try:
    signal.signal(signal.SIGTERM, _shutdown_handler)
except OSError:
    pass  # SIGTERM may be unavailable on some platforms
try:
    signal.signal(signal.SIGINT, _shutdown_handler)
except OSError:
    pass  # Can fail in threads on Windows


# ---------------------------------------------------------------------------
# Lock helper
# ---------------------------------------------------------------------------
def _get_lock() -> asyncio.Lock:
    """Get or create the global write lock (deprecated alias for _write_lock).

    Prefer using _acquire_read() or _acquire_write() directly.
    This returns the write lock for backward compatibility.
    """
    global _tool_lock, _write_lock
    if _write_lock is None:
        _write_lock = asyncio.Lock()
    # Keep deprecated alias in sync
    _tool_lock = _write_lock
    return _write_lock


@contextlib.asynccontextmanager
async def _acquire_read():
    """Acquire read permission — multiple readers can proceed concurrently.

    Readers block writers by holding _write_lock when at least one reader
    is active. The _write_lock is released when the last reader exits.
    """
    global _readers_count

    # Lazily initialize locks
    if _readers_lock is None:
        # We need to set the global — but we can't use 'global' in a
        # nested scope for assignment. Use _get_lock pattern instead.
        pass
    _ensure_rw_locks()

    async with _readers_lock:
        _readers_count += 1
        if _readers_count == 1:
            # First reader acquires write lock to block writers
            await _write_lock.acquire()

    try:
        yield
    finally:
        async with _readers_lock:
            _readers_count -= 1
            if _readers_count == 0:
                # Last reader releases write lock
                _write_lock.release()


@contextlib.asynccontextmanager
async def _acquire_write():
    """Acquire exclusive write permission — blocks until all readers finish."""
    _ensure_rw_locks()

    async with _write_lock:
        yield


def _ensure_rw_locks() -> None:
    """Lazily initialize the readers-writer lock primitives."""
    global _write_lock, _readers_lock
    if _write_lock is None:
        _write_lock = asyncio.Lock()
    if _readers_lock is None:
        _readers_lock = asyncio.Lock()


async def _run_reflection_async(
    project_root: Path,
    kg,
    config,
) -> None:
    """Run lightweight reflection in a background task (fire-and-forget).

    Runs Consolidator.run_lightweight() in a thread pool executor
    since it's CPU-bound. Cancels any previously running reflection
    to avoid piling up reflections from rapid ingestions.
    """
    global _reflection_task
    try:
        from graphite.reflection.consolidator import Consolidator

        consolidator = Consolidator(
            knowledge_graph=kg,
            embedding_manager=_embedding_manager,
            config=config,
        )
        loop = asyncio.get_event_loop()
        cleanup = await loop.run_in_executor(None, consolidator.run_lightweight)
        if cleanup.orphans_removed > 0:
            log.info("Post-ingest reflection: %d orphan(s) removed", cleanup.orphans_removed)
        else:
            log.debug("Post-ingest reflection: no orphans found")
    except asyncio.CancelledError:
        log.debug("Post-ingest reflection cancelled (superseded by newer ingestion)")
    except Exception as e:
        log.warning("Post-ingest reflection failed: %s", e)
    finally:
        _reflection_task = None


def _schedule_reflection() -> None:
    """Schedule a background reflection task, cancelling any existing one.

    Safe to call from any thread — uses call_soon_threadsafe for
    cross-thread scheduling.
    """
    global _reflection_task

    def _do_schedule():
        global _reflection_task
        if _reflection_task is not None and not _reflection_task.done():
            _reflection_task.cancel()
        _reflection_task = asyncio.create_task(
            _run_reflection_async(_project_root, _kg, _config)
        )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(_do_schedule)
        else:
            _do_schedule()
    except RuntimeError:
        # No event loop — skip reflection
        log.debug("No event loop available for scheduling reflection")


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------
def _ensure_config() -> None:
    """Load GraphiteConfig from .graphite.toml or use defaults, and pin the
    graph storage root. The config file stays per-project; the graph root
    defaults to ~ unless overridden by CLI flag or the [paths] TOML section.
    """
    global _config, _graph_root
    if _config is not None:
        return
    from graphite.config import GraphiteConfig

    toml_path = _project_root / ".graphite.toml"
    if toml_path.exists():
        try:
            _config = GraphiteConfig.from_toml(toml_path)
            log.info("Loaded config from %s", toml_path)
        except Exception as e:
            log.warning("Failed to load .graphite.toml: %s — using defaults", e)
            _config = GraphiteConfig()
    else:
        _config = GraphiteConfig()

    # CLI override wins over TOML / default.
    if _graph_root_override is not None:
        _graph_root = _graph_root_override
    else:
        _graph_root = Path(_config.graph_root).expanduser().resolve()
    log.info("Graph storage root: %s", _graph_root)


_daemon_client = None  # Lazy GraphiteClient — connects to graphited on first use


def _ensure_graph() -> None:
    """Connect to the graphited daemon and wrap its graph as the ``_kg`` proxy.

    PR 6 changeover: instead of opening the msgpack file in-process, every
    graph operation now goes through ``graphited`` over its Unix socket.
    The daemon is the sole writer, which kills the class of cross-process
    concurrency bugs the prior architecture was vulnerable to.
    """
    global _kg, _graph_initialized, _daemon_client
    if _graph_initialized:
        return
    if _project_root is None:
        raise RuntimeError("Project root not set")

    _ensure_config()

    from graphite.client import DaemonBackedGraph, DaemonUnavailable, GraphiteClient

    _daemon_client = GraphiteClient()
    try:
        _daemon_client.ping()
    except DaemonUnavailable as e:
        raise RuntimeError(
            f"graphited is not running: {e}. "
            f"Start it with `graphited` or `graphite daemon start`."
        )

    _kg = DaemonBackedGraph(_daemon_client)
    log.info("Connected to graphited (graph_root: %s)", _graph_root)
    _graph_initialized = True

    # P0: Pre-load embeddings at graph init so first search isn't cold-start
    # Previously lazy — only loaded on first find_relevant_entities call
    try:
        _ensure_embeddings()
        log.info("Embeddings pre-loaded at startup")
    except Exception as e:
        log.warning("Failed to pre-load embeddings at startup: %s — will retry on first search", e)


_embeddings_cache_path: Optional[Path] = None  # Set in _ensure_embeddings


def _ensure_embeddings() -> None:
    """Lazily initialize EmbeddingManager and MemoryContextManager."""
    global _embedding_manager, _context_manager, _embeddings_cache_path
    if _embedding_manager is not None:
        return
    _ensure_graph()
    _ensure_config()

    log.info("Initializing embedding manager (first semantic search call)...")
    from graphite.embeddings import EmbeddingManager
    from graphite.context import MemoryContextManager

    # Phase 2: Persist embeddings to <graph_root>/.graphite/entity_embeddings.npz
    # (embeddings live beside the graph file so a single global cache serves
    # every project pointing at the same graph_root).
    _embeddings_cache_path = _graph_root / ".graphite" / "entity_embeddings.npz"
    _embedding_manager = EmbeddingManager(cache_path=_embeddings_cache_path)
    _context_manager = MemoryContextManager(
        knowledge_graph=_kg,
        embedding_manager=_embedding_manager,
        config=_config,
    )
    log.info("Embedding manager ready")


def _ensure_pipeline() -> None:
    """Lazily create the ingestion pipeline (requires LLM)."""
    global _pipeline
    if _pipeline is not None:
        return
    _ensure_graph()
    _ensure_config()

    log.info("Initializing ingestion pipeline...")
    from graphite.ingestion.pipeline import IngestionPipeline

    llm_client = _create_llm_client(_config.llm_provider, _config.llm_model)

    _pipeline = IngestionPipeline(
        knowledge_graph=_kg,
        llm_client=llm_client,
        embedding_manager=_embedding_manager,
        config=_config,
    )
    log.info("Ingestion pipeline ready (provider=%s, model=%s)",
             _config.llm_provider, _config.llm_model)


def _ensure_agent_assembler() -> None:
    """Lazily create the AgentContextAssembler."""
    global _agent_assembler
    if _agent_assembler is not None:
        return
    _ensure_embeddings()

    log.info("Initializing agent context assembler...")
    from graphite.agent_context import AgentContextAssembler

    _agent_assembler = AgentContextAssembler(
        knowledge_graph=_kg,
        embedding_manager=_embedding_manager,
        config=_config,
    )
    log.info("Agent context assembler ready")


# ---------------------------------------------------------------------------
# Entity resolution helper
# ---------------------------------------------------------------------------
def _resolve_entity(ref: str) -> dict:
    """Resolve an entity reference (ID or name) to an entity dict.

    Tries direct ID lookup first, then falls back to substring name search.

    Args:
        ref: Entity UUID or canonical name.

    Returns:
        Parsed entity dict.

    Raises:
        ValueError: If entity is not found.
    """
    # Try direct ID lookup
    entity_json = _kg.get_entity(ref)
    if entity_json is not None:
        return json.loads(entity_json)

    # Fall back to name search
    results_json = _kg.search_entities(ref, 5)
    results = json.loads(results_json)
    if not results:
        raise ValueError(f"Entity not found: '{ref}'")

    # If only one result, use it; otherwise pick exact match or first
    if len(results) == 1:
        return results[0]

    # Try exact case-insensitive match
    ref_lower = ref.lower()
    for entity in results:
        if entity["canonical_name"].lower() == ref_lower:
            return entity

    # Return first match
    return results[0]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _sync_save() -> None:
    """Synchronous graph save (called from thread pool or shutdown)."""
    global _graph_dirty, _last_save_time
    if not _graph_dirty or _kg is None or _graph_root is None:
        return
    try:
        _kg.save(str(_graph_root))
        _graph_dirty = False
        _last_save_time = __import__('time').time()
        log.info("Graph saved")
    except Exception as e:
        log.error("Save failed: %s", e)


def _auto_save() -> None:
    """Debounced synchronous save — only saves if enough time has elapsed.

    Called from within _run() functions (which already run in thread pools
    via anyio.to_thread.run_sync), so the save itself doesn't block the
    event loop. The debounce prevents unnecessary frequent saves when
    multiple rapid mutations occur.

    If the debounced sync save skips, the graph will still be saved on the
    next eligible _auto_save() call, via _schedule_save() from async context,
    or via _force_save() at shutdown.
    """
    import time as _time
    global _last_save_time

    if not _graph_dirty or _kg is None or _graph_root is None:
        return

    now = _time.time()
    if now - _last_save_time < _save_interval and _last_save_time > 0:
        # Too soon — skip. The graph will be saved on the next eligible
        # _auto_save() call, via _schedule_save() from async context, or
        # via _force_save() at shutdown.
        log.debug("Skipping auto-save (debounced, %.0fs since last save)", now - _last_save_time)
        return

    _sync_save()


async def _schedule_save() -> None:
    """Debounced async save — saves graph after _save_interval seconds of quiet.

    If enough time has elapsed since the last save, saves immediately in a
    thread pool (non-blocking). Otherwise schedules a delayed save.
    Use this from async code when you want non-blocking saves.
    """
    global _last_save_time, _save_task
    import time

    if not _graph_dirty or _kg is None or _graph_root is None:
        return

    # Cancel any previously scheduled save
    if _save_task is not None and not _save_task.done():
        _save_task.cancel()

    now = time.time()
    elapsed = now - _last_save_time

    if elapsed >= _save_interval:
        # Enough time passed — save immediately in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_save)
    else:
        # Schedule a save after the remaining interval
        delay = _save_interval - elapsed

        async def _delayed_save():
            await asyncio.sleep(delay)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _sync_save)

        _save_task = asyncio.create_task(_delayed_save())


def _invalidate_caches(entity_ids: Optional[list] = None) -> None:
    """Invalidate embedding and context caches after graph mutations.

    Args:
        entity_ids: If provided, only invalidate these specific entities.
                    If None, invalidate all caches (full reset).
    """
    global _dirty_entity_ids
    if _embedding_manager is not None:
        if entity_ids is not None:
            # Targeted invalidation — only clear specified entities
            for eid in entity_ids:
                _embedding_manager.invalidate_entity_cache(eid)
            _dirty_entity_ids.update(entity_ids)
        else:
            # Full invalidation — clear all embeddings
            _embedding_manager.invalidate_entity_cache()
            # Mark all known entity IDs as dirty
            if _kg is not None:
                try:
                    stats = json.loads(_kg.get_statistics())
                    # We can't list all entity IDs from stats, but clearing the
                    # entire cache forces re-embed on next search anyway
                    _dirty_entity_ids.clear()  # No need to track — full clear already done
                except Exception:
                    _dirty_entity_ids.clear()
    if _context_manager is not None:
        _context_manager.invalidate_caches()
    if _agent_assembler is not None:
        _agent_assembler.invalidate_caches()


def _parse_date(s: str) -> Optional[int]:
    """Parse ISO date string 'YYYY-MM-DD' to Unix timestamp, or None for empty."""
    if not s or not s.strip():
        return None
    try:
        dt = datetime.strptime(s.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        raise ValueError(f"Invalid date format: '{s}'. Expected YYYY-MM-DD.")


def _get_stats() -> dict:
    """Parse graph statistics JSON into a dict."""
    return json.loads(_kg.get_statistics())


def _create_llm_client(provider: str, model: str):
    """Create an LLM client for the given provider and model."""
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


def _get_llm_client():
    """Try to create an LLM client from config. Returns None on failure."""
    _ensure_config()
    try:
        return _create_llm_client(_config.llm_provider, _config.llm_model)
    except Exception as e:
        log.info("LLM not available for reflection: %s", e)
        return None


# ---------------------------------------------------------------------------
# Read-only tools (9) — require only _ensure_graph()
# ---------------------------------------------------------------------------

@mcp.tool()
async def graphite_status() -> str:
    """Get knowledge graph statistics and readiness status. Call this first to verify Graphite is ready."""
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                stats = _get_stats()
                lines = [
                    "Graphite Knowledge Graph Status",
                    f"  Project root: {_project_root}",
                    f"  Graph root: {_graph_root}",
                    f"  Entities: {stats.get('entity_count', 0)}",
                    f"  Co-occurrence edges: {stats.get('edge_count', 0)}",
                    f"  Chunks stored: {stats.get('chunk_count', 0)}",
                    f"  Documents indexed: {stats.get('documents_indexed', 0)}",
                ]
                by_type = stats.get("entities_by_type", {})
                if by_type:
                    lines.append("  Entities by type:")
                    for etype, count in sorted(by_type.items()):
                        lines.append(f"    {etype}: {count}")
                graph_file = _graph_root / ".graphite" / "graph.msgpack"
                lines.append(f"  Persisted: {graph_file.exists()}")
                # Report actual embedding cache state, not just whether manager exists
                emb_loaded = _embedding_manager is not None and len(_embedding_manager.entity_embeddings_cache) > 0
                lines.append(f"  Embeddings loaded: {emb_loaded}")
                if _embedding_manager is not None:
                    lines.append(f"  Cached entity vectors: {len(_embedding_manager.entity_embeddings_cache)}")

                # Ingest queue + spool health — lets Claude see at a glance
                # whether auto-ingest is actually going to produce anything.
                try:
                    if _daemon_client is not None:
                        qstatus = _daemon_client.ingest_queue_status()
                        running_suffix = ""
                        if qstatus.get("current"):
                            running_suffix = f"  (running: {qstatus['current']['id']})"
                        lines.append(f"  Ingest queue depth: {qstatus.get('depth', 0)}{running_suffix}")
                        if qstatus.get("llm_configured"):
                            lines.append(
                                f"  LLM: {qstatus.get('llm_provider')} / {qstatus.get('llm_model')}"
                            )
                        else:
                            lines.append(
                                "  LLM: NOT CONFIGURED — ingest jobs will fail."
                            )
                except Exception as e:
                    lines.append(f"  Ingest queue status unavailable: {e}")

                try:
                    if _daemon_client is not None:
                        sp = _daemon_client.spool_status()
                        c = sp.get("counts", {})
                        lines.append(
                            f"  Spool: pending={c.get('pending', 0)} "
                            f"extracting={c.get('extracting', 0)} "
                            f"extracted={c.get('extracted', 0)} "
                            f"failed={c.get('failed', 0)}"
                        )
                except Exception as e:
                    lines.append(f"  Spool status unavailable: {e}")

                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Consolidated tool surface (PR 7) — 8 tools total
# ---------------------------------------------------------------------------
# The internal helper functions below (get_entity_profile, find_relevant_entities,
# etc.) are intentionally no longer @mcp.tool()-decorated. They remain callable
# as module-level functions so these wrappers can compose them, and so the
# in-process test suite keeps working. Their external MCP exposure has been
# retired to reduce the surface Claude has to reason over when picking tools.


@mcp.tool()
async def recall(
    query: str,
    time_window: str = "",
    source: str = "",
    scope: str = "",
) -> str:
    """Unified retrieval over the knowledge graph.

    Returns the entities most relevant to the query, plus the evidence chunks
    where they appear together. This is the primary read tool — prefer it
    over entity(name) when you don't already know the entity name.

    Args:
        query: Natural-language description of what you're looking for.
        time_window: Reserved for Phase 4 — unused in Phase 1.
        source: Reserved for Phase 4 — unused in Phase 1.
        scope: Reserved for Phase 4 — unused in Phase 1.
    """
    # Phase 4 will respect time_window / source / scope and use an LLM
    # decomposer to compose multi-step retrieval. For Phase 1 we concat
    # the two existing retrieval paths.
    entities_section = await find_relevant_entities(query, top_n=10)
    evidence_section = await assemble_memory(query)
    return (
        f"## Relevant entities\n\n{entities_section}\n\n"
        f"## Evidence\n\n{evidence_section}"
    )


@mcp.tool()
async def entity(name: str) -> str:
    """Full profile for a named entity: type, aliases, co-occurring entities,
    recent source mentions. Use after ``recall`` surfaces a name of interest.
    """
    return await get_entity_profile(name)


@mcp.tool()
async def timeline(name: str, time_window: str = "") -> str:
    """Chronological timeline of when and where an entity was mentioned.

    Args:
        name: Entity name to look up.
        time_window: Reserved for Phase 4 — unused in Phase 1.
    """
    return await get_timeline(name, limit=30)


@mcp.tool()
async def remember(text: str, entities: str = "", source: str = "") -> str:
    """Durably capture a fact, observation, or exchange into long-term memory.

    Returns within milliseconds: the text lands in a SQLite spool managed by
    the graphited daemon, and a batch extractor drains pending fragments
    through the LLM pipeline asynchronously. Use this freely mid-conversation
    when something is worth remembering — there's no extraction cost on the
    hot path.

    Args:
        text: The content to remember (sentence, paragraph, exchange).
        entities: Optional comma-separated hint list of entity names Claude
                  has already nominated. Phase 4 will use these to seed the
                  tagger's canon prompt; for now they're stored alongside
                  the fragment for later use.
        source: Optional stable URI for deduplication
                (e.g. ``"session://abc-123#turn-7"``). The daemon
                synthesizes one if you don't supply it, but a real URI is
                preferred so re-captures coalesce instead of accumulating.
    """
    if _daemon_client is None:
        # Should never happen in normal usage — _ensure_graph wires this up
        # before any tool can run. Surface the failure clearly if it does.
        return "ERROR: graphited not connected — was _ensure_graph skipped?"

    hints = [h.strip() for h in entities.split(",") if h.strip()] if entities else None
    source_id = source or None  # daemon synthesizes one if missing

    def _run():
        try:
            return _daemon_client.remember(
                text=text,
                source_id=source_id,
                category="Episodic",
                entity_hints=hints,
            )
        except Exception as e:
            return {"error": str(e)}

    async with _acquire_read():  # remember is logically a write but the
        # daemon is the writer; the local lock here just keeps us off the
        # exclusive write lock so concurrent reads aren't blocked.
        result = await anyio.to_thread.run_sync(_run)

    if "error" in result:
        return f"ERROR: {result['error']}"
    return (
        f"Remembered (fragment_id={result['fragment_id']}, "
        f"source_id={result['source_id']}, "
        f"pending={result['pending_count']})."
    )


@mcp.tool()
async def ingest_source(source: str, records: str) -> str:
    """Bulk-ingest records from a registered source (Gmail, Calendar, Slack, ...).

    Not implemented in Phase 1. Ships in Phase 3 alongside the multi-source
    ingester architecture. Use the ``graphite`` CLI for file-based ingest
    in the meantime.
    """
    return (
        "ERROR: ingest_source is a Phase 3 stub. "
        "Use the `graphite` CLI for file-based ingest in the meantime."
    )


@mcp.tool()
async def register_source(name: str, schema: str) -> str:
    """Register a new source type with the daemon.

    Not implemented in Phase 1. Ships in Phase 3.
    """
    return "ERROR: register_source is a Phase 3 stub."


@mcp.tool()
async def reflect(mode: str = "full") -> str:
    """Run graph consolidation — merge duplicate entities, dedupe edges.

    Decay and orphan pruning were retired in PR 3 (lifelong-memory mode);
    this tool now performs only merges and deduplication.

    Args:
        mode: "full" (all passes) or "merge" (merges only).
    """
    return await graphite_reflect(mode=mode)


# ---------------------------------------------------------------------------
# Internal helpers (below) — retained for composition by the 8 tools above
# and for direct use in tests. Not exposed via MCP after PR 7.
# ---------------------------------------------------------------------------


async def get_knowledge_map(max_entities: int = 50) -> str:
    """Get a PageRank-ranked knowledge map of the most important entities.

    Returns entities grouped by type with their co-occurrence connections,
    ordered by importance score. Use this to understand the knowledge graph
    structure before diving into specific entities.

    Args:
        max_entities: Maximum number of entities to include (default 50).
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                # Compute PageRank and get top entities
                pagerank_json = _kg.compute_pagerank()
                ranked = json.loads(pagerank_json)
                if not ranked:
                    return "Knowledge graph is empty — no entities indexed."

                # Limit to max_entities
                ranked = ranked[:max_entities]

                # Group by type
                by_type = {}
                for entity_id, score in ranked:
                    entity_json = _kg.get_entity(entity_id)
                    if entity_json is None:
                        continue
                    entity = json.loads(entity_json)
                    etype = entity.get("entity_type", "Unknown")
                    if etype not in by_type:
                        by_type[etype] = []

                    # Get co-occurrence neighbors
                    cooc_json = _kg.get_cooccurrences(entity_id)
                    coocs = json.loads(cooc_json)
                    neighbor_names = set()
                    for item in coocs:
                        nid = item[0]
                        n_json = _kg.get_entity(nid)
                        if n_json:
                            n = json.loads(n_json)
                            neighbor_names.add(n["canonical_name"])

                    by_type[etype].append({
                        "name": entity["canonical_name"],
                        "score": score,
                        "neighbors": sorted(neighbor_names)[:5],
                    })

                lines = ["Knowledge Map (PageRank-ranked):"]
                for etype, entities in sorted(by_type.items()):
                    lines.append(f"\n## {etype}")
                    for e in entities:
                        neighbors_str = ", ".join(e["neighbors"]) if e["neighbors"] else "none"
                        lines.append(
                            f"  - **{e['name']}** (score: {e['score']:.4f}) "
                            f"→ co-occurs with: {neighbors_str}"
                        )
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def get_cooccurrences(entity: str) -> str:
    """Get entities that co-occur with a given entity in document chunks.

    Shows which entities appear alongside the target in the same text chunks,
    with frequency counts and most recent timestamps.

    Args:
        entity: Entity name or UUID.
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                ent = _resolve_entity(entity)
                entity_id = ent["id"]

                cooc_json = _kg.get_cooccurrences(entity_id)
                coocs = json.loads(cooc_json)

                if not coocs:
                    return f"No co-occurrences found for '{ent['canonical_name']}'."

                # Aggregate by neighbor
                neighbor_data = {}
                for item in coocs:
                    nid = item[0]
                    edge = item[1]
                    if nid not in neighbor_data:
                        neighbor_data[nid] = {"count": 0, "latest_ts": None}
                    neighbor_data[nid]["count"] += 1
                    ts = edge.get("timestamp")
                    if ts is not None:
                        prev = neighbor_data[nid]["latest_ts"]
                        if prev is None or ts > prev:
                            neighbor_data[nid]["latest_ts"] = ts

                # Resolve names and sort by count
                entries = []
                for nid, data in neighbor_data.items():
                    n_json = _kg.get_entity(nid)
                    if n_json is None:
                        continue
                    n = json.loads(n_json)
                    entries.append({
                        "name": n["canonical_name"],
                        "type": n.get("entity_type", "Unknown"),
                        "count": data["count"],
                        "latest_ts": data["latest_ts"],
                    })

                entries.sort(key=lambda x: x["count"], reverse=True)

                lines = [f"Co-occurrences for **{ent['canonical_name']}** ({ent.get('entity_type', 'Unknown')}):"]
                for e in entries:
                    ts_str = _format_timestamp(e["latest_ts"])
                    lines.append(
                        f"  - {e['name']} ({e['type']}): {e['count']} co-occurrence(s), last: {ts_str}"
                    )
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def get_entity_mentions(entity: str, limit: int = 20) -> str:
    """Get all chunks where an entity is tagged, ordered by time.

    Returns the actual text chunks from documents where this entity
    was identified, most recent first.

    Args:
        entity: Entity name or UUID.
        limit: Maximum number of chunks to return (default 20).
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                ent = _resolve_entity(entity)
                entity_id = ent["id"]

                chain_json = _kg.get_temporal_chain(entity_id)
                chunks = json.loads(chain_json)

                if not chunks:
                    return f"No mentions found for '{ent['canonical_name']}'."

                # Temporal chain is already time-ordered; take most recent
                chunks = chunks[:limit]

                lines = [f"Mentions of **{ent['canonical_name']}** ({len(chunks)} chunk(s)):"]
                for chunk in chunks:
                    ts_str = _format_timestamp(chunk.get("timestamp"))
                    ctype = chunk.get("chunk_type", "Unknown")
                    source = chunk.get("source_document", "unknown")
                    text = chunk.get("text", "").strip()
                    if len(text) > 300:
                        text = text[:297] + "..."
                    lines.append(f"\n**[{ctype}]** {source} ({ts_str})")
                    lines.append(f"> {text}")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def get_key_entities(limit: int = 20, entity_type: str = "") -> str:
    """Get the most important entities ranked by PageRank.

    Higher-ranked entities are more central to the knowledge graph — they
    co-occur with many other entities across many documents.

    Args:
        limit: Number of entities to return (default 20).
        entity_type: Optional filter by type (e.g. "Person", "Project"). Empty for all.
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                pagerank_json = _kg.compute_pagerank()
                ranked = json.loads(pagerank_json)

                if not ranked:
                    return "No entities in graph."

                entries = []
                for entity_id, score in ranked:
                    entity_json = _kg.get_entity(entity_id)
                    if entity_json is None:
                        continue
                    entity = json.loads(entity_json)
                    etype = entity.get("entity_type", "Unknown")
                    if entity_type and etype != entity_type:
                        continue
                    entries.append({
                        "name": entity["canonical_name"],
                        "type": etype,
                        "score": score,
                    })
                    if len(entries) >= limit:
                        break

                if not entries:
                    filter_msg = f" of type '{entity_type}'" if entity_type else ""
                    return f"No entities{filter_msg} found."

                type_label = f" ({entity_type})" if entity_type else ""
                lines = [f"Key Entities{type_label} by importance (PageRank):"]
                for i, e in enumerate(entries, 1):
                    lines.append(f"  {i:3d}. **{e['name']}** ({e['type']}) — score: {e['score']:.4f}")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def get_entity_profile(entity: str) -> str:
    """Get a full profile for an entity: type, aliases, co-occurrences, and recent chunks.

    Args:
        entity: Entity name or UUID.
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                ent = _resolve_entity(entity)
                entity_id = ent["id"]

                lines = [f"# {ent['canonical_name']} ({ent.get('entity_type', 'Unknown')})"]

                # Aliases
                aliases = ent.get("aliases", [])
                if aliases:
                    lines.append(f"**Aliases:** {', '.join(aliases)}")

                # Source documents
                docs = ent.get("source_documents", [])
                if docs:
                    lines.append(f"**Source documents:** {', '.join(docs)}")

                # Co-occurrences
                cooc_json = _kg.get_cooccurrences(entity_id)
                coocs = json.loads(cooc_json)
                if coocs:
                    neighbor_counts = {}
                    for item in coocs:
                        nid = item[0]
                        neighbor_counts[nid] = neighbor_counts.get(nid, 0) + 1

                    sorted_neighbors = sorted(
                        neighbor_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    lines.append("\n**Co-occurs with:**")
                    for nid, count in sorted_neighbors[:10]:
                        n_json = _kg.get_entity(nid)
                        if n_json:
                            n = json.loads(n_json)
                            lines.append(f"  - {n['canonical_name']} ({n.get('entity_type', '?')}): {count}x")

                # Recent chunks
                chain_json = _kg.get_temporal_chain(entity_id)
                chunks = json.loads(chain_json)
                if chunks:
                    lines.append(f"\n**Recent mentions** ({len(chunks)} total):")
                    for chunk in chunks[:5]:
                        ts_str = _format_timestamp(chunk.get("timestamp"))
                        ctype = chunk.get("chunk_type", "?")
                        text = chunk.get("text", "").strip()
                        if len(text) > 200:
                            text = text[:197] + "..."
                        lines.append(f"  [{ctype}] ({ts_str}) {text}")

                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def get_timeline(entity: str, limit: int = 30) -> str:
    """Get a chronological timeline of chunks mentioning an entity.

    Returns chunks sorted oldest-to-newest, useful for understanding
    how discussions about an entity evolved over time.

    Args:
        entity: Entity name or UUID.
        limit: Maximum number of chunks (default 30).
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                ent = _resolve_entity(entity)
                entity_id = ent["id"]

                chain_json = _kg.get_temporal_chain(entity_id)
                chunks = json.loads(chain_json)

                if not chunks:
                    return f"No timeline data for '{ent['canonical_name']}'."

                # Reverse to oldest-first for timeline view
                chunks = list(reversed(chunks[:limit]))

                lines = [f"Timeline for **{ent['canonical_name']}** ({len(chunks)} entries):"]
                for chunk in chunks:
                    ts_str = _format_timestamp(chunk.get("timestamp"))
                    ctype = chunk.get("chunk_type", "?")
                    source = chunk.get("source_document", "unknown")
                    text = chunk.get("text", "").strip()
                    if len(text) > 200:
                        text = text[:197] + "..."
                    lines.append(f"\n[{ts_str}] **{ctype}** — {source}")
                    lines.append(f"> {text}")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def get_evidence(entity_a: str, entity_b: str, limit: int = 15) -> str:
    """Get chunks where two entities co-occur — evidence of their relationship.

    Returns the actual text from documents where both entities were
    mentioned together, useful for understanding how they relate.

    Args:
        entity_a: First entity name or UUID.
        entity_b: Second entity name or UUID.
        limit: Maximum number of chunks (default 15).
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                ent_a = _resolve_entity(entity_a)
                ent_b = _resolve_entity(entity_b)
                id_a = ent_a["id"]
                id_b = ent_b["id"]

                # Get chunks containing both entities
                chunks_json = _kg.get_chunks_for_entities(json.dumps([id_a, id_b]))
                all_chunks = json.loads(chunks_json)

                # Filter to chunks that contain BOTH entities
                both = []
                for chunk in all_chunks:
                    tags = set(chunk.get("tags", []))
                    if id_a in tags and id_b in tags:
                        both.append(chunk)

                if not both:
                    return (
                        f"No shared chunks found between "
                        f"'{ent_a['canonical_name']}' and '{ent_b['canonical_name']}'."
                    )

                # Sort by timestamp (most recent first), None timestamps last
                both.sort(key=lambda c: c.get("timestamp") or 0, reverse=True)
                both = both[:limit]

                lines = [
                    f"Evidence for **{ent_a['canonical_name']}** ↔ **{ent_b['canonical_name']}** "
                    f"({len(both)} chunk(s)):"
                ]
                for chunk in both:
                    ts_str = _format_timestamp(chunk.get("timestamp"))
                    ctype = chunk.get("chunk_type", "?")
                    source = chunk.get("source_document", "unknown")
                    text = chunk.get("text", "").strip()
                    if len(text) > 400:
                        text = text[:397] + "..."
                    lines.append(f"\n**[{ctype}]** {source} ({ts_str})")
                    lines.append(f"> {text}")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def get_entity_summary(entity: str) -> str:
    """Get a one-line summary of an entity.

    Returns: "Name (Type): co-occurs with X, Y in N chunks"

    Args:
        entity: Entity name or UUID.
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                ent = _resolve_entity(entity)
                entity_id = ent["id"]

                # Count co-occurrences
                cooc_json = _kg.get_cooccurrences(entity_id)
                coocs = json.loads(cooc_json)

                neighbor_counts = {}
                for item in coocs:
                    nid = item[0]
                    neighbor_counts[nid] = neighbor_counts.get(nid, 0) + 1

                # Get chunk count
                chain_json = _kg.get_temporal_chain(entity_id)
                chunks = json.loads(chain_json)

                # Build one-liner
                name = ent["canonical_name"]
                etype = ent.get("entity_type", "Unknown")

                if neighbor_counts:
                    sorted_n = sorted(
                        neighbor_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    top_names = []
                    for nid, _ in sorted_n[:3]:
                        n_json = _kg.get_entity(nid)
                        if n_json:
                            n = json.loads(n_json)
                            top_names.append(n["canonical_name"])
                    neighbors_str = ", ".join(top_names)
                    return f"{name} ({etype}): co-occurs with {neighbors_str} in {len(chunks)} chunk(s)"
                else:
                    return f"{name} ({etype}): {len(chunks)} chunk(s), no co-occurrences"
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Source document lookup tool (Phase 3)
# ---------------------------------------------------------------------------

async def graphite_search_by_source(source_document: str) -> str:
    """Find all entities and chunks associated with a source document.

    Useful for deduplication: query by source_id like "news-2026-04-19-morning"
    to find what was already covered from that source.

    Args:
        source_document: Source document identifier to search for.
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                if _kg is None:
                    return json.dumps({"error": "Graph not initialized"})

                # Get chunks for this source document
                chunks_json = _kg.get_chunks_by_document(source_document)
                chunks = json.loads(chunks_json)

                # Extract unique entity IDs from chunk tags
                entity_ids = set()
                for chunk in chunks:
                    for tag in chunk.get("tags", []):
                        entity_ids.add(tag)

                # Get entity profiles for found entities
                entities = []
                for eid in entity_ids:
                    entity_json = _kg.get_entity(eid)
                    if entity_json:
                        entities.append(json.loads(entity_json))

                return json.dumps({
                    "source_document": source_document,
                    "chunk_count": len(chunks),
                    "entity_count": len(entities),
                    "chunks": chunks[:10],  # Limit output
                    "entities": [{
                        "id": e.get("id"),
                        "name": e.get("canonical_name"),
                        "type": e.get("entity_type"),
                    } for e in entities],
                }, indent=2)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Semantic search tools (2) — require _ensure_embeddings()
# ---------------------------------------------------------------------------

async def find_relevant_entities(query: str, top_n: int = 10) -> str:
    """Find entities most relevant to a query using semantic search.

    Uses embedding similarity combined with PageRank re-ranking to find
    entities whose context is semantically similar to the query.

    Args:
        query: Natural language description of what you're looking for.
        top_n: Number of results to return (default 10).
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                _ensure_embeddings()

                stats = _get_stats()
                if stats.get("entity_count", 0) == 0:
                    return "Knowledge graph is empty — no entities to search."

                # Get all entities and embed them
                pagerank_json = _kg.compute_pagerank()
                ranked = json.loads(pagerank_json)
                all_ids = [eid for eid, _ in ranked]

                # Build entity dicts for embedding
                entities_for_embed = []
                for eid in all_ids:
                    ej = _kg.get_entity(eid)
                    if ej:
                        entities_for_embed.append(json.loads(ej))

                # Mark dirty entities for re-embedding
                if _dirty_entity_ids:
                    _embedding_manager.mark_entities_dirty(list(_dirty_entity_ids))
                    _dirty_entity_ids.clear()

                _embedding_manager.embed_entities(entities_for_embed, _kg)

                # Semantic search
                scored = _embedding_manager.find_relevant_entities_scored(
                    query, all_ids, top_n=top_n * 3
                )

                if not scored:
                    return f"No relevant entities found for: {query}"

                # Re-rank with PageRank
                pr_map = {eid: score for eid, score in ranked}
                max_pr = max(pr_map.values()) if pr_map else 1.0

                sim_weight = _config.similarity_weight if _config else 0.80
                pr_weight = _config.pagerank_weight if _config else 0.20

                reranked = []
                for eid, sim in scored:
                    pr = pr_map.get(eid, 0.0)
                    normalized_pr = pr / max_pr if max_pr > 0 else 0.0
                    combined = sim_weight * sim + pr_weight * normalized_pr
                    reranked.append((eid, combined))

                reranked.sort(key=lambda x: x[1], reverse=True)
                reranked = reranked[:top_n]

                lines = [f"Entities relevant to '{query}':"]
                for i, (eid, score) in enumerate(reranked, 1):
                    ej = _kg.get_entity(eid)
                    if ej:
                        e = json.loads(ej)
                        lines.append(
                            f"  {i}. **{e['canonical_name']}** "
                            f"({e.get('entity_type', '?')}) — score: {score:.3f}"
                        )
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def assemble_memory(query: str, time_start: str = "", time_end: str = "") -> str:
    """Assemble knowledge context for a query using Anchor & Expand.

    This is Graphite's core intelligence: finds relevant entities via semantic
    search (anchor), walks the co-occurrence graph (expand), retrieves evidence
    chunks, and assembles a three-tier context:
      - Tier 1: Key Entities (knowledge map)
      - Tier 2: Evidence (actual text chunks)
      - Tier 3: Peripheral entity summaries

    Args:
        query: The question or topic to assemble context for.
        time_start: Optional start date filter (YYYY-MM-DD). Empty for no filter.
        time_end: Optional end date filter (YYYY-MM-DD). Empty for no filter.
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                _ensure_embeddings()

                ts_start = _parse_date(time_start)
                ts_end = _parse_date(time_end)

                result = _context_manager.assemble_context(
                    query,
                    time_start=ts_start,
                    time_end=ts_end,
                )

                if not result:
                    return f"No relevant knowledge found for: {query}"

                if len(result) > MAX_RESULT_CHARS:
                    return result[:MAX_RESULT_CHARS] + "\n\n[... truncated to fit MCP limit ...]"
                return result
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Write tools (2) — require _ensure_pipeline() (LLM)
# ---------------------------------------------------------------------------

async def graphite_ingest(path: str) -> str:
    """Ingest a document or directory into the knowledge graph.

    Runs the three-pass extraction pipeline (structural parse → classify →
    tag entities) on the specified file or all .md files in a directory.
    Requires a configured LLM (set in .graphite.toml or defaults to Ollama).

    The graph is auto-saved after ingestion.

    Args:
        path: Absolute or project-relative path to a file or directory.
    """
    async with _acquire_write():
        try:
            def _run():
                global _graph_dirty
                _ensure_pipeline()

                p = Path(path)
                if not p.is_absolute():
                    p = _project_root / p
                p = p.resolve()

                if not p.exists():
                    return f"ERROR: Path does not exist: {p}"

                if p.is_file():
                    # If file was previously ingested, use update path (idempotent)
                    existing_hash = _kg.get_document_hash(str(p))
                    if existing_hash is not None:
                        update_result = _pipeline.update_document(p)
                        if update_result.action == "unchanged":
                            return f"Document unchanged (same content hash): {p.name}"
                        # Format update result
                        lines = [f"Document Updated: {p.name}"]
                        lines.append(
                            f"  Removed: {update_result.chunks_removed} chunks, "
                            f"{update_result.edges_removed} edges, "
                            f"{update_result.entities_removed} entities orphaned"
                        )
                        if update_result.ingestion_result:
                            ir = update_result.ingestion_result
                            ent_count = ir.entities_created + ir.entities_linked
                            lines.append(
                                f"  Re-ingested: {ir.chunks_tagged} chunks, "
                                f"{ent_count} entities, {ir.edges_created} edges "
                                f"({update_result.duration_seconds:.1f}s)"
                            )
                        if update_result.errors:
                            for err in update_result.errors:
                                lines.append(f"  ⚠ {err}")
                        _graph_dirty = True
                        _invalidate_caches()
                        _auto_save()
                        lines.append("Graph saved.")
                        return "\n".join(lines)
                    results = [_pipeline.ingest_file(p)]
                elif p.is_dir():
                    results = _pipeline.ingest_directory(p)
                else:
                    return f"ERROR: Path is not a file or directory: {p}"

                if not results:
                    return "No files to ingest."

                _graph_dirty = True
                _invalidate_caches()

                # Format results
                lines = ["Ingestion Results:"]
                total_entities = 0
                total_edges = 0
                total_chunks = 0
                for r in results:
                    doc_name = Path(r.source_document).name
                    ent_count = r.entities_created + r.entities_linked
                    lines.append(
                        f"  {doc_name}: {r.status} — "
                        f"{r.chunks_tagged} chunks, {ent_count} entities, "
                        f"{r.edges_created} edges ({r.duration_seconds:.1f}s)"
                    )
                    if r.errors:
                        for err in r.errors:
                            lines.append(f"    ⚠ {err}")
                    total_entities += ent_count
                    total_edges += r.edges_created
                    total_chunks += r.chunks_tagged

                lines.append(
                    f"\nTotal: {total_chunks} chunks, {total_entities} entities, "
                    f"{total_edges} edges across {len(results)} file(s)"
                )

                # Auto-save
                _auto_save()
                lines.append("Graph saved.")

                # Lightweight reflection post-ingest (async, fire-and-forget)
                if _config and _config.lightweight_reflection_on_ingest:
                    lines.append("Reflection scheduled (background).")
                    _schedule_reflection()

                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def graphite_ingest_text(
    text: str,
    source_id: str,
    category: str = "Episodic",
) -> str:
    """Ingest raw text into the knowledge graph (no file on disk required).

    Runs the same three-pass extraction pipeline as graphite_ingest but accepts
    text directly. Ideal for content from external sources (Slack, email,
    calendar, Notion, etc.) passed through via other MCP servers.

    Content hashing makes this idempotent — re-calling with the same text
    and source_id returns "unchanged" without re-processing.

    Args:
        text: Raw document content (meeting transcript, message thread, email body, etc.).
        source_id: Stable unique identifier for deduplication and updates
                   (e.g. "slack://C0123-thread-1234", "gmail://msg-abc123").
        category: Memory category — "Episodic" (meetings, conversations, events),
                  "Semantic" (people, facts, profiles), or "Procedural"
                  (projects, tasks, how-to). Defaults to "Episodic".
    """
    async with _acquire_write():
        try:
            def _run():
                global _graph_dirty
                _ensure_pipeline()

                # Validate category
                valid_categories = {"Episodic", "Semantic", "Procedural"}
                if category not in valid_categories:
                    return (
                        f"ERROR: Invalid category '{category}'. "
                        f"Must be one of: {', '.join(sorted(valid_categories))}"
                    )

                if not text.strip():
                    return "ERROR: Text is empty."

                if not source_id.strip():
                    return "ERROR: source_id is required."

                # Check content hash for idempotency
                from graphite.ingestion.pipeline import _compute_text_hash

                new_hash = _compute_text_hash(text)
                old_hash = _kg.get_document_hash(source_id)

                if old_hash is not None and old_hash == new_hash:
                    return f"Document unchanged (same content hash): {source_id}"

                # If previously ingested with different content, update
                if old_hash is not None:
                    update_result = _pipeline.update_text(
                        text, source_id, category
                    )
                    if update_result.action == "unchanged":
                        return f"Document unchanged (same content hash): {source_id}"

                    lines = [f"Document Updated: {source_id}"]
                    lines.append(
                        f"  Removed: {update_result.chunks_removed} chunks, "
                        f"{update_result.edges_removed} edges, "
                        f"{update_result.entities_removed} entities orphaned"
                    )
                    if update_result.ingestion_result:
                        ir = update_result.ingestion_result
                        ent_count = ir.entities_created + ir.entities_linked
                        lines.append(
                            f"  Re-ingested: {ir.chunks_tagged} chunks, "
                            f"{ent_count} entities, {ir.edges_created} edges "
                            f"({update_result.duration_seconds:.1f}s)"
                        )
                    if update_result.errors:
                        for err in update_result.errors:
                            lines.append(f"  Warning: {err}")
                    _graph_dirty = True
                    _invalidate_caches()
                    _auto_save()
                    lines.append("Graph saved.")
                    return "\n".join(lines)

                # New document — ingest fresh
                ingestion_result = _pipeline.ingest_text(
                    text, source_id, category
                )

                if ingestion_result.status == "failed":
                    errors = "; ".join(ingestion_result.errors) if ingestion_result.errors else "unknown"
                    return f"ERROR: Ingestion failed for {source_id}: {errors}"

                _graph_dirty = True
                _invalidate_caches()

                ent_count = (
                    ingestion_result.entities_created
                    + ingestion_result.entities_linked
                )
                lines = [
                    f"Ingested: {source_id}",
                    f"  {ingestion_result.chunks_tagged} chunks, "
                    f"{ent_count} entities, {ingestion_result.edges_created} edges "
                    f"({ingestion_result.duration_seconds:.1f}s)",
                ]

                if ingestion_result.errors:
                    for err in ingestion_result.errors:
                        lines.append(f"  Warning: {err}")

                _auto_save()
                lines.append("Graph saved.")

                # Lightweight reflection post-ingest (async, fire-and-forget)
                if _config and _config.lightweight_reflection_on_ingest:
                    lines.append("Reflection scheduled (background).")
                    _schedule_reflection()

                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def graphite_reingest() -> str:
    """Clear the graph and re-ingest all documents from the memory root.

    This is a destructive operation — the existing graph is discarded and
    rebuilt from scratch by re-running the extraction pipeline on all
    documents. Requires a configured LLM.
    """
    async with _acquire_write():
        try:
            def _run():
                global _kg, _graph_initialized, _graph_dirty
                global _embedding_manager, _context_manager, _pipeline

                _ensure_config()

                # Reset everything
                _kg = PyKnowledgeGraph(str(_graph_root))
                _embedding_manager = None
                _context_manager = None
                _pipeline = None
                _graph_initialized = True
                _graph_dirty = False

                # Re-initialize pipeline with fresh graph
                _ensure_pipeline()

                # Determine memory root
                memory_root = _config.memory_root
                if not memory_root.is_absolute():
                    memory_root = _project_root / memory_root

                if not memory_root.is_dir():
                    return f"Memory root not found: {memory_root}"

                results = _pipeline.ingest_directory(memory_root)

                if not results:
                    return "No documents found in memory root."

                _graph_dirty = True

                total_entities = sum(r.entities_created + r.entities_linked for r in results)
                total_edges = sum(r.edges_created for r in results)
                total_chunks = sum(r.chunks_tagged for r in results)

                # Full invalidation after reingest — graph was rebuilt from scratch
                _invalidate_caches()
                _auto_save()

                return (
                    f"Re-ingestion complete: {len(results)} files, "
                    f"{total_chunks} chunks, {total_entities} entities, "
                    f"{total_edges} edges. Graph saved."
                )
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Phase 6: Incremental update tools (2)
# ---------------------------------------------------------------------------

async def graphite_update_document(path: str) -> str:
    """Update a previously-ingested document in the knowledge graph.

    Compares the file's content hash against the stored hash. If unchanged,
    returns immediately. If changed, removes old artifacts (chunks, edges,
    orphaned entities) and re-runs the three-pass extraction pipeline.

    The graph is auto-saved after update.

    Args:
        path: Absolute or project-relative path to the document file.
    """
    async with _acquire_write():
        try:
            def _run():
                global _graph_dirty
                _ensure_pipeline()

                p = Path(path)
                if not p.is_absolute():
                    p = _project_root / p
                p = p.resolve()

                if not p.exists():
                    return f"ERROR: File does not exist: {p}"
                if not p.is_file():
                    return f"ERROR: Path is not a file: {p}"

                update_result = _pipeline.update_document(p)

                if update_result.action == "unchanged":
                    return f"Document unchanged (same content hash): {p.name}"

                if update_result.action == "failed":
                    errors = "; ".join(update_result.errors) if update_result.errors else "unknown"
                    return f"ERROR: Update failed for {p.name}: {errors}"

                lines = [f"Document Updated: {p.name}"]
                lines.append(
                    f"  Removed: {update_result.chunks_removed} chunks, "
                    f"{update_result.edges_removed} edges, "
                    f"{update_result.entities_removed} entities orphaned"
                )
                if update_result.ingestion_result:
                    ir = update_result.ingestion_result
                    ent_count = ir.entities_created + ir.entities_linked
                    lines.append(
                        f"  Re-ingested: {ir.chunks_tagged} chunks, "
                        f"{ent_count} entities, {ir.edges_created} edges "
                        f"({update_result.duration_seconds:.1f}s)"
                    )
                if update_result.errors:
                    for err in update_result.errors:
                        lines.append(f"  ⚠ {err}")

                _graph_dirty = True
                _invalidate_caches()
                _auto_save()
                lines.append("Graph saved.")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def graphite_remove_document(path: str) -> str:
    """Remove a document and all its artifacts from the knowledge graph.

    Cascade-removes the document's chunks, co-occurrence edges, and any
    entities that were solely sourced from this document. Entities shared
    with other documents are updated (source list trimmed) but kept.

    The graph is auto-saved after removal.

    Args:
        path: Absolute or project-relative path to the document.
    """
    async with _acquire_write():
        try:
            def _run():
                global _graph_dirty
                _ensure_pipeline()

                p = Path(path)
                if not p.is_absolute():
                    p = _project_root / p
                p = p.resolve()

                result = _pipeline.remove_document(p)

                if result.action == "failed":
                    errors = "; ".join(result.errors) if result.errors else "unknown"
                    return f"ERROR: Removal failed for {p.name}: {errors}"

                total_removed = result.chunks_removed + result.edges_removed + result.entities_removed
                if total_removed == 0:
                    return f"No artifacts found for document: {p.name}"

                lines = [f"Document Removed: {p.name}"]
                lines.append(f"  Chunks removed: {result.chunks_removed}")
                lines.append(f"  Edges removed: {result.edges_removed}")
                lines.append(f"  Entities removed (orphaned): {result.entities_removed}")
                lines.append(f"  Entities updated (trimmed): {result.entities_updated}")

                _graph_dirty = True
                _invalidate_caches()
                _auto_save()
                lines.append("Graph saved.")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Phase 5: Reflection & consolidation tools (3)
# ---------------------------------------------------------------------------

async def graphite_reflect(mode: str = "full") -> str:
    """Run reflection and consolidation on the knowledge graph.

    Performs entity merging, orphan cleanup, decay scoring, and optionally
    LLM-powered synthesis. The graph is auto-saved after reflection.

    Args:
        mode: "full" (all operations), "light" or "lightweight" (orphan cleanup only),
              or "merge" (find and execute merges only).
    """
    async with _acquire_write():
        try:
            def _run():
                global _graph_dirty
                _ensure_graph()
                _ensure_config()

                from graphite.reflection.consolidator import Consolidator
                from graphite.reflection.synthesizer import Synthesizer

                llm = _get_llm_client()

                consolidator = Consolidator(
                    knowledge_graph=_kg,
                    embedding_manager=_embedding_manager,
                    config=_config,
                    llm_client=llm,
                )

                lines = ["Reflection Results:"]

                if mode in ("light", "lightweight"):
                    result = consolidator.run_lightweight()
                    lines.append(f"  Orphans removed: {result.orphans_removed}")
                elif mode == "merge":
                    candidates = consolidator.find_merge_candidates()
                    confirmed = consolidator.confirm_merges(candidates)
                    merges = consolidator.execute_merges(confirmed)
                    lines.append(f"  Merge candidates found: {len(candidates)}")
                    lines.append(f"  Merges executed: {merges}")
                else:
                    # Full mode
                    result = consolidator.run_full()
                    lines.append(f"  Merge candidates found: {result.merges_found}")
                    lines.append(f"  Merges executed: {result.merges_executed}")
                    lines.append(f"  Orphans removed: {result.orphans_removed}")
                    lines.append(f"  Entities decayed: {result.entities_decayed}")
                    lines.append(f"  Low-access entities: {result.entities_flagged_low}")
                    lines.append(f"  Edges deduplicated: {result.edges_deduplicated}")
                    lines.append(f"  Edges pruned: {result.edges_pruned}")

                    # Run synthesis if LLM available
                    if llm is not None:
                        synthesizer = Synthesizer(
                            knowledge_graph=_kg,
                            llm_client=llm,
                            embedding_manager=_embedding_manager,
                            config=_config,
                        )
                        syn_result = synthesizer.run()
                        lines.append(f"  Entities synthesized: {syn_result.entities_synthesized}")
                        lines.append(f"  Embeddings refreshed: {syn_result.embeddings_invalidated}")
                        lines.append(f"  Edges recalculated: {syn_result.edges_updated}")
                        if syn_result.errors:
                            for err in syn_result.errors:
                                lines.append(f"  Warning: {err}")

                    if result.errors:
                        for err in result.errors:
                            lines.append(f"  Warning: {err}")

                    lines.append(f"  Duration: {result.duration_seconds:.1f}s")

                _graph_dirty = True
                _invalidate_caches()
                _auto_save()
                lines.append("Graph saved.")

                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def graphite_forget(entity: str) -> str:
    """Permanently remove an entity from the knowledge graph.

    Removes the entity and all its co-occurrence edges. Chunks that
    mention it are kept (they may reference other entities). The graph
    is auto-saved after removal.

    Args:
        entity: Entity name or UUID to remove.
    """
    async with _acquire_write():
        try:
            def _run():
                global _graph_dirty
                _ensure_graph()

                ent = _resolve_entity(entity)
                entity_id = ent["id"]
                name = ent["canonical_name"]

                removed = _kg.remove_entity(entity_id)
                if not removed:
                    return f"ERROR: Failed to remove entity '{name}' ({entity_id})."

                _graph_dirty = True
                _invalidate_caches(entity_ids=[entity_id])
                _auto_save()

                return f"Removed entity '{name}' ({entity_id}). Graph saved."
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def graphite_review() -> str:
    """Show merge candidates for human review.

    Finds entity pairs that might be duplicates based on alias overlap
    and embedding similarity. Shows candidates with confidence below
    the auto-approve threshold (< 0.95) that need human judgment.
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_graph()
                _ensure_config()

                from graphite.reflection.consolidator import Consolidator

                consolidator = Consolidator(
                    knowledge_graph=_kg,
                    embedding_manager=_embedding_manager,
                    config=_config,
                )

                candidates = consolidator.find_merge_candidates()

                # Filter to review zone (below auto-approve threshold)
                review = [c for c in candidates if c.confidence < 0.95]

                if not review:
                    return "No merge candidates for review. Graph is clean."

                lines = [f"Merge Review Queue ({len(review)} candidate(s)):"]
                for i, c in enumerate(review, 1):
                    lines.append(
                        f"\n  {i}. **{c.keep_name}** ← {c.merge_name}"
                        f"\n     Confidence: {c.confidence:.2f} — {c.reason}"
                    )

                lines.append(
                    "\nUse graphite_reflect(mode='merge') to execute merges, "
                    "or graphite_forget to remove specific entities."
                )
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Agent context injection tool
# ---------------------------------------------------------------------------


async def graphite_agent_context(
    situation: str,
    depth: str = "brief",
    max_entities: int = 0,
    max_events: int = 0,
    time_start: str = "",
    time_end: str = "",
) -> str:
    """Get structured knowledge context for agent system prompt injection.

    Designed for AI agent reasoning loops. Returns entities, relationships,
    pending items, and recent events relevant to the current situation.
    No LLM calls — fast enough for every agent turn (~200ms warm).

    Two depths:
      - "brief": Entity-level context, ~100-200 tokens. Every turn.
      - "full": Includes evidence chunks, ~1000-5000 tokens. Deep reasoning.

    Args:
        situation: The agent's current task or situation description.
        depth: "brief" or "full" (default: "brief").
        max_entities: Override entity limit (0 = use config default).
        max_events: Override event limit for full mode (0 = use config default).
        time_start: Optional lower bound date filter (YYYY-MM-DD).
        time_end: Optional upper bound date filter (YYYY-MM-DD).
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_agent_assembler()

                ts_start = _parse_date(time_start) if time_start else None
                ts_end = _parse_date(time_end) if time_end else None

                ctx = _agent_assembler.assemble(
                    situation=situation,
                    depth=depth,
                    max_entities=max_entities if max_entities > 0 else None,
                    max_events=max_events if max_events > 0 else None,
                    time_start=ts_start,
                    time_end=ts_end,
                )

                result = ctx.to_dict()
                result["injection_text"] = ctx.to_injection_text()

                output = json.dumps(result, indent=2)
                if len(output) > MAX_RESULT_CHARS:
                    output = output[:MAX_RESULT_CHARS] + "\n... (truncated)"
                return output
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# User profile + session ingestion tools
# ---------------------------------------------------------------------------


async def graphite_user_profile() -> str:
    """Build a user profile from conversation-derived knowledge.

    Queries the knowledge graph for Preference, Goal, Pattern, Skill,
    Project, and Concept entities extracted from ingested conversation
    sessions. Returns a structured narrative about the user.

    This tool requires that conversation sessions have been ingested
    first (via graphite_ingest_sessions or `graphite ingest-sessions`).
    """
    async with _acquire_read():
        try:
            def _run():
                _ensure_agent_assembler()

                profile = _agent_assembler.assemble_user_profile()
                result = profile.to_dict()

                output = json.dumps(result, indent=2)
                if len(output) > MAX_RESULT_CHARS:
                    output = output[:MAX_RESULT_CHARS] + "\n... (truncated)"
                return output
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


async def graphite_ingest_sessions(
    project_filter: str = "",
    since: str = "",
) -> str:
    """Ingest Claude Code conversation transcripts into the knowledge graph.

    Discovers JSONL session files from ~/.claude/projects/ and runs them
    through the extraction pipeline (parse -> classify -> tag -> graph write).

    This is the primary way to build a user profile from conversation history.
    Run this before using graphite_user_profile.

    Args:
        project_filter: Only ingest sessions from projects matching this
                        name substring (e.g. "Graphite"). Empty = all projects.
        since: Only ingest sessions modified after this date (YYYY-MM-DD).
               Empty = all sessions.
    """
    async with _acquire_write():
        try:
            def _run():
                global _graph_dirty
                _ensure_pipeline()

                results = _pipeline.ingest_all_sessions(
                    claude_dir=_config.claude_data_dir,
                    project_filter=project_filter if project_filter else None,
                    since=since if since else None,
                )

                _graph_dirty = True
                _invalidate_caches()
                _auto_save()

                # Summarize results
                total = len(results)
                complete = sum(1 for r in results if r.status == "complete" and r.chunks_tagged > 0)
                skipped = sum(1 for r in results if r.status == "complete" and r.chunks_tagged == 0)
                failed = sum(1 for r in results if r.status in ("failed", "partial"))
                total_chunks = sum(r.chunks_tagged for r in results)
                total_entities = sum(r.entities_created + r.entities_linked for r in results)
                total_edges = sum(r.edges_created for r in results)
                total_time = sum(r.duration_seconds for r in results)

                lines = [
                    f"Session ingestion complete: {total} sessions processed",
                    f"  Ingested: {complete}",
                    f"  Skipped (unchanged): {skipped}",
                    f"  Failed: {failed}",
                    f"  Total chunks: {total_chunks}",
                    f"  Total entities: {total_entities}",
                    f"  Total edges: {total_edges}",
                    f"  Duration: {total_time:.1f}s",
                ]

                errors = [e for r in results for e in r.errors]
                if errors:
                    lines.append(f"\nErrors ({len(errors)}):")
                    for err in errors[:10]:
                        lines.append(f"  - {err}")
                    if len(errors) > 10:
                        lines.append(f"  ... and {len(errors) - 10} more")

                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Workflow templates — resources + discovery tool
# ---------------------------------------------------------------------------
WORKFLOWS_DIR = Path(__file__).parent / "workflows"


@mcp.resource("graphite://workflow/{name}")
def read_workflow(name: str) -> str:
    """Read a workflow template by name."""
    workflow_file = WORKFLOWS_DIR / f"{name}.md"
    if not workflow_file.exists():
        available = [f.stem for f in sorted(WORKFLOWS_DIR.glob("*.md")) if f.stem != "README"]
        return (
            f"Workflow '{name}' not found. "
            f"Available workflows: {', '.join(available)}"
        )
    return workflow_file.read_text(encoding="utf-8")


async def graphite_workflows() -> str:
    """List available workflow templates for multi-source data ingestion.

    Workflows teach Claude how to read from external MCP servers (Slack,
    email, calendar, Notion) and ingest the content into Graphite using
    graphite_ingest_text(). Each workflow describes the step-by-step process
    for a specific data source.

    To use a workflow, read it via the resource URI (e.g.
    graphite://workflow/ingest-slack) or just ask Claude to ingest from
    that source — it will follow the workflow automatically.
    """
    if not WORKFLOWS_DIR.exists():
        return "No workflows directory found."

    workflows = sorted(WORKFLOWS_DIR.glob("*.md"))
    if not workflows:
        return "No workflow templates found."

    lines = ["Available Graphite Workflows:"]
    for f in workflows:
        if f.stem == "README":
            continue
        # Read first line after the "# Workflow:" header for description
        try:
            content = f.read_text(encoding="utf-8")
            # Extract title from first heading
            for line in content.splitlines():
                if line.startswith("# Workflow:"):
                    title = line.replace("# Workflow:", "").strip()
                    break
            else:
                title = f.stem.replace("-", " ").title()
        except Exception:
            title = f.stem.replace("-", " ").title()

        lines.append(f"  - {f.stem}: {title}")
        lines.append(f"    Resource: graphite://workflow/{f.stem}")

    lines.append("")
    lines.append(
        "Read a workflow with its resource URI, or ask Claude to "
        "ingest from a specific source."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timestamp formatting helper
# ---------------------------------------------------------------------------
def _format_timestamp(ts) -> str:
    """Format a Unix timestamp to YYYY-MM-DD, or 'unknown'."""
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return "unknown"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="graphite-mcp",
        description="Graphite MCP Server — knowledge graph memory for LLMs",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to the project root — used to tag ingested content and to "
             "locate .graphite.toml (default: current directory)",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=None,
        help="Directory holding .graphite/graph.msgpack. Overrides config; "
             "defaults to ~ for a single global graph across projects.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("graphite").setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    global _project_root, _graph_root_override

    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        print(f"ERROR: {project_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    _project_root = project_root
    if args.graph_root is not None:
        _graph_root_override = args.graph_root.expanduser().resolve()

    log.info(
        "Graphite MCP server starting (project: %s, graph loads on first tool call)",
        project_root,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
