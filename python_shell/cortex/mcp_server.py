"""
Cortex MCP Server — Knowledge graph memory tools for LLMs.

Provides 16 tools for knowledge graph interaction: entity lookup,
co-occurrence analysis, semantic search, context assembly, document
ingestion, and graph management.

The server loads a persisted graph from .cortex/graph.msgpack on first
tool call. Ingestion (which requires LLM calls) is only triggered
explicitly via cortex_ingest or cortex_reingest.

Usage:
    cortex-mcp --project-root /path/to/project
    cortex-mcp --project-root /path/to/project --verbose
"""

import sys
import asyncio
import argparse
import json
import logging
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
    from cortex.semantic_engine import PyKnowledgeGraph
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
    instructions="Knowledge graph memory system for LLMs — entity lookup, "
    "co-occurrence analysis, semantic search, evidence retrieval, "
    "and document ingestion.",
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_kg: Optional[PyKnowledgeGraph] = None
_embedding_manager = None  # Lazy: cortex.embeddings.EmbeddingManager
_context_manager = None    # Lazy: cortex.context.MemoryContextManager
_pipeline = None           # Lazy: cortex.ingestion.pipeline.IngestionPipeline
_config = None             # Lazy: cortex.config.CortexConfig
_project_root: Optional[Path] = None
_graph_initialized: bool = False
_graph_dirty: bool = False
_tool_lock: Optional[asyncio.Lock] = None

MAX_RESULT_CHARS = 60_000  # Safety cap for MCP tool results


# ---------------------------------------------------------------------------
# Lock helper
# ---------------------------------------------------------------------------
def _get_lock() -> asyncio.Lock:
    """Get or create the global tool lock (must be called from async context)."""
    global _tool_lock
    if _tool_lock is None:
        _tool_lock = asyncio.Lock()
    return _tool_lock


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------
def _ensure_config() -> None:
    """Load CortexConfig from .cortex.toml or use defaults."""
    global _config
    if _config is not None:
        return
    from cortex.config import CortexConfig

    toml_path = _project_root / ".cortex.toml"
    if toml_path.exists():
        try:
            _config = CortexConfig.from_toml(toml_path)
            log.info("Loaded config from %s", toml_path)
        except Exception as e:
            log.warning("Failed to load .cortex.toml: %s — using defaults", e)
            _config = CortexConfig()
    else:
        _config = CortexConfig()


def _ensure_graph() -> None:
    """Load graph from .cortex/graph.msgpack or create empty PyKnowledgeGraph."""
    global _kg, _graph_initialized
    if _graph_initialized:
        return
    if _project_root is None:
        raise RuntimeError("Project root not set")

    _ensure_config()
    graph_file = _project_root / ".cortex" / "graph.msgpack"

    if graph_file.exists():
        log.info("Loading persisted graph from %s", graph_file)
        _kg = PyKnowledgeGraph.load(str(_project_root))
        log.info("Graph loaded")
    else:
        log.info("No persisted graph found — starting with empty graph")
        _kg = PyKnowledgeGraph(str(_project_root))

    _graph_initialized = True


def _ensure_embeddings() -> None:
    """Lazily initialize EmbeddingManager and MemoryContextManager."""
    global _embedding_manager, _context_manager
    if _embedding_manager is not None:
        return
    _ensure_graph()
    _ensure_config()

    log.info("Initializing embedding manager (first semantic search call)...")
    from cortex.embeddings import EmbeddingManager
    from cortex.context import MemoryContextManager

    _embedding_manager = EmbeddingManager()
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
    from cortex.ingestion.pipeline import IngestionPipeline

    # Create LLM client from config
    if _config.llm_provider == "mlx":
        from cortex.llm import MLXClient
        llm_client = MLXClient(model=_config.llm_model)
    else:
        from cortex.llm import OllamaClient
        llm_client = OllamaClient(model=_config.llm_model)

    _pipeline = IngestionPipeline(
        knowledge_graph=_kg,
        llm_client=llm_client,
        embedding_manager=_embedding_manager,
        config=_config,
    )
    log.info("Ingestion pipeline ready (provider=%s, model=%s)",
             _config.llm_provider, _config.llm_model)


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
def _auto_save() -> None:
    """Save the graph if it has been modified."""
    global _graph_dirty
    if not _graph_dirty or _kg is None or _project_root is None:
        return
    try:
        _kg.save(str(_project_root))
        _graph_dirty = False
        log.info("Graph auto-saved")
    except Exception as e:
        log.error("Auto-save failed: %s", e)


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


# ---------------------------------------------------------------------------
# Read-only tools (9) — require only _ensure_graph()
# ---------------------------------------------------------------------------

@mcp.tool()
async def cortex_status() -> str:
    """Get knowledge graph statistics and readiness status. Call this first to verify Cortex is ready."""
    async with _get_lock():
        try:
            def _run():
                _ensure_graph()
                stats = _get_stats()
                lines = [
                    "Cortex Knowledge Graph Status",
                    f"  Project root: {_project_root}",
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
                graph_file = _project_root / ".cortex" / "graph.msgpack"
                lines.append(f"  Persisted: {graph_file.exists()}")
                lines.append(f"  Embeddings loaded: {_embedding_manager is not None}")
                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def get_knowledge_map(max_entities: int = 50) -> str:
    """Get a PageRank-ranked knowledge map of the most important entities.

    Returns entities grouped by type with their co-occurrence connections,
    ordered by importance score. Use this to understand the knowledge graph
    structure before diving into specific entities.

    Args:
        max_entities: Maximum number of entities to include (default 50).
    """
    async with _get_lock():
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


@mcp.tool()
async def get_cooccurrences(entity: str) -> str:
    """Get entities that co-occur with a given entity in document chunks.

    Shows which entities appear alongside the target in the same text chunks,
    with frequency counts and most recent timestamps.

    Args:
        entity: Entity name or UUID.
    """
    async with _get_lock():
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


@mcp.tool()
async def get_entity_mentions(entity: str, limit: int = 20) -> str:
    """Get all chunks where an entity is tagged, ordered by time.

    Returns the actual text chunks from documents where this entity
    was identified, most recent first.

    Args:
        entity: Entity name or UUID.
        limit: Maximum number of chunks to return (default 20).
    """
    async with _get_lock():
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


@mcp.tool()
async def get_key_entities(limit: int = 20, entity_type: str = "") -> str:
    """Get the most important entities ranked by PageRank.

    Higher-ranked entities are more central to the knowledge graph — they
    co-occur with many other entities across many documents.

    Args:
        limit: Number of entities to return (default 20).
        entity_type: Optional filter by type (e.g. "Person", "Project"). Empty for all.
    """
    async with _get_lock():
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


@mcp.tool()
async def get_entity_profile(entity: str) -> str:
    """Get a full profile for an entity: type, aliases, co-occurrences, and recent chunks.

    Args:
        entity: Entity name or UUID.
    """
    async with _get_lock():
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


@mcp.tool()
async def get_timeline(entity: str, limit: int = 30) -> str:
    """Get a chronological timeline of chunks mentioning an entity.

    Returns chunks sorted oldest-to-newest, useful for understanding
    how discussions about an entity evolved over time.

    Args:
        entity: Entity name or UUID.
        limit: Maximum number of chunks (default 30).
    """
    async with _get_lock():
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


@mcp.tool()
async def get_evidence(entity_a: str, entity_b: str, limit: int = 15) -> str:
    """Get chunks where two entities co-occur — evidence of their relationship.

    Returns the actual text from documents where both entities were
    mentioned together, useful for understanding how they relate.

    Args:
        entity_a: First entity name or UUID.
        entity_b: Second entity name or UUID.
        limit: Maximum number of chunks (default 15).
    """
    async with _get_lock():
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


@mcp.tool()
async def get_entity_summary(entity: str) -> str:
    """Get a one-line summary of an entity.

    Returns: "Name (Type): co-occurs with X, Y in N chunks"

    Args:
        entity: Entity name or UUID.
    """
    async with _get_lock():
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
# Semantic search tools (2) — require _ensure_embeddings()
# ---------------------------------------------------------------------------

@mcp.tool()
async def find_relevant_entities(query: str, top_n: int = 10) -> str:
    """Find entities most relevant to a query using semantic search.

    Uses embedding similarity combined with PageRank re-ranking to find
    entities whose context is semantically similar to the query.

    Args:
        query: Natural language description of what you're looking for.
        top_n: Number of results to return (default 10).
    """
    async with _get_lock():
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


@mcp.tool()
async def assemble_memory(query: str, time_start: str = "", time_end: str = "") -> str:
    """Assemble knowledge context for a query using Anchor & Expand.

    This is Cortex's core intelligence: finds relevant entities via semantic
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
    async with _get_lock():
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

@mcp.tool()
async def cortex_ingest(path: str) -> str:
    """Ingest a document or directory into the knowledge graph.

    Runs the three-pass extraction pipeline (structural parse → classify →
    tag entities) on the specified file or all .md files in a directory.
    Requires a configured LLM (set in .cortex.toml or defaults to Ollama).

    The graph is auto-saved after ingestion.

    Args:
        path: Absolute or project-relative path to a file or directory.
    """
    async with _get_lock():
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
                    results = [_pipeline.ingest_file(p)]
                elif p.is_dir():
                    results = _pipeline.ingest_directory(p)
                else:
                    return f"ERROR: Path is not a file or directory: {p}"

                if not results:
                    return "No files to ingest."

                _graph_dirty = True

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

                return "\n".join(lines)
            return await anyio.to_thread.run_sync(_run)
        except Exception as e:
            return f"ERROR: {e}"


@mcp.tool()
async def cortex_reingest() -> str:
    """Clear the graph and re-ingest all documents from the memory root.

    This is a destructive operation — the existing graph is discarded and
    rebuilt from scratch by re-running the extraction pipeline on all
    documents. Requires a configured LLM.
    """
    async with _get_lock():
        try:
            def _run():
                global _kg, _graph_initialized, _graph_dirty
                global _embedding_manager, _context_manager, _pipeline

                _ensure_config()

                # Reset everything
                _kg = PyKnowledgeGraph(str(_project_root))
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
# Phase 5 stubs (3)
# ---------------------------------------------------------------------------

@mcp.tool()
async def cortex_reflect() -> str:
    """Run reflection and consolidation on the knowledge graph.

    (Phase 5 — not yet implemented)
    Will perform: entity merging, decay scoring, stale evidence detection,
    and higher-order insight synthesis.
    """
    return "cortex_reflect is not yet implemented (Phase 5). Coming soon."


@mcp.tool()
async def cortex_forget(entity: str) -> str:
    """Mark an entity as decayed/archived in the knowledge graph.

    (Phase 5 — not yet implemented)
    Will mark the entity for decay, reducing its PageRank weight
    and eventually archiving it from active queries.

    Args:
        entity: Entity name or UUID to forget.
    """
    return f"cortex_forget is not yet implemented (Phase 5). Entity '{entity}' unchanged."


@mcp.tool()
async def cortex_review() -> str:
    """Present the human review queue for ambiguous entity disambiguations.

    (Phase 5 — not yet implemented)
    Will show entities with disambiguation confidence between 0.70–0.85
    that need human confirmation to merge or keep separate.
    """
    return "cortex_review is not yet implemented (Phase 5). Coming soon."


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
        prog="cortex-mcp",
        description="Cortex MCP Server — knowledge graph memory for LLMs",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Path to the project root (default: current directory)",
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
    log.info(
        "Cortex MCP server starting (project: %s, graph loads on first tool call)",
        project_root,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
