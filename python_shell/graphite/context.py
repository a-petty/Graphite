import collections
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tiktoken
import logging

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

# Re-ranking weights for combining embedding similarity with PageRank.
SIMILARITY_WEIGHT = 0.80
PAGERANK_WEIGHT = 0.20

# Known context window sizes (tokens). Used to set max_tokens automatically.
# Conservative utilization: use 60% of window (leave room for system prompt + response).
MODEL_CONTEXT_WINDOWS = {
    # Cloud models
    "claude": 200_000,
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "gpt-4": 128_000,
    "gpt-4o": 128_000,
    "gemini": 1_000_000,
    "gemini-pro": 1_000_000,
    # Local models (Ollama)
    "deepseek-coder": 128_000,
    "deepseek-r2-distill-qwen-32b": 128_000,
    "codellama": 16_000,
    "llama3": 8_000,
    "mistral": 32_000,
    "qwen2.5-coder": 128_000,
    # Local models (MLX — HuggingFace model IDs)
    "mlx-community/deepseek-coder": 128_000,
    "mlx-community/mistral": 32_000,
    "mlx-community/codellama": 16_000,
    "mlx-community/llama-3": 8_000,
    "mlx-community/qwen2.5-coder": 128_000,
}

CONTEXT_UTILIZATION = 0.60  # Use 60% of window for context (rest for system prompt + response)
DEFAULT_CONTEXT_WINDOW = 100_000  # Fallback for unknown models


def resolve_max_tokens(model: str, explicit_max_tokens: Optional[int]) -> int:
    """Determine max_tokens from model name, unless explicitly overridden."""
    if explicit_max_tokens is not None:
        return explicit_max_tokens

    # Try exact match, then prefix match (e.g., "deepseek-coder:7b" matches "deepseek-coder")
    model_lower = model.lower()
    window = MODEL_CONTEXT_WINDOWS.get(model_lower)
    if window is None:
        for prefix, w in MODEL_CONTEXT_WINDOWS.items():
            if model_lower.startswith(prefix):
                window = w
                break
    if window is None:
        window = DEFAULT_CONTEXT_WINDOW

    return int(window * CONTEXT_UTILIZATION)



# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: MemoryContextManager — Knowledge Graph Context Assembly
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryContextParams:
    """Adaptive context assembly parameters for knowledge graph queries."""
    tier1_tokens: int           # Knowledge Map budget
    tier2_tokens: int           # Evidence Chunks budget
    tier3_tokens: int           # Entity Summaries budget
    anchor_count: int           # 3-10
    neighborhood_max_hops: int  # 2-3
    neighborhood_max_entities: int  # 10-40


class MemoryContextManager:
    """Assembles knowledge graph context for LLM queries.

    Uses the Anchor & Expand strategy adapted for knowledge graphs:
      1. Anchor: find entities most relevant to the query via embedding similarity
      2. Expand: BFS through co-occurrence edges to find related entities/chunks
      3. Assemble: build three-tier context (map, evidence, summaries)

    The graph is a retrieval index — it finds the right chunks.
    The LLM reads the evidence and draws its own conclusions.
    """

    def __init__(
        self,
        knowledge_graph,              # PyKnowledgeGraph
        embedding_manager: EmbeddingManager,
        config=None,                  # Optional[GraphiteConfig]
        model: str = "gpt-4",
        max_tokens: Optional[int] = None,
    ):
        self.kg = knowledge_graph
        self.embedding_manager = embedding_manager

        # Import here to avoid circular dependency at module level
        if config is None:
            from .config import GraphiteConfig
            config = GraphiteConfig()
        self.config = config

        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.max_tokens = resolve_max_tokens(model, max_tokens)
        self._entities_embedded = False
        # Cache: entity_id → {canonical_name, entity_type, source_documents, ...}
        self._entity_cache: Dict[str, Dict] = {}
        # Cache: entity_id → pagerank score
        self._pagerank_cache: Dict[str, float] = {}

        logger.info(
            f"MemoryContextManager initialized with model '{model}', "
            f"max_tokens={self.max_tokens}"
        )

    # ── Public API ──

    def invalidate_caches(self) -> None:
        """Clear all internal caches after graph mutations."""
        self._entities_embedded = False
        self._entity_cache.clear()
        self._pagerank_cache.clear()

    def assemble_context(
        self,
        query: str,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> str:
        """Assemble three-tier knowledge context for a query.

        Args:
            query: The user's natural language query.
            time_start: Optional Unix timestamp lower bound for temporal filtering.
            time_end: Optional Unix timestamp upper bound for temporal filtering.

        Returns:
            Markdown-formatted context string with three tiers:
              - Tier 1: Key Entities (knowledge map)
              - Tier 2: Evidence (relevant chunks)
              - Tier 3: Peripheral Entities (summaries)
        """
        # Budget calculation
        total_budget = self.max_tokens
        total_budget -= self.count_tokens(query)
        total_budget -= 1000  # System prompt overhead
        if total_budget <= 0:
            return ""

        params = self._compute_adaptive_params(total_budget)

        # Ensure entity embeddings are ready
        self._ensure_entity_embeddings()

        # If no entities in graph, return empty
        if not self._pagerank_cache:
            return ""

        # Step 1: Find anchor entities
        anchors = self._find_anchors(query, params.anchor_count)
        if not anchors:
            return ""

        anchor_ids = [eid for eid, _ in anchors]

        # Step 2: Expand neighborhood
        expanded = self._expand_neighborhood(
            anchor_ids, params, time_start, time_end
        )

        # Step 3: Build tiers
        tier1 = self._build_knowledge_map(expanded, params.tier1_tokens)
        tier2, tier2_entity_ids = self._build_evidence_tier(
            expanded, params.tier2_tokens
        )
        tier3 = self._build_entity_summaries(
            expanded, tier2_entity_ids, params.tier3_tokens
        )

        # Step 4: Format
        return self._format_knowledge_context(tier1, tier2, tier3)

    # ── Budget & Params ──

    def _compute_adaptive_params(self, total_budget: int) -> MemoryContextParams:
        """Compute tier budgets and traversal params from graph size."""
        stats_json = self.kg.get_statistics()
        stats = json.loads(stats_json)
        entity_count = stats.get("entity_count", 0)
        edge_count = stats.get("edge_count", 0)
        density = edge_count / max(entity_count, 1)

        tier1_tokens = int(total_budget * self.config.tier1_budget_pct)
        tier2_tokens = int(total_budget * self.config.tier2_budget_pct)
        tier3_tokens = total_budget - tier1_tokens - tier2_tokens

        # Anchor count: scale with entity count, clamped
        anchor_count = min(
            max(self.config.anchor_count_min, entity_count // 10),
            self.config.anchor_count_max,
        )

        # Neighborhood hops: sparse graphs get deeper traversal
        if density < 3.0:
            neighborhood_max_hops = self.config.neighborhood_max_hops_sparse
        else:
            neighborhood_max_hops = self.config.neighborhood_max_hops_dense

        # Neighborhood entity cap: scale with entity count, clamped 10–40
        neighborhood_max_entities = min(
            max(10, entity_count // 3),
            self.config.neighborhood_max_entities,
        )

        params = MemoryContextParams(
            tier1_tokens=tier1_tokens,
            tier2_tokens=tier2_tokens,
            tier3_tokens=tier3_tokens,
            anchor_count=anchor_count,
            neighborhood_max_hops=neighborhood_max_hops,
            neighborhood_max_entities=neighborhood_max_entities,
        )

        logger.info(
            f"Memory context params: entities={entity_count}, density={density:.1f}, "
            f"tier1={tier1_tokens}, tier2={tier2_tokens}, tier3={tier3_tokens}, "
            f"anchors={anchor_count}, hops={neighborhood_max_hops}"
        )

        return params

    # ── Anchor Phase ──

    def _ensure_entity_embeddings(self) -> None:
        """Lazily embed all entities on first query.

        Uses compute_pagerank() to get all entity IDs and scores,
        then calls embedding_manager.embed_entities() for uncached entities.
        """
        if self._entities_embedded:
            return

        # compute_pagerank returns JSON array of [id, score] pairs
        pagerank_json = self.kg.compute_pagerank()
        pagerank_pairs = json.loads(pagerank_json)

        self._pagerank_cache.clear()
        self._entity_cache.clear()

        entities_for_embedding: List[Dict] = []
        for entity_id, score in pagerank_pairs:
            self._pagerank_cache[entity_id] = score
            entity_json = self.kg.get_entity(entity_id)
            if entity_json:
                entity_data = json.loads(entity_json)
                self._entity_cache[entity_id] = entity_data

                # Format entity_type for descriptor
                etype = entity_data.get("entity_type", "Concept")
                if isinstance(etype, dict):
                    # Custom type comes as {"Custom": "TypeName"}
                    etype = next(iter(etype.values()), "Concept")

                entities_for_embedding.append({
                    "id": entity_id,
                    "canonical_name": entity_data["canonical_name"],
                    "entity_type": etype,
                })

        if entities_for_embedding:
            self.embedding_manager.embed_entities(entities_for_embedding, self.kg)

        self._entities_embedded = True
        logger.debug(f"Embedded {len(entities_for_embedding)} entities.")

    def _find_anchors(
        self, query: str, anchor_count: int
    ) -> List[Tuple[str, float]]:
        """Find anchor entities by combining embedding similarity with PageRank.

        combined_score = similarity_weight * similarity + pagerank_weight * normalized_pagerank
        """
        entity_ids = list(self._pagerank_cache.keys())
        if not entity_ids:
            return []

        # Get similarity scores from embeddings
        # Fetch more candidates than needed for re-ranking
        candidates = self.embedding_manager.find_relevant_entities_scored(
            query, entity_ids, top_n=anchor_count * 3,
        )

        if not candidates:
            return []

        # Normalize PageRank scores
        max_pr = max(self._pagerank_cache.values()) if self._pagerank_cache else 1.0
        if max_pr == 0:
            max_pr = 1.0

        # Re-rank with combined score
        reranked: List[Tuple[str, float]] = []
        for eid, similarity in candidates:
            pr = self._pagerank_cache.get(eid, 0.0)
            normalized_pr = pr / max_pr
            combined = (
                self.config.similarity_weight * similarity
                + self.config.pagerank_weight * normalized_pr
            )
            reranked.append((eid, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        result = reranked[:anchor_count]

        logger.debug(
            f"Anchors: {[(self._entity_cache.get(eid, {}).get('canonical_name', eid), f'{score:.3f}') for eid, score in result]}"
        )
        return result

    # ── Expand Phase ──

    def _expand_neighborhood(
        self,
        anchor_ids: List[str],
        params: MemoryContextParams,
        time_start: Optional[int],
        time_end: Optional[int],
    ) -> Dict:
        """Expand anchor entities through co-occurrence graph.

        Returns a dict with:
          - entities: {entity_id: {entity_data, weight, hop}}
          - edges: [(source_id, target_id, edge_data), ...]
          - chunks: [chunk_data, ...]
        """
        all_entities: Dict[str, Dict] = {}
        all_edges: List[Tuple[str, str, Dict]] = []
        all_chunks: List[Dict] = []
        seen_chunk_ids: set = set()
        seen_edge_keys: set = set()

        # Query neighborhood for each anchor
        for anchor_id in anchor_ids:
            result_json = self.kg.query_neighborhood(
                anchor_id,
                params.neighborhood_max_hops,
                time_start,
                time_end,
            )
            result = json.loads(result_json)

            for entity in result.get("entities", []):
                eid = entity["id"]
                if eid not in all_entities:
                    all_entities[eid] = entity

            for edge in result.get("edges", []):
                src, tgt, edge_data = edge[0], edge[1], edge[2]
                edge_key = (src, tgt, edge_data.get("chunk_id", ""))
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)
                    all_edges.append((src, tgt, edge_data))

            for chunk in result.get("chunks", []):
                cid = chunk["id"]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_chunks.append(chunk)

        # Reconstruct hop distances
        hop_map = self._reconstruct_hops(anchor_ids, all_edges)

        # Assign weights based on hop distance
        for eid in all_entities:
            hop = hop_map.get(eid, params.neighborhood_max_hops + 1)
            all_entities[eid]["_hop"] = hop
            all_entities[eid]["_weight"] = 1.0 / (2 ** max(hop - 1, 0))

        # Sort by weight descending, cap at max_entities
        sorted_entities = sorted(
            all_entities.items(),
            key=lambda x: x[1]["_weight"],
            reverse=True,
        )[:params.neighborhood_max_entities]

        kept_entity_ids = {eid for eid, _ in sorted_entities}
        kept_entities = {eid: data for eid, data in sorted_entities}

        # Filter edges and chunks to only include kept entities
        kept_edges = [
            (s, t, e) for s, t, e in all_edges
            if s in kept_entity_ids and t in kept_entity_ids
        ]
        kept_chunk_ids = {e.get("chunk_id", "") for _, _, e in kept_edges}
        kept_chunks = [c for c in all_chunks if c["id"] in kept_chunk_ids]

        # Score chunks
        scored_chunks = []
        for chunk in kept_chunks:
            score = self._score_chunk(chunk, hop_map)
            chunk["_score"] = score
            scored_chunks.append(chunk)
        scored_chunks.sort(key=lambda c: c["_score"], reverse=True)

        return {
            "entities": kept_entities,
            "edges": kept_edges,
            "chunks": scored_chunks,
        }

    def _reconstruct_hops(
        self,
        anchor_ids: List[str],
        edges: List[Tuple[str, str, Dict]],
    ) -> Dict[str, int]:
        """BFS through returned edges to assign min hop distances from anchors.

        This reconstructs approximate hop distances since query_neighborhood
        returns a flat SubgraphResult without hop info.
        """
        # Build adjacency from edges (undirected for BFS)
        adjacency: Dict[str, set] = {}
        for src, tgt, _ in edges:
            adjacency.setdefault(src, set()).add(tgt)
            adjacency.setdefault(tgt, set()).add(src)

        hop_map: Dict[str, int] = {}
        queue = collections.deque()

        for aid in anchor_ids:
            hop_map[aid] = 0
            queue.append((aid, 0))

        while queue:
            current, current_hop = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor not in hop_map:
                    hop_map[neighbor] = current_hop + 1
                    queue.append((neighbor, current_hop + 1))

        return hop_map

    def _score_chunk(
        self, chunk: Dict, entity_hop_map: Dict[str, int]
    ) -> float:
        """Score a chunk by combining chunk type weight and entity distance.

        chunk_score = distance_weight * chunk_type_weight
        distance_weight = min(1/2^(hop-1)) for entities tagged in this chunk
        chunk_type_weight from config.chunk_type_weights
        """
        # Get chunk type weight
        chunk_type = chunk.get("chunk_type", "Discussion")
        if isinstance(chunk_type, dict):
            chunk_type = next(iter(chunk_type.keys()), "Discussion")
        type_weight = self.config.chunk_type_weights.get(chunk_type, 0.5)

        # Distance weight: best (closest) entity tagged in this chunk
        tags = chunk.get("tags", [])
        if tags:
            min_hop = min(
                (entity_hop_map.get(tag, 99) for tag in tags),
                default=99,
            )
        else:
            min_hop = 99

        distance_weight = 1.0 / (2 ** max(min_hop - 1, 0))

        return distance_weight * type_weight

    # ── Tier Building ──

    def _build_knowledge_map(
        self, expanded: Dict, budget: int
    ) -> str:
        """Tier 1: Key entity summaries ordered by weight.

        Format:
        - **Name** (Type): Appears in N chunks across M documents.
          Co-occurs most frequently with: X, Y, Z.
          Known since: YYYY-MM-DD. Last referenced: YYYY-MM-DD.
        """
        entities = expanded["entities"]
        edges = expanded["edges"]

        if not entities:
            return ""

        # Count co-occurrences per entity pair
        cooccurrence_counts: Dict[str, Dict[str, int]] = {}
        for src, tgt, _ in edges:
            cooccurrence_counts.setdefault(src, {})
            cooccurrence_counts[src][tgt] = cooccurrence_counts[src].get(tgt, 0) + 1
            cooccurrence_counts.setdefault(tgt, {})
            cooccurrence_counts[tgt][src] = cooccurrence_counts[tgt].get(src, 0) + 1

        # Sort entities by weight descending
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: x[1].get("_weight", 0),
            reverse=True,
        )

        lines = []
        tokens_used = 0
        for eid, entity in sorted_entities:
            name = entity.get("canonical_name", eid)
            etype = entity.get("entity_type", "Concept")
            if isinstance(etype, dict):
                etype = next(iter(etype.values()), "Concept")

            source_chunks = entity.get("source_chunks", [])
            source_docs = entity.get("source_documents", [])
            n_chunks = len(source_chunks)
            n_docs = len(source_docs)

            # Top co-occurring entities
            cooc = cooccurrence_counts.get(eid, {})
            top_cooc = sorted(cooc.items(), key=lambda x: x[1], reverse=True)[:3]
            cooc_names = []
            for cid, _ in top_cooc:
                if cid in entities:
                    cooc_names.append(
                        entities[cid].get("canonical_name", cid)
                    )

            created = self._format_date(entity.get("created_at"))
            updated = self._format_date(entity.get("updated_at"))

            line = f"- **{name}** ({etype}): Appears in {n_chunks} chunks across {n_docs} documents."
            if cooc_names:
                line += f"\n  Co-occurs most frequently with: {', '.join(cooc_names)}."
            line += f"\n  Known since: {created}. Last referenced: {updated}."

            line_tokens = self.count_tokens(line)
            if tokens_used + line_tokens > budget:
                break
            lines.append(line)
            tokens_used += line_tokens

        return "\n".join(lines)

    def _build_evidence_tier(
        self, expanded: Dict, budget: int
    ) -> Tuple[str, set]:
        """Tier 2: Evidence chunks ordered by score.

        Format:
        **[ChunkType] Section — Date**
        > chunk text

        Returns (formatted_text, set_of_entity_ids_mentioned_in_tier2).
        """
        chunks = expanded["chunks"]
        if not chunks:
            return "", set()

        entity_ids_in_tier2: set = set()
        lines = []
        tokens_used = 0

        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "Discussion")
            if isinstance(chunk_type, dict):
                chunk_type = next(iter(chunk_type.keys()), "Discussion")

            section = chunk.get("section_name", "")
            source_doc = chunk.get("source_document", "")
            timestamp = self._format_date(chunk.get("timestamp"))
            text = chunk.get("text", "")

            # Build header
            header_parts = [f"**[{chunk_type}]"]
            if section:
                header_parts.append(f" {section}")
            if source_doc:
                doc_name = Path(source_doc).stem
                header_parts.append(f" — {doc_name}")
            if timestamp != "unknown":
                header_parts.append(f" — {timestamp}")
            header_parts.append("**")
            header = "".join(header_parts)

            # Quote the text
            quoted = "\n".join(f"> {line}" for line in text.split("\n"))

            entry = f"{header}\n{quoted}"
            entry_tokens = self.count_tokens(entry)

            if tokens_used + entry_tokens > budget:
                break

            lines.append(entry)
            tokens_used += entry_tokens
            entity_ids_in_tier2.update(chunk.get("tags", []))

        return "\n\n".join(lines), entity_ids_in_tier2

    def _build_entity_summaries(
        self,
        expanded: Dict,
        tier2_entity_ids: set,
        budget: int,
    ) -> str:
        """Tier 3: Compact summaries for entities not fully covered in Tier 2.

        Format:
        - **Name** (Type): co-occurs with X, Y in N chunks
        """
        entities = expanded["entities"]
        edges = expanded["edges"]

        if not entities:
            return ""

        # Count co-occurrences per entity
        cooccurrence_counts: Dict[str, Dict[str, int]] = {}
        for src, tgt, _ in edges:
            cooccurrence_counts.setdefault(src, {})
            cooccurrence_counts[src][tgt] = cooccurrence_counts[src].get(tgt, 0) + 1
            cooccurrence_counts.setdefault(tgt, {})
            cooccurrence_counts[tgt][src] = cooccurrence_counts[tgt].get(src, 0) + 1

        # Only include entities NOT prominently featured in Tier 2
        peripheral = [
            (eid, data) for eid, data in entities.items()
            if eid not in tier2_entity_ids
        ]
        # Sort by weight
        peripheral.sort(key=lambda x: x[1].get("_weight", 0), reverse=True)

        lines = []
        tokens_used = 0
        for eid, entity in peripheral:
            name = entity.get("canonical_name", eid)
            etype = entity.get("entity_type", "Concept")
            if isinstance(etype, dict):
                etype = next(iter(etype.values()), "Concept")

            cooc = cooccurrence_counts.get(eid, {})
            top_cooc = sorted(cooc.items(), key=lambda x: x[1], reverse=True)[:2]
            cooc_names = []
            total_chunks = 0
            for cid, count in top_cooc:
                if cid in entities:
                    cooc_names.append(entities[cid].get("canonical_name", cid))
                total_chunks += count

            line = f"- **{name}** ({etype})"
            if cooc_names:
                line += f": co-occurs with {', '.join(cooc_names)} in {total_chunks} chunks"

            line_tokens = self.count_tokens(line)
            if tokens_used + line_tokens > budget:
                break
            lines.append(line)
            tokens_used += line_tokens

        return "\n".join(lines)

    # ── Formatting ──

    def _format_knowledge_context(
        self, tier1: str, tier2: str, tier3: str
    ) -> str:
        """Wrap tiers in markdown structure."""
        parts = ["## Knowledge Context"]

        if tier1:
            parts.append("")
            parts.append("### Key Entities")
            parts.append(tier1)

        if tier2:
            parts.append("")
            parts.append("### Evidence (most relevant chunks, ordered by relevance)")
            parts.append("")
            parts.append(tier2)

        if tier3:
            parts.append("")
            parts.append("### Peripheral Entities (summaries only)")
            parts.append(tier3)

        return "\n".join(parts)

    def _format_date(self, timestamp: Optional[int]) -> str:
        """Unix timestamp → 'YYYY-MM-DD' or 'unknown'."""
        if timestamp is None:
            return "unknown"
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError, OSError):
            return "unknown"

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens) + "\n\n[... truncated ...]"
