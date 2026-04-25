"""Agent Context Injection Layer for Graphite.

Provides structured knowledge context for AI agent system prompt injection.
No LLM calls in the retrieval path — fast enough for every agent turn (~200ms warm).

Two depth modes:
  - "brief": Entity-level context (~100-200 tokens). Use every turn.
  - "full": Includes evidence chunks + recent events (~1000-5000 tokens).
            Use for deep reasoning turns.

Usage:
    assembler = AgentContextAssembler(kg, embedding_manager, config)
    ctx = assembler.assemble("Sarah is working on the dashboard redesign")
    print(ctx.to_injection_text())   # For system prompt
    print(ctx.to_dict())             # For structured consumption
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from graphite.config import GraphiteConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EntityBrief:
    """Summary of a single entity relevant to the agent's current situation."""

    id: str
    name: str
    type: str
    importance: float  # Normalized PageRank 0-1
    last_seen: Optional[str]  # "YYYY-MM-DD"
    top_connections: List[str]  # Names of top 3 co-occurring entities
    mention_count: int

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "importance": round(self.importance, 3),
            "last_seen": self.last_seen,
            "top_connections": self.top_connections,
            "mention_count": self.mention_count,
        }


@dataclass
class RelationshipBrief:
    """Summary of a co-occurrence relationship between two entities."""

    entity_a: str  # name
    entity_b: str  # name
    co_occurrence_count: int
    most_recent: Optional[str]  # "YYYY-MM-DD"

    def to_dict(self) -> Dict:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "co_occurrence_count": self.co_occurrence_count,
            "most_recent": self.most_recent,
        }


@dataclass
class RecentEvent:
    """A chunk-level event with timestamp and entity tags."""

    date: Optional[str]
    type: str  # "Decision", "ActionItem", etc.
    source: str  # Document name (stem)
    summary: str  # Chunk text, truncated ~200 chars
    entities: List[str]  # Entity names

    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "type": self.type,
            "source": self.source,
            "summary": self.summary,
            "entities": self.entities,
        }


@dataclass
class AgentContext:
    """Complete agent context for system prompt injection."""

    depth: str
    entities: List[EntityBrief]
    relationships: List[RelationshipBrief]
    recent_events: List[RecentEvent]  # Full mode only
    pending_items: List[RecentEvent]  # Both modes
    graph_stats: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "depth": self.depth,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "recent_events": [e.to_dict() for e in self.recent_events],
            "pending_items": [p.to_dict() for p in self.pending_items],
            "graph_stats": self.graph_stats,
        }

    def to_injection_text(self) -> str:
        """Format as compact text for system prompt injection."""
        lines = ["## Memory Context"]

        # Key Entities
        if self.entities:
            lines.append("")
            lines.append("**Key Entities:**")
            for e in self.entities:
                connections = ", ".join(e.top_connections) if e.top_connections else "none"
                last = f", last seen {e.last_seen}" if e.last_seen else ""
                lines.append(
                    f"- {e.name} ({e.type}, importance: {e.importance:.2f}): "
                    f"connected to {connections}. "
                    f"{e.mention_count} mentions{last}."
                )

        # Key Relationships
        if self.relationships:
            lines.append("")
            lines.append("**Key Relationships:**")
            for r in self.relationships:
                last = f" (last: {r.most_recent})" if r.most_recent else ""
                lines.append(
                    f"- {r.entity_a} <> {r.entity_b}: "
                    f"{r.co_occurrence_count} co-occurrences{last}"
                )

        # Pending Items (both modes)
        if self.pending_items:
            lines.append("")
            lines.append("**Pending Items:**")
            for p in self.pending_items:
                date_str = f" {p.date}" if p.date else ""
                entities_str = f" ({', '.join(p.entities)})" if p.entities else ""
                lines.append(
                    f"- [{p.type}]{date_str}: {p.summary}{entities_str}"
                )

        # Recent Events (full mode only)
        if self.recent_events:
            lines.append("")
            lines.append("**Recent Events:**")
            for e in self.recent_events:
                date_str = f" {e.date}" if e.date else ""
                entities_str = f" ({', '.join(e.entities)})" if e.entities else ""
                lines.append(
                    f"- [{e.type}]{date_str} {e.source}: {e.summary}{entities_str}"
                )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Assembler
# ═══════════════════════════════════════════════════════════════════════════════


def _format_timestamp(ts) -> Optional[str]:
    """Format a Unix timestamp to YYYY-MM-DD, or None."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return None


def _truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def _format_entity_type(etype) -> str:
    """Normalize entity type from Rust enum format."""
    if isinstance(etype, dict):
        # Custom type comes as {"Custom": "TypeName"}
        return next(iter(etype.values()), "Concept")
    return str(etype)


class AgentContextAssembler:
    """Assembles structured knowledge context for agent system prompts.

    Combines entity name extraction (no LLM), embedding search, and
    PageRank re-ranking to find relevant entities fast enough for
    every agent turn (~200ms warm).

    Args:
        knowledge_graph: PyKnowledgeGraph instance.
        embedding_manager: EmbeddingManager instance (shared with MemoryContextManager).
        config: Optional GraphiteConfig (uses defaults if None).
    """

    def __init__(self, knowledge_graph, embedding_manager, config=None):
        self.kg = knowledge_graph
        self.embedding_manager = embedding_manager
        self.config = config or GraphiteConfig()

        # Caches (populated lazily on first assemble)
        self._pagerank_cache: Dict[str, float] = {}
        self._entity_cache: Dict[str, Dict] = {}
        self._entities_embedded: bool = False

    # ── Public API ──

    def assemble(
        self,
        situation: str,
        depth: str = "brief",
        max_entities: Optional[int] = None,
        max_events: Optional[int] = None,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> AgentContext:
        """Assemble agent context for the given situation.

        Args:
            situation: The agent's current situation / task description.
            depth: "brief" (every turn) or "full" (deep reasoning).
            max_entities: Override entity count limit.
            max_events: Override event count limit (full mode).
            time_start: Unix timestamp lower bound for temporal filtering.
            time_end: Unix timestamp upper bound for temporal filtering.

        Returns:
            AgentContext with entities, relationships, pending items,
            and (in full mode) recent events.
        """
        if depth not in ("brief", "full"):
            depth = "brief"

        if max_entities is None:
            max_entities = (
                self.config.agent_brief_max_entities
                if depth == "brief"
                else self.config.agent_full_max_entities
            )
        if max_events is None:
            max_events = self.config.agent_full_max_events

        # 1. Ensure embeddings are cached
        self._ensure_entity_embeddings()

        if not self._pagerank_cache:
            # Empty graph
            return AgentContext(
                depth=depth,
                entities=[],
                relationships=[],
                recent_events=[],
                pending_items=[],
                graph_stats={"entity_count": 0, "edge_count": 0},
            )

        # 2. Extract entity names from situation text
        name_matches = self._extract_entity_names(situation)

        # 3. Embedding search
        entity_ids = list(self._pagerank_cache.keys())
        embedding_candidates = self.embedding_manager.find_relevant_entities_scored(
            situation, entity_ids, top_n=max_entities * 3,
        )

        # 4. Merge & re-rank
        ranked_entities = self._merge_and_rerank(
            name_matches, embedding_candidates, max_entities
        )

        # 5. Build EntityBriefs
        entity_briefs = [
            self._build_entity_brief(eid) for eid, _ in ranked_entities
        ]

        # 6. Build RelationshipBriefs
        result_ids = {eid for eid, _ in ranked_entities}
        relationship_briefs = self._build_relationship_briefs(result_ids)

        # 7. Build PendingItems
        pending_items = self._build_pending_items(
            [eid for eid, _ in ranked_entities]
        )

        # 8-9. Full mode: expand + recent events
        recent_events = []
        if depth == "full":
            recent_events = self._build_recent_events(
                [eid for eid, _ in ranked_entities],
                max_events,
                time_start,
                time_end,
            )

        # Graph stats
        graph_stats = {
            "entity_count": len(self._entity_cache),
            "edge_count": self._count_edges(),
        }

        return AgentContext(
            depth=depth,
            entities=entity_briefs,
            relationships=relationship_briefs,
            recent_events=recent_events,
            pending_items=pending_items,
            graph_stats=graph_stats,
        )

    # ── Embedding Cache (reuses MemoryContextManager pattern) ──

    def _ensure_entity_embeddings(self) -> None:
        """Lazily embed all entities on first call.

        Same pattern as MemoryContextManager._ensure_entity_embeddings().
        Shares the EmbeddingManager instance so embeddings computed by
        assemble_memory() are reused for free.
        """
        if self._entities_embedded:
            return

        pagerank_json = self.kg.compute_pagerank()
        pagerank_pairs = json.loads(pagerank_json)

        self._pagerank_cache.clear()
        self._entity_cache.clear()

        entities_for_embedding = []
        for entity_id, score in pagerank_pairs:
            self._pagerank_cache[entity_id] = score
            entity_json = self.kg.get_entity(entity_id)
            if entity_json:
                entity_data = json.loads(entity_json)
                self._entity_cache[entity_id] = entity_data

                etype = _format_entity_type(entity_data.get("entity_type", "Concept"))

                entities_for_embedding.append({
                    "id": entity_id,
                    "canonical_name": entity_data["canonical_name"],
                    "entity_type": etype,
                })

        if entities_for_embedding:
            self.embedding_manager.embed_entities(entities_for_embedding, self.kg)

        self._entities_embedded = True
        logger.debug(f"Agent assembler: embedded {len(entities_for_embedding)} entities.")

    # ── Entity Name Extraction (no LLM) ──

    def _extract_entity_names(
        self, text: str
    ) -> Dict[str, float]:
        """Extract entity names from situation text via graph index lookup.

        Performs case-insensitive matching of entity canonical names and
        aliases against the text. No LLM calls — O(entity_count).

        Returns:
            {entity_id: match_score} where score indicates match quality:
              1.0 = word-boundary match
              0.7 = substring match
              0.6 = alias match
        """
        matches: Dict[str, float] = {}
        text_lower = text.lower()

        for entity_id, entity_data in self._entity_cache.items():
            canonical = entity_data["canonical_name"]

            # Skip very short names to avoid false positives
            if len(canonical) < 3:
                continue

            canonical_lower = canonical.lower()

            # Check canonical name
            if canonical_lower in text_lower:
                # Word-boundary check
                pattern = r'\b' + re.escape(canonical_lower) + r'\b'
                if re.search(pattern, text_lower):
                    matches[entity_id] = 1.0
                else:
                    matches[entity_id] = 0.7
                continue

            # Check aliases
            aliases = entity_data.get("aliases", [])
            for alias in aliases:
                if len(alias) < 3:
                    continue
                alias_lower = alias.lower()
                if alias_lower in text_lower:
                    matches[entity_id] = max(matches.get(entity_id, 0), 0.6)
                    break

        return matches

    # ── Merge & Re-rank ──

    def _merge_and_rerank(
        self,
        name_matches: Dict[str, float],
        embedding_candidates: List[Tuple[str, float]],
        max_entities: int,
    ) -> List[Tuple[str, float]]:
        """Merge name-matched and embedding-searched entities, then re-rank.

        Name-matched entities get a bonus added to their similarity score.
        Combined score: similarity_weight * sim + pagerank_weight * normalized_pr

        Returns:
            [(entity_id, combined_score)] sorted descending, capped to max_entities.
        """
        # Build similarity map from embedding candidates
        sim_map: Dict[str, float] = {}
        for eid, sim_score in embedding_candidates:
            sim_map[eid] = sim_score

        # Add name-matched entities that weren't in embedding results
        for eid in name_matches:
            if eid not in sim_map:
                sim_map[eid] = 0.0

        if not sim_map:
            return []

        # Normalize PageRank scores
        max_pr = max(self._pagerank_cache.values()) if self._pagerank_cache else 1.0
        if max_pr == 0:
            max_pr = 1.0

        bonus = self.config.agent_name_match_bonus

        reranked: List[Tuple[str, float]] = []
        for eid, sim_score in sim_map.items():
            # Apply name match bonus
            if eid in name_matches:
                sim_score += bonus * name_matches[eid]

            pr = self._pagerank_cache.get(eid, 0.0)
            normalized_pr = pr / max_pr

            combined = (
                self.config.similarity_weight * sim_score
                + self.config.pagerank_weight * normalized_pr
            )
            reranked.append((eid, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:max_entities]

    # ── Build EntityBriefs ──

    def _build_entity_brief(self, entity_id: str) -> EntityBrief:
        """Build an EntityBrief from cached entity data."""
        entity_data = self._entity_cache.get(entity_id, {})
        canonical = entity_data.get("canonical_name", entity_id)
        etype = _format_entity_type(entity_data.get("entity_type", "Concept"))

        # PageRank normalized 0-1
        max_pr = max(self._pagerank_cache.values()) if self._pagerank_cache else 1.0
        if max_pr == 0:
            max_pr = 1.0
        importance = self._pagerank_cache.get(entity_id, 0.0) / max_pr

        # Top connections from co-occurrence edges
        top_connections = self._get_top_connections(entity_id, limit=3)

        # Mention count from source_chunks
        mention_count = len(entity_data.get("source_chunks", []))

        # Last seen: most recent timestamp from co-occurrence edges
        last_seen = self._get_last_seen(entity_id)

        return EntityBrief(
            id=entity_id,
            name=canonical,
            type=etype,
            importance=importance,
            last_seen=last_seen,
            top_connections=top_connections,
            mention_count=mention_count,
        )

    def _get_top_connections(self, entity_id: str, limit: int = 3) -> List[str]:
        """Get names of top co-occurring entities."""
        cooccurrences_json = self.kg.get_cooccurrences(entity_id)
        cooccurrences = json.loads(cooccurrences_json)

        # Count per neighbor
        neighbor_counts: Dict[str, int] = {}
        for neighbor_id, _edge_data in cooccurrences:
            neighbor_counts[neighbor_id] = neighbor_counts.get(neighbor_id, 0) + 1

        # Sort by count, take top N
        sorted_neighbors = sorted(
            neighbor_counts.items(), key=lambda x: x[1], reverse=True
        )[:limit]

        names = []
        for nid, _ in sorted_neighbors:
            neighbor = self._entity_cache.get(nid)
            if neighbor:
                names.append(neighbor["canonical_name"])
            else:
                # Try loading from graph
                neighbor_json = self.kg.get_entity(nid)
                if neighbor_json:
                    names.append(json.loads(neighbor_json)["canonical_name"])
        return names

    def _get_last_seen(self, entity_id: str) -> Optional[str]:
        """Get the most recent timestamp for this entity's co-occurrence edges."""
        cooccurrences_json = self.kg.get_cooccurrences(entity_id)
        cooccurrences = json.loads(cooccurrences_json)

        max_ts = None
        for _neighbor_id, edge_data in cooccurrences:
            ts = edge_data.get("timestamp")
            if ts is not None:
                if max_ts is None or ts > max_ts:
                    max_ts = ts

        # Fall back to entity's updated_at
        if max_ts is None:
            entity_data = self._entity_cache.get(entity_id, {})
            max_ts = entity_data.get("updated_at")

        return _format_timestamp(max_ts)

    # ── Build RelationshipBriefs ──

    def _build_relationship_briefs(
        self, entity_ids: set
    ) -> List[RelationshipBrief]:
        """Build relationship briefs for entity pairs within the result set."""
        pair_data: Dict[Tuple[str, str], Dict] = {}

        for eid in entity_ids:
            cooccurrences_json = self.kg.get_cooccurrences(eid)
            cooccurrences = json.loads(cooccurrences_json)

            for neighbor_id, edge_data in cooccurrences:
                if neighbor_id not in entity_ids:
                    continue

                # Canonical pair key (alphabetical by ID for dedup)
                pair = tuple(sorted([eid, neighbor_id]))
                if pair not in pair_data:
                    pair_data[pair] = {"count": 0, "max_ts": None}

                pair_data[pair]["count"] += 1
                ts = edge_data.get("timestamp")
                if ts is not None:
                    current_max = pair_data[pair]["max_ts"]
                    if current_max is None or ts > current_max:
                        pair_data[pair]["max_ts"] = ts

        # Build briefs sorted by count descending
        briefs = []
        for (id_a, id_b), data in pair_data.items():
            name_a = self._entity_cache.get(id_a, {}).get("canonical_name", id_a)
            name_b = self._entity_cache.get(id_b, {}).get("canonical_name", id_b)
            briefs.append(RelationshipBrief(
                entity_a=name_a,
                entity_b=name_b,
                co_occurrence_count=data["count"],
                most_recent=_format_timestamp(data["max_ts"]),
            ))

        briefs.sort(key=lambda r: r.co_occurrence_count, reverse=True)
        return briefs

    # ── Build PendingItems ──

    def _build_pending_items(
        self, entity_ids: List[str]
    ) -> List[RecentEvent]:
        """Build pending items (Decision/ActionItem chunks) for result entities."""
        pending_types = set(self.config.agent_pending_chunk_types)
        seen_chunk_ids: set = set()
        items: List[Tuple[Optional[int], RecentEvent]] = []

        for eid in entity_ids:
            chain_json = self.kg.get_temporal_chain(eid)
            chain = json.loads(chain_json)

            for chunk in chain:
                chunk_id = chunk.get("id", "")
                if chunk_id in seen_chunk_ids:
                    continue

                chunk_type = chunk.get("chunk_type", "")
                if chunk_type not in pending_types:
                    continue

                seen_chunk_ids.add(chunk_id)
                ts = chunk.get("timestamp")

                # Resolve entity names from tags
                tag_names = self._resolve_tag_names(chunk.get("tags", []))

                items.append((
                    ts,
                    RecentEvent(
                        date=_format_timestamp(ts),
                        type=chunk_type,
                        source=Path(chunk.get("source_document", "")).stem,
                        summary=_truncate(chunk.get("text", "")),
                        entities=tag_names,
                    ),
                ))

        # Sort by timestamp desc (most recent first), None last
        items.sort(key=lambda x: x[0] if x[0] is not None else 0, reverse=True)
        return [item for _, item in items]

    # ── Build RecentEvents (full mode) ──

    def _build_recent_events(
        self,
        entity_ids: List[str],
        max_events: int,
        time_start: Optional[int],
        time_end: Optional[int],
    ) -> List[RecentEvent]:
        """Build recent events by expanding entity neighborhoods."""
        seen_chunk_ids: set = set()
        all_chunks: List[Dict] = []

        for eid in entity_ids:
            result_json = self.kg.query_neighborhood(
                eid, 2, time_start, time_end
            )
            result = json.loads(result_json)

            for chunk in result.get("chunks", []):
                cid = chunk.get("id", "")
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_chunks.append(chunk)

        # Sort by timestamp desc
        all_chunks.sort(
            key=lambda c: c.get("timestamp") if c.get("timestamp") is not None else 0,
            reverse=True,
        )

        events = []
        for chunk in all_chunks[:max_events]:
            ts = chunk.get("timestamp")
            tag_names = self._resolve_tag_names(chunk.get("tags", []))

            events.append(RecentEvent(
                date=_format_timestamp(ts),
                type=chunk.get("chunk_type", "Unknown"),
                source=Path(chunk.get("source_document", "")).stem,
                summary=_truncate(chunk.get("text", "")),
                entities=tag_names,
            ))

        return events

    # ── Helpers ──

    def _resolve_tag_names(self, tags: List[str]) -> List[str]:
        """Resolve entity IDs to canonical names."""
        names = []
        for tag in tags:
            entity = self._entity_cache.get(tag)
            if entity:
                names.append(entity["canonical_name"])
        return names

    def _count_edges(self) -> int:
        """Count total edges in the graph (from co-occurrence queries)."""
        seen = set()
        for eid in self._entity_cache:
            cooccurrences_json = self.kg.get_cooccurrences(eid)
            cooccurrences = json.loads(cooccurrences_json)
            for neighbor_id, edge_data in cooccurrences:
                edge_key = tuple(sorted([eid, neighbor_id])) + (
                    edge_data.get("chunk_id", ""),
                )
                seen.add(edge_key)
        return len(seen)

    def invalidate_caches(self) -> None:
        """Clear all cached data. Call after graph changes."""
        self._pagerank_cache.clear()
        self._entity_cache.clear()
        self._entities_embedded = False

    # ── User Profile Assembly ──

    def assemble_user_profile(self) -> "UserProfile":
        """Build a user profile from conversation-derived entities.

        Queries the graph for Preference, Goal, Pattern, Skill, Project,
        and Concept entities, then constructs a narrative description
        of the user based on accumulated conversation history.

        Returns:
            UserProfile with categorized entity lists and narrative text.
        """
        self._ensure_entity_embeddings()

        # Categorize entities by type
        preferences: List[EntityBrief] = []
        goals: List[EntityBrief] = []
        work_patterns: List[EntityBrief] = []
        skills: List[EntityBrief] = []
        active_projects: List[EntityBrief] = []
        recurring_themes: List[EntityBrief] = []

        # Type mapping for categorization
        type_buckets = {
            "Preference": preferences,
            "Goal": goals,
            "Pattern": work_patterns,
            "Skill": skills,
        }

        # Sort entities by PageRank (importance)
        sorted_entities = sorted(
            self._pagerank_cache.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for entity_id, pr_score in sorted_entities:
            entity_data = self._entity_cache.get(entity_id)
            if not entity_data:
                continue

            etype = _format_entity_type(entity_data.get("entity_type", "Concept"))
            brief = self._build_entity_brief(entity_id)

            # Route to the right bucket
            if etype in type_buckets:
                type_buckets[etype].append(brief)
            elif etype == "Project":
                active_projects.append(brief)
            elif etype == "Concept" and brief.mention_count >= 3:
                recurring_themes.append(brief)

        # Cap each category
        max_per = 10
        preferences = preferences[:max_per]
        goals = goals[:max_per]
        work_patterns = work_patterns[:max_per]
        skills = skills[:max_per]
        active_projects = active_projects[:max_per]
        recurring_themes = recurring_themes[:max_per]

        # Build narrative
        narrative = self._build_profile_narrative(
            preferences, goals, work_patterns, skills,
            active_projects, recurring_themes,
        )

        return UserProfile(
            preferences=preferences,
            goals=goals,
            work_patterns=work_patterns,
            skills=skills,
            active_projects=active_projects,
            recurring_themes=recurring_themes,
            narrative=narrative,
        )

    def _build_profile_narrative(
        self,
        preferences: List[EntityBrief],
        goals: List[EntityBrief],
        work_patterns: List[EntityBrief],
        skills: List[EntityBrief],
        active_projects: List[EntityBrief],
        recurring_themes: List[EntityBrief],
    ) -> str:
        """Build a human-readable narrative from entity categories."""
        lines = ["## What I Know About You"]

        if preferences:
            lines.append("")
            lines.append("**Preferences:**")
            for p in preferences:
                lines.append(f"- {p.name} ({p.mention_count} mentions)")

        if goals:
            lines.append("")
            lines.append("**Current Goals:**")
            for g in goals:
                connections = ", ".join(g.top_connections) if g.top_connections else ""
                suffix = f" (related: {connections})" if connections else ""
                lines.append(f"- {g.name}{suffix}")

        if work_patterns:
            lines.append("")
            lines.append("**Work Style:**")
            for wp in work_patterns:
                lines.append(f"- {wp.name}")

        if skills:
            lines.append("")
            lines.append("**Skills & Growth:**")
            for s in skills:
                lines.append(f"- {s.name}")

        if active_projects:
            lines.append("")
            lines.append("**Active Projects:**")
            for p in active_projects:
                last = f" (last: {p.last_seen})" if p.last_seen else ""
                lines.append(f"- {p.name}{last}")

        if recurring_themes:
            lines.append("")
            lines.append("**Recurring Themes:**")
            for t in recurring_themes:
                lines.append(
                    f"- {t.name} (importance: {t.importance:.2f}, "
                    f"{t.mention_count} mentions)"
                )

        if len(lines) == 1:
            lines.append("")
            lines.append("No user profile data yet. Ingest conversation sessions first.")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# User Profile Data Structure
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class UserProfile:
    """Structured user profile assembled from conversation-derived entities."""

    preferences: List[EntityBrief]
    goals: List[EntityBrief]
    work_patterns: List[EntityBrief]
    skills: List[EntityBrief]
    active_projects: List[EntityBrief]
    recurring_themes: List[EntityBrief]
    narrative: str

    def to_dict(self) -> Dict:
        return {
            "preferences": [e.to_dict() for e in self.preferences],
            "goals": [e.to_dict() for e in self.goals],
            "work_patterns": [e.to_dict() for e in self.work_patterns],
            "skills": [e.to_dict() for e in self.skills],
            "active_projects": [e.to_dict() for e in self.active_projects],
            "recurring_themes": [e.to_dict() for e in self.recurring_themes],
            "narrative": self.narrative,
        }
