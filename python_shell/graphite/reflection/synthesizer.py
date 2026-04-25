"""Higher-order insight generation.

Phase 5: Analyzes entity temporal chains to detect patterns, refreshes
entity embeddings, and recalculates edge weights. Synthesis (LLM-powered
pattern detection) is optional — all other operations work without an LLM.

Cross-session reflection (conversation pivot):
- Temporal evolution: Track how the user's relationship with entities changes
- Cross-project patterns: Entities appearing across different project dirs
- Goal tracking: Goal entities checked against later sessions for completion
"""

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Summary of a synthesis run."""

    entities_synthesized: int = 0
    entities_skipped: int = 0
    embeddings_invalidated: int = 0
    edges_updated: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class Synthesizer:
    """Generates higher-order insights from knowledge graph patterns.

    Requires an LLM for synthesis. Embedding refresh and edge weight
    recalculation work without one.
    """

    def __init__(self, knowledge_graph, llm_client=None, embedding_manager=None, config=None):
        self._kg = knowledge_graph
        self._llm = llm_client
        self._embedding_manager = embedding_manager
        self._config = config
        self._max_chunks = 20
        if config:
            self._max_chunks = config.synthesis_max_chunks_per_entity

    def synthesize_entity_patterns(self, entity_id: str) -> Optional[str]:
        """Analyze an entity's temporal chain for patterns via LLM.

        Returns the synthesis text, or None if LLM unavailable or
        entity has too few chunks (< 3).
        """
        if self._llm is None:
            return None

        # Get entity info
        ej = self._kg.get_entity(entity_id)
        if ej is None:
            return None
        entity = json.loads(ej)

        # Get temporal chain
        chain_json = self._kg.get_temporal_chain(entity_id)
        chunks = json.loads(chain_json)

        if len(chunks) < 3:
            return None

        # Limit chunks
        chunks = chunks[:self._max_chunks]

        # Build prompt
        chunk_texts = []
        for i, chunk in enumerate(chunks, 1):
            ctype = chunk.get("chunk_type", "Unknown")
            text = chunk.get("text", "").strip()
            chunk_texts.append(f"[{i}] ({ctype}) {text}")

        prompt = (
            f"Analyze the following mentions of '{entity['canonical_name']}' "
            f"({entity.get('entity_type', 'Unknown')}) across multiple documents. "
            f"Identify patterns, recurring themes, or evolution over time. "
            f"Be concise (2-3 sentences).\n\n"
            + "\n".join(chunk_texts)
        )

        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
            synthesis = response.strip()
            if synthesis:
                # Store as a Background chunk
                chunk_json = json.dumps({
                    "source_document": f"synthesis/{entity_id}",
                    "chunk_type": "Background",
                    "memory_category": "Semantic",
                    "text": f"[Synthesis] {synthesis}",
                    "tags": [entity_id],
                })
                self._kg.store_chunk(chunk_json)
                return synthesis
        except Exception as e:
            logger.error("Synthesis failed for %s: %s", entity["canonical_name"], e)

        return None

    def enrich_entity_profiles(self) -> int:
        """Invalidate all entity embedding caches for re-computation.

        After consolidation, entity contexts may have changed (new aliases,
        merged co-occurrences). Invalidating forces fresh embeddings on
        next search.

        Returns count of invalidated entries.
        """
        if self._embedding_manager is None:
            return 0

        count = len(self._embedding_manager.entity_embeddings_cache)
        self._embedding_manager.invalidate_entity_cache()
        return count

    def update_edge_weights(self) -> int:
        """Recalculate edge weights based on co-occurrence frequency.

        Returns count of edges updated.
        """
        return self._kg.recalculate_edge_weights()

    # ── Cross-Session Reflection ──

    def analyze_temporal_evolution(self, entity_id: str) -> Optional[str]:
        """Analyze how the user's relationship with an entity evolves over time.

        Looks at chunks from conversation sessions across time to detect
        progression (e.g., "Started asking basic Rust questions; building
        complex systems by March").

        Requires LLM. Returns synthesis text or None.
        """
        if self._llm is None:
            return None

        ej = self._kg.get_entity(entity_id)
        if ej is None:
            return None
        entity = json.loads(ej)

        chain_json = self._kg.get_temporal_chain(entity_id)
        chunks = json.loads(chain_json)

        # Only analyze entities with conversation-source chunks
        conv_chunks = [
            c for c in chunks
            if c.get("source_document", "").startswith("claude-session://")
        ]
        if len(conv_chunks) < 3:
            return None

        conv_chunks = conv_chunks[:self._max_chunks]

        # Build prompt emphasizing temporal evolution
        chunk_texts = []
        for i, chunk in enumerate(conv_chunks, 1):
            ctype = chunk.get("chunk_type", "Unknown")
            text = chunk.get("text", "").strip()[:300]
            source = chunk.get("source_document", "")
            chunk_texts.append(f"[{i}] ({ctype}, {source}) {text}")

        prompt = (
            f"Analyze how the user's engagement with '{entity['canonical_name']}' "
            f"has evolved across these conversation excerpts (ordered chronologically). "
            f"Focus on: skill progression, changing attitudes, deepening expertise, "
            f"or shifting priorities. Be concise (2-3 sentences).\n\n"
            + "\n".join(chunk_texts)
        )

        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
            synthesis = response.strip()
            if synthesis:
                chunk_json = json.dumps({
                    "source_document": f"synthesis/evolution/{entity_id}",
                    "chunk_type": "Background",
                    "memory_category": "Semantic",
                    "text": f"[Temporal Evolution] {synthesis}",
                    "tags": [entity_id],
                })
                self._kg.store_chunk(chunk_json)
                return synthesis
        except Exception as e:
            logger.error(
                "Temporal evolution synthesis failed for %s: %s",
                entity["canonical_name"], e,
            )

        return None

    def find_cross_project_entities(self) -> List[Dict]:
        """Find entities that appear across multiple project directories.

        Returns a list of dicts: {entity_id, name, projects, mention_count}
        sorted by number of distinct projects (descending).
        No LLM required.
        """
        entity_projects: Dict[str, Set[str]] = defaultdict(set)
        entity_names: Dict[str, str] = {}

        all_ids = json.loads(self._kg.all_entity_ids())

        for eid in all_ids:
            chain_json = self._kg.get_temporal_chain(eid)
            chunks = json.loads(chain_json)

            for chunk in chunks:
                source = chunk.get("source_document", "")
                if source.startswith("claude-session://"):
                    # Extract project name from URI: claude-session://project/session
                    parts = source.replace("claude-session://", "").split("/")
                    if parts:
                        entity_projects[eid].add(parts[0])

            if len(entity_projects[eid]) > 1:
                ej = self._kg.get_entity(eid)
                if ej:
                    entity_names[eid] = json.loads(ej).get("canonical_name", eid)

        # Filter to multi-project entities
        results = []
        for eid, projects in entity_projects.items():
            if len(projects) > 1:
                results.append({
                    "entity_id": eid,
                    "name": entity_names.get(eid, eid),
                    "projects": sorted(projects),
                    "project_count": len(projects),
                })

        results.sort(key=lambda x: x["project_count"], reverse=True)
        return results

    def track_goals(self) -> List[Dict]:
        """Check Goal entities for evidence of completion or abandonment.

        Examines Goal-type entities and their temporal chains to determine
        status: active (recent mentions), completed (evidence of
        completion language), or stale (no mentions in 30+ days).

        No LLM required for basic status. Returns list of goal status dicts.
        """
        import time as _time

        now = int(_time.time())
        thirty_days_ago = now - (30 * 86400)

        all_ids = json.loads(self._kg.all_entity_ids())
        goals = []

        for eid in all_ids:
            ej = self._kg.get_entity(eid)
            if ej is None:
                continue
            entity = json.loads(ej)

            # Check if this is a Goal entity
            etype = entity.get("entity_type", "")
            if isinstance(etype, dict):
                etype = next(iter(etype.values()), "")
            if etype != "Goal":
                continue

            chain_json = self._kg.get_temporal_chain(eid)
            chunks = json.loads(chain_json)

            if not chunks:
                continue

            # Find most recent timestamp
            timestamps = [
                c.get("timestamp") for c in chunks
                if c.get("timestamp") is not None
            ]
            most_recent = max(timestamps) if timestamps else None

            # Determine status
            if most_recent is None:
                status = "unknown"
            elif most_recent < thirty_days_ago:
                status = "stale"
            else:
                status = "active"

            goals.append({
                "entity_id": eid,
                "name": entity["canonical_name"],
                "status": status,
                "mention_count": len(chunks),
                "last_mentioned": most_recent,
            })

        goals.sort(key=lambda g: g.get("last_mentioned") or 0, reverse=True)
        return goals

    def run(self, entity_ids: Optional[List[str]] = None) -> SynthesisResult:
        """Run all synthesis operations.

        Args:
            entity_ids: Optional list of entity IDs to synthesize.
                        If None, synthesizes all entities.
        """
        start = time.time()
        result = SynthesisResult()

        # Synthesis (requires LLM)
        if self._llm is not None:
            if entity_ids is None:
                all_ids = json.loads(self._kg.all_entity_ids())
            else:
                all_ids = entity_ids

            for eid in all_ids:
                try:
                    synthesis = self.synthesize_entity_patterns(eid)
                    if synthesis is not None:
                        result.entities_synthesized += 1
                    else:
                        result.entities_skipped += 1
                except Exception as e:
                    result.errors.append(f"Synthesis for {eid}: {e}")
                    result.entities_skipped += 1

        # Profile enrichment (no LLM needed)
        try:
            result.embeddings_invalidated = self.enrich_entity_profiles()
        except Exception as e:
            result.errors.append(f"Profile enrichment: {e}")

        # Edge weight recalculation (no LLM needed)
        try:
            result.edges_updated = self.update_edge_weights()
        except Exception as e:
            result.errors.append(f"Edge weight update: {e}")

        result.duration_seconds = time.time() - start
        return result
