"""Entity merging, orphan cleanup, and decay scoring.

Phase 5: Periodically reviews the knowledge graph to merge duplicate
entities, apply temporal decay to stale relationships, and prune
orphan entities with no connections. Works without an LLM for core
operations; LLM enhances merge confirmation quality.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MergeCandidate:
    """A pair of entities that may be duplicates."""

    keep_id: str
    keep_name: str
    merge_id: str
    merge_name: str
    confidence: float
    reason: str
    confirmed: bool = False


@dataclass
class ConsolidationResult:
    """Summary of a consolidation run."""

    merges_found: int = 0
    merges_executed: int = 0
    orphans_removed: int = 0
    entities_decayed: int = 0
    entities_flagged_low: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class Consolidator:
    """Finds and executes entity merges, cleans orphans, applies decay.

    Core operations work without an LLM. If an LLM is provided, it can
    confirm ambiguous merge candidates.
    """

    def __init__(self, knowledge_graph, embedding_manager=None, config=None, llm_client=None):
        self._kg = knowledge_graph
        self._embedding_manager = embedding_manager
        self._config = config
        self._llm = llm_client

        # Config defaults
        self._merge_alias_threshold = 0.80
        self._merge_embedding_threshold = 0.90
        self._auto_approve_threshold = 0.95
        self._decay_half_life = 30.0
        self._decay_archival_threshold = 5
        self._orphan_max_age_days = 7

        if config:
            self._merge_alias_threshold = config.merge_alias_overlap_threshold
            self._merge_embedding_threshold = config.merge_embedding_threshold
            self._decay_half_life = config.decay_half_life_days
            self._decay_archival_threshold = config.decay_archival_threshold
            self._orphan_max_age_days = config.orphan_max_age_days

    # ── Core operations ──

    def find_merge_candidates(self) -> List[MergeCandidate]:
        """Find entity pairs that may be duplicates.

        Checks alias overlap (Jaccard) and embedding similarity within
        same-type groups. Returns deduplicated candidates sorted by
        confidence descending.
        """
        all_ids_json = self._kg.all_entity_ids()
        all_ids = json.loads(all_ids_json)

        if len(all_ids) < 2:
            return []

        # Load all entities, group by type
        by_type = {}
        entities = {}
        for eid in all_ids:
            ej = self._kg.get_entity(eid)
            if ej is None:
                continue
            entity = json.loads(ej)
            entities[eid] = entity
            etype = entity.get("entity_type", "Unknown")
            by_type.setdefault(etype, []).append(eid)

        candidates = {}  # (keep, merge) -> MergeCandidate

        # Check alias overlap within each type group
        for etype, ids in by_type.items():
            for i, id_a in enumerate(ids):
                for id_b in ids[i + 1:]:
                    ea = entities[id_a]
                    eb = entities[id_b]

                    names_a = {ea["canonical_name"].lower()}
                    names_a.update(a.lower() for a in ea.get("aliases", []))
                    names_b = {eb["canonical_name"].lower()}
                    names_b.update(a.lower() for a in eb.get("aliases", []))

                    jaccard = _jaccard(names_a, names_b)
                    if jaccard >= self._merge_alias_threshold:
                        # Keep the one with more source_chunks
                        if len(ea.get("source_chunks", [])) >= len(eb.get("source_chunks", [])):
                            keep_id, keep_name = id_a, ea["canonical_name"]
                            merge_id, merge_name = id_b, eb["canonical_name"]
                        else:
                            keep_id, keep_name = id_b, eb["canonical_name"]
                            merge_id, merge_name = id_a, ea["canonical_name"]

                        key = tuple(sorted([keep_id, merge_id]))
                        if key not in candidates or jaccard > candidates[key].confidence:
                            candidates[key] = MergeCandidate(
                                keep_id=keep_id,
                                keep_name=keep_name,
                                merge_id=merge_id,
                                merge_name=merge_name,
                                confidence=jaccard,
                                reason=f"alias overlap (Jaccard={jaccard:.2f})",
                            )

        # Check embedding similarity if available
        if self._embedding_manager is not None:
            for etype, ids in by_type.items():
                # Build entity dicts for embedding
                ents_for_embed = [entities[eid] for eid in ids if eid in entities]
                if len(ents_for_embed) < 2:
                    continue

                self._embedding_manager.embed_entities(ents_for_embed, self._kg)
                cache = self._embedding_manager.entity_embeddings_cache

                for i, id_a in enumerate(ids):
                    if id_a not in cache:
                        continue
                    emb_a = cache[id_a]
                    for id_b in ids[i + 1:]:
                        if id_b not in cache:
                            continue
                        emb_b = cache[id_b]

                        import numpy as np
                        norm_a = np.linalg.norm(emb_a)
                        norm_b = np.linalg.norm(emb_b)
                        if norm_a == 0 or norm_b == 0:
                            continue
                        sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

                        if sim >= self._merge_embedding_threshold:
                            ea = entities[id_a]
                            eb = entities[id_b]
                            if len(ea.get("source_chunks", [])) >= len(eb.get("source_chunks", [])):
                                keep_id, keep_name = id_a, ea["canonical_name"]
                                merge_id, merge_name = id_b, eb["canonical_name"]
                            else:
                                keep_id, keep_name = id_b, eb["canonical_name"]
                                merge_id, merge_name = id_a, ea["canonical_name"]

                            key = tuple(sorted([keep_id, merge_id]))
                            if key not in candidates or sim > candidates[key].confidence:
                                candidates[key] = MergeCandidate(
                                    keep_id=keep_id,
                                    keep_name=keep_name,
                                    merge_id=merge_id,
                                    merge_name=merge_name,
                                    confidence=sim,
                                    reason=f"embedding similarity ({sim:.2f})",
                                )

        result = sorted(candidates.values(), key=lambda c: c.confidence, reverse=True)
        return result

    def confirm_merges(self, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        """Confirm merge candidates.

        Without LLM: auto-approve candidates with confidence >= 0.95.
        With LLM: ask for confirmation on each candidate.
        """
        confirmed = []
        for candidate in candidates:
            if candidate.confidence >= self._auto_approve_threshold:
                candidate.confirmed = True
                confirmed.append(candidate)
                logger.info(
                    "Auto-approved merge: %s → %s (%.2f)",
                    candidate.merge_name, candidate.keep_name, candidate.confidence,
                )
            elif self._llm is not None:
                try:
                    response = self._llm.chat([{
                        "role": "user",
                        "content": (
                            f"Are these two entities the same? Answer YES or NO only.\n"
                            f"Entity A: {candidate.keep_name}\n"
                            f"Entity B: {candidate.merge_name}\n"
                            f"Reason for suspecting duplicate: {candidate.reason}"
                        ),
                    }])
                    if "yes" in response.strip().lower():
                        candidate.confirmed = True
                        confirmed.append(candidate)
                        logger.info(
                            "LLM approved merge: %s → %s",
                            candidate.merge_name, candidate.keep_name,
                        )
                    else:
                        logger.info(
                            "LLM rejected merge: %s → %s",
                            candidate.merge_name, candidate.keep_name,
                        )
                except Exception as e:
                    logger.warning("LLM merge confirmation failed: %s", e)
            # else: no LLM and below threshold — skip

        return confirmed

    def execute_merges(self, confirmed: List[MergeCandidate]) -> int:
        """Execute confirmed merges. Returns count of merges performed."""
        count = 0
        for candidate in confirmed:
            if not candidate.confirmed:
                continue
            try:
                self._kg.merge_entities(
                    candidate.keep_id, candidate.merge_id,
                    confidence=candidate.confidence, method=candidate.reason,
                )
                # Invalidate embedding cache for both
                if self._embedding_manager is not None:
                    self._embedding_manager.invalidate_entity_cache(candidate.keep_id)
                    self._embedding_manager.invalidate_entity_cache(candidate.merge_id)
                count += 1
                logger.info(
                    "Merged '%s' into '%s'",
                    candidate.merge_name, candidate.keep_name,
                )
            except Exception as e:
                logger.error(
                    "Merge failed (%s → %s): %s",
                    candidate.merge_name, candidate.keep_name, e,
                )
        return count

    def cleanup_orphans(self) -> int:
        """Remove orphan entities (no edges) older than max age.

        Returns the count of entities removed.
        """
        orphan_ids = json.loads(self._kg.find_orphan_entities())
        if not orphan_ids:
            return 0

        now = int(time.time())
        max_age_seconds = self._orphan_max_age_days * 86400
        removed = 0

        for eid in orphan_ids:
            ej = self._kg.get_entity(eid)
            if ej is None:
                continue
            entity = json.loads(ej)
            created_at = entity.get("created_at", now)
            age = now - created_at
            if age >= max_age_seconds:
                self._kg.remove_entity(eid)
                if self._embedding_manager is not None:
                    self._embedding_manager.invalidate_entity_cache(eid)
                removed += 1
                logger.info("Removed orphan entity: %s (%s)", entity["canonical_name"], eid)

        return removed

    def apply_decay(self) -> Tuple[int, int]:
        """Apply exponential decay to access counts.

        Returns (entities_decayed, entities_flagged_below_threshold).
        """
        self._kg.decay_scores(self._decay_half_life)

        # Count entities below archival threshold
        all_ids = json.loads(self._kg.all_entity_ids())
        total = len(all_ids)
        flagged = 0
        for eid in all_ids:
            ej = self._kg.get_entity(eid)
            if ej is None:
                continue
            entity = json.loads(ej)
            if entity.get("access_count", 0) < self._decay_archival_threshold:
                flagged += 1

        return total, flagged

    # ── Orchestration ──

    def run_full(self) -> ConsolidationResult:
        """Run all consolidation operations.

        1. Find and execute merges
        2. Clean up orphans
        3. Apply decay scoring
        """
        start = time.time()
        result = ConsolidationResult()

        try:
            # Merges
            candidates = self.find_merge_candidates()
            result.merges_found = len(candidates)
            confirmed = self.confirm_merges(candidates)
            result.merges_executed = self.execute_merges(confirmed)
        except Exception as e:
            result.errors.append(f"Merge phase: {e}")
            logger.error("Merge phase failed: %s", e)

        try:
            # Orphan cleanup
            result.orphans_removed = self.cleanup_orphans()
        except Exception as e:
            result.errors.append(f"Orphan cleanup: {e}")
            logger.error("Orphan cleanup failed: %s", e)

        try:
            # Decay
            result.entities_decayed, result.entities_flagged_low = self.apply_decay()
        except Exception as e:
            result.errors.append(f"Decay: {e}")
            logger.error("Decay failed: %s", e)

        result.duration_seconds = time.time() - start
        return result

    def run_lightweight(self) -> ConsolidationResult:
        """Run lightweight consolidation (orphan cleanup only).

        Suitable for post-ingest quick cleanup.
        """
        start = time.time()
        result = ConsolidationResult()

        try:
            result.orphans_removed = self.cleanup_orphans()
        except Exception as e:
            result.errors.append(f"Orphan cleanup: {e}")
            logger.error("Orphan cleanup failed: %s", e)

        result.duration_seconds = time.time() - start
        return result
