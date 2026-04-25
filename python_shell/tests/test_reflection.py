"""Tests for Phase 5: Reflection & Consolidation.

Covers:
- Consolidator: merge candidate detection, confirmation, execution,
  orphan cleanup, decay scoring, orchestration
- Synthesizer: LLM synthesis, embedding refresh, edge weight update
"""

import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from graphite.config import GraphiteConfig
from tests.test_memory_context import (
    FakeKnowledgeGraph,
    _build_test_graph,
    _make_mock_embedding_manager,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


def _build_merge_candidate_graph():
    """Build a graph with merge candidates.

    Contains:
      - "John Doe" (Person) with aliases ["JD"]
      - "J. Doe" (Person) with aliases ["John Doe", "JD"]
        → These should be merge candidates (high alias overlap)
      - "React" (Technology) — no merge partner
      - "Orphan" (Person) — no edges, old creation time
    """
    kg = FakeKnowledgeGraph()

    old_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    recent_ts = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())

    kg.add_test_entity("e-john", "John Doe", "Person",
                       source_chunks=["c1", "c2"],
                       source_documents=["meetings/q3.md"],
                       created_at=recent_ts, updated_at=recent_ts)
    kg._entities["e-john"]["aliases"] = ["JD"]

    kg.add_test_entity("e-jdoe", "J. Doe", "Person",
                       source_chunks=["c3"],
                       source_documents=["meetings/q4.md"],
                       created_at=recent_ts, updated_at=recent_ts)
    kg._entities["e-jdoe"]["aliases"] = ["John Doe", "JD"]

    kg.add_test_entity("e-react", "React", "Technology",
                       source_chunks=["c4"],
                       source_documents=["work/tech.md"],
                       created_at=recent_ts, updated_at=recent_ts)

    # Orphan: old, no edges
    kg.add_test_entity("e-orphan", "Old Orphan", "Person",
                       source_chunks=[],
                       source_documents=[],
                       created_at=old_ts, updated_at=old_ts)

    # Edges: John <-> React, J. Doe <-> React
    kg.add_test_edge("e-john", "e-react", "c1", "Discussion", recent_ts)
    kg.add_test_edge("e-jdoe", "e-react", "c3", "Discussion", recent_ts)

    # Chunks
    kg.add_test_chunk("c1", "John Doe likes React.", tags=["e-john", "e-react"],
                      timestamp=recent_ts)
    kg.add_test_chunk("c3", "J. Doe reviewed React code.", tags=["e-jdoe", "e-react"],
                      timestamp=recent_ts)

    return kg


class MockMergeLLM:
    """Mock LLM that approves or rejects merges."""

    def __init__(self, approve=True):
        self._approve = approve

    def chat(self, messages):
        return "YES" if self._approve else "NO"


class MockSynthesisLLM:
    """Mock LLM that returns canned synthesis."""

    def chat(self, messages):
        return "This entity shows a pattern of increasing involvement."


# ═══════════════════════════════════════════════════════════════════════════════
# Consolidator Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindMergeCandidates:
    def test_alias_overlap_found(self):
        """Entities with overlapping aliases are detected as candidates."""
        from graphite.reflection.consolidator import Consolidator

        kg = _build_merge_candidate_graph()
        config = GraphiteConfig(merge_alias_overlap_threshold=0.30)
        consolidator = Consolidator(kg, config=config)

        candidates = consolidator.find_merge_candidates()
        assert len(candidates) >= 1
        names = {(c.keep_name, c.merge_name) for c in candidates}
        # John Doe and J. Doe should be candidates
        assert any(
            ("John Doe" in pair and "J. Doe" in pair)
            for pair in [(c.keep_name, c.merge_name) for c in candidates]
        )

    def test_different_types_excluded(self):
        """Entities of different types are never merge candidates."""
        from graphite.reflection.consolidator import Consolidator

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-react-person", "React", "Person")
        kg._entities["e-react-person"]["aliases"] = ["React"]
        kg.add_test_entity("e-react-tech", "React", "Technology")
        kg._entities["e-react-tech"]["aliases"] = ["React"]

        config = GraphiteConfig(merge_alias_overlap_threshold=0.50)
        consolidator = Consolidator(kg, config=config)

        candidates = consolidator.find_merge_candidates()
        assert len(candidates) == 0

    def test_clean_graph_empty(self):
        """Graph with no duplicates returns no candidates."""
        from graphite.reflection.consolidator import Consolidator

        kg = _build_test_graph()  # clean graph
        config = GraphiteConfig(merge_alias_overlap_threshold=0.80)
        consolidator = Consolidator(kg, config=config)

        candidates = consolidator.find_merge_candidates()
        assert len(candidates) == 0


class TestConfirmMerges:
    def test_no_llm_high_confidence(self):
        """Without LLM, high-confidence candidates are auto-approved."""
        from graphite.reflection.consolidator import Consolidator, MergeCandidate

        kg = FakeKnowledgeGraph()
        consolidator = Consolidator(kg, llm_client=None)

        candidates = [
            MergeCandidate("a", "A", "b", "B", 0.96, "alias overlap"),
        ]
        confirmed = consolidator.confirm_merges(candidates)
        assert len(confirmed) == 1
        assert confirmed[0].confirmed is True

    def test_no_llm_low_confidence(self):
        """Without LLM, low-confidence candidates are not approved."""
        from graphite.reflection.consolidator import Consolidator, MergeCandidate

        kg = FakeKnowledgeGraph()
        consolidator = Consolidator(kg, llm_client=None)

        candidates = [
            MergeCandidate("a", "A", "b", "B", 0.80, "alias overlap"),
        ]
        confirmed = consolidator.confirm_merges(candidates)
        assert len(confirmed) == 0

    def test_llm_approves(self):
        """LLM that says YES approves the merge."""
        from graphite.reflection.consolidator import Consolidator, MergeCandidate

        kg = FakeKnowledgeGraph()
        llm = MockMergeLLM(approve=True)
        consolidator = Consolidator(kg, llm_client=llm)

        candidates = [
            MergeCandidate("a", "A", "b", "B", 0.80, "alias overlap"),
        ]
        confirmed = consolidator.confirm_merges(candidates)
        assert len(confirmed) == 1

    def test_llm_rejects(self):
        """LLM that says NO rejects the merge."""
        from graphite.reflection.consolidator import Consolidator, MergeCandidate

        kg = FakeKnowledgeGraph()
        llm = MockMergeLLM(approve=False)
        consolidator = Consolidator(kg, llm_client=llm)

        candidates = [
            MergeCandidate("a", "A", "b", "B", 0.80, "alias overlap"),
        ]
        confirmed = consolidator.confirm_merges(candidates)
        assert len(confirmed) == 0


class TestExecuteMerges:
    def test_merges_performed(self):
        """Confirmed merges are executed on the knowledge graph."""
        from graphite.reflection.consolidator import Consolidator, MergeCandidate

        kg = _build_merge_candidate_graph()
        consolidator = Consolidator(kg)

        confirmed = [
            MergeCandidate("e-john", "John Doe", "e-jdoe", "J. Doe", 0.96, "alias", confirmed=True),
        ]
        count = consolidator.execute_merges(confirmed)
        assert count == 1
        # J. Doe should be gone
        assert kg.get_entity("e-jdoe") is None
        # John Doe should have absorbed aliases
        john = json.loads(kg.get_entity("e-john"))
        assert "J. Doe" in john["aliases"]

    def test_embedding_cache_invalidated(self):
        """Merge invalidates embedding cache for both entities."""
        from graphite.reflection.consolidator import Consolidator, MergeCandidate

        kg = _build_merge_candidate_graph()
        mock_embed = _make_mock_embedding_manager()
        mock_embed.entity_embeddings_cache["e-john"] = "vec1"
        mock_embed.entity_embeddings_cache["e-jdoe"] = "vec2"

        consolidator = Consolidator(kg, embedding_manager=mock_embed)

        confirmed = [
            MergeCandidate("e-john", "John Doe", "e-jdoe", "J. Doe", 0.96, "alias", confirmed=True),
        ]
        consolidator.execute_merges(confirmed)

        assert "e-john" not in mock_embed.entity_embeddings_cache
        assert "e-jdoe" not in mock_embed.entity_embeddings_cache


class TestCleanupOrphans:
    def test_old_orphan_removed(self):
        """Orphan entities older than max age are removed."""
        from graphite.reflection.consolidator import Consolidator

        kg = _build_merge_candidate_graph()
        config = GraphiteConfig(orphan_max_age_days=7)
        consolidator = Consolidator(kg, config=config)

        count = consolidator.cleanup_orphans()
        assert count == 1
        assert kg.get_entity("e-orphan") is None

    def test_new_orphan_preserved(self):
        """Recently created orphans are kept."""
        from graphite.reflection.consolidator import Consolidator

        kg = FakeKnowledgeGraph()
        now = int(time.time())
        kg.add_test_entity("e-new", "New Orphan", "Person",
                           created_at=now, updated_at=now)
        config = GraphiteConfig(orphan_max_age_days=7)
        consolidator = Consolidator(kg, config=config)

        count = consolidator.cleanup_orphans()
        assert count == 0
        assert kg.get_entity("e-new") is not None

    def test_connected_entity_kept(self):
        """Entities with edges are never orphan-removed."""
        from graphite.reflection.consolidator import Consolidator

        kg = _build_merge_candidate_graph()
        config = GraphiteConfig(orphan_max_age_days=0)  # remove any orphan regardless of age
        consolidator = Consolidator(kg, config=config)

        count = consolidator.cleanup_orphans()
        # Only orphan should be removed, connected entities stay
        assert kg.get_entity("e-john") is not None
        assert kg.get_entity("e-react") is not None


class TestApplyDecay:
    def test_access_count_reduced(self):
        """Decay reduces access_count based on age."""
        from graphite.reflection.consolidator import Consolidator

        kg = FakeKnowledgeGraph()
        old_ts = int(time.time()) - (30 * 86400)  # 30 days ago
        kg.add_test_entity("e-old", "Old Entity", "Person",
                           created_at=old_ts, updated_at=old_ts)
        kg._entities["e-old"]["access_count"] = 100

        config = GraphiteConfig(decay_half_life_days=30.0, decay_archival_threshold=5)
        consolidator = Consolidator(kg, config=config)

        total, flagged = consolidator.apply_decay()
        assert total == 1


class TestConsolidatorOrchestration:
    def test_run_full_does_not_remove_orphans_or_decay(self):
        """Lifelong-memory semantics: run_full no longer prunes orphans
        or applies decay. The planted orphan survives a full run."""
        from graphite.reflection.consolidator import Consolidator

        kg = _build_merge_candidate_graph()
        config = GraphiteConfig(orphan_max_age_days=7)
        consolidator = Consolidator(kg, config=config)

        assert kg.get_entity("e-orphan") is not None

        result = consolidator.run_full()
        assert result.duration_seconds >= 0
        assert result.orphans_removed == 0
        assert result.entities_decayed == 0
        assert kg.get_entity("e-orphan") is not None, (
            "run_full must not remove orphans in lifelong-memory mode"
        )

    def test_run_lightweight_is_noop(self):
        """run_lightweight is a compatibility stub after PR 3 — it performs
        no graph mutations and returns a zeroed result."""
        from graphite.reflection.consolidator import Consolidator

        kg = _build_merge_candidate_graph()
        config = GraphiteConfig(orphan_max_age_days=7)
        consolidator = Consolidator(kg, config=config)

        result = consolidator.run_lightweight()
        assert result.duration_seconds >= 0
        assert result.orphans_removed == 0
        assert result.merges_found == 0
        assert result.entities_decayed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Synthesizer Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSynthesizer:
    def test_no_llm_returns_none(self):
        """Without LLM, synthesis returns None."""
        from graphite.reflection.synthesizer import Synthesizer

        kg = _build_test_graph()
        synthesizer = Synthesizer(kg, llm_client=None)

        result = synthesizer.synthesize_entity_patterns("e-john")
        assert result is None

    def test_mock_llm_creates_background_chunk(self):
        """With LLM, synthesis stores a Background chunk."""
        from graphite.reflection.synthesizer import Synthesizer

        kg = _build_test_graph()
        # Add a third chunk tagged with e-john (synthesis requires >= 3)
        ts = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())
        kg.add_test_chunk("c5", "John Doe approved the final design.",
                          chunk_type="Decision", tags=["e-john", "e-dash"],
                          timestamp=ts)
        llm = MockSynthesisLLM()
        synthesizer = Synthesizer(kg, llm_client=llm)

        result = synthesizer.synthesize_entity_patterns("e-john")
        assert result is not None
        assert "pattern" in result.lower() or "involvement" in result.lower()

        # A chunk should have been stored via store_chunk(json_str)
        # In FakeKnowledgeGraph, store_chunk takes a dict, but synthesizer
        # passes JSON string — need to check this works
        stored = False
        for chunk in kg._chunks.values():
            text = chunk.get("text", "") if isinstance(chunk, dict) else ""
            if "[Synthesis]" in text:
                stored = True
                break
        assert stored

    def test_entity_with_few_chunks_skipped(self):
        """Entity with < 3 chunks is skipped."""
        from graphite.reflection.synthesizer import Synthesizer

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-lone", "Lone", "Person")
        kg.add_test_chunk("c1", "One mention.", tags=["e-lone"])
        kg.add_test_chunk("c2", "Two mentions.", tags=["e-lone"])

        llm = MockSynthesisLLM()
        synthesizer = Synthesizer(kg, llm_client=llm)

        result = synthesizer.synthesize_entity_patterns("e-lone")
        assert result is None

    def test_enrichment_invalidates_embeddings(self):
        """enrich_entity_profiles invalidates the embedding cache."""
        from graphite.reflection.synthesizer import Synthesizer

        kg = _build_test_graph()
        mock_embed = _make_mock_embedding_manager()
        mock_embed.entity_embeddings_cache["e-john"] = "vec"
        mock_embed.entity_embeddings_cache["e-jane"] = "vec"

        synthesizer = Synthesizer(kg, embedding_manager=mock_embed)
        count = synthesizer.enrich_entity_profiles()
        assert count == 2
        assert len(mock_embed.entity_embeddings_cache) == 0

    def test_edge_weight_update(self):
        """update_edge_weights calls recalculate."""
        from graphite.reflection.synthesizer import Synthesizer

        kg = _build_test_graph()
        synthesizer = Synthesizer(kg)
        updated = synthesizer.update_edge_weights()
        assert updated >= 0

    def test_full_run_completes(self):
        """Full synthesis run completes without errors."""
        from graphite.reflection.synthesizer import Synthesizer

        kg = _build_test_graph()
        llm = MockSynthesisLLM()
        mock_embed = _make_mock_embedding_manager()

        synthesizer = Synthesizer(kg, llm_client=llm, embedding_manager=mock_embed)
        result = synthesizer.run()
        assert result.duration_seconds >= 0
        assert result.entities_synthesized + result.entities_skipped > 0
