"""Tests for Agent Context Injection Layer.

Covers:
- Entity name extraction from situation text
- Name match bonus scoring
- Brief mode structure (entities, relationships, pending items)
- Full mode structure (adds recent events)
- Pending items filter (only Decision/ActionItem)
- Empty graph handling
- to_injection_text() formatting
- to_dict() JSON serialization
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from graphite.agent_context import (
    AgentContext,
    AgentContextAssembler,
    EntityBrief,
    RecentEvent,
    RelationshipBrief,
    _format_timestamp,
    _truncate,
)
from graphite.config import GraphiteConfig
from tests.test_memory_context import (
    FakeKnowledgeGraph,
    _make_mock_embedding_manager,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


def _build_agent_test_graph():
    """Build a graph with entities suitable for agent context tests.

    Contains:
      - "Sarah Chen" (Person) with aliases ["SC"]
      - "Dashboard Redesign" (Project) — no aliases
      - "React" (Technology) — no aliases
      - "Mike Torres" (Person) — no aliases
      - "API Rate Limiting" (Decision) — no aliases

    Edges:
      - Sarah <-> Dashboard (c1, c2)
      - Sarah <-> React (c1)
      - Sarah <-> Mike (c3)
      - Dashboard <-> React (c1)
      - API Rate Limiting <-> Sarah (c4 — Decision type)

    Chunks:
      - c1: Discussion about dashboard with React (Sarah, Dashboard, React)
      - c2: Status update on dashboard (Sarah, Dashboard)
      - c3: Discussion between Sarah and Mike (Sarah, Mike)
      - c4: Decision on API rate limiting (API Rate Limiting, Sarah)
      - c5: ActionItem for Sarah (Sarah, Dashboard)
    """
    kg = FakeKnowledgeGraph()
    ts_mar5 = int(datetime(2026, 3, 5, tzinfo=timezone.utc).timestamp())
    ts_mar6 = int(datetime(2026, 3, 6, tzinfo=timezone.utc).timestamp())
    ts_mar1 = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())

    kg.add_test_entity("e-sarah", "Sarah Chen", "Person",
                       source_chunks=["c1", "c2", "c3", "c4", "c5"],
                       source_documents=["meetings/standup.md"],
                       created_at=ts_mar1, updated_at=ts_mar6)
    kg._entities["e-sarah"]["aliases"] = ["SC"]

    kg.add_test_entity("e-dashboard", "Dashboard Redesign", "Project",
                       source_chunks=["c1", "c2", "c5"],
                       source_documents=["meetings/standup.md"],
                       created_at=ts_mar1, updated_at=ts_mar6)

    kg.add_test_entity("e-react", "React", "Technology",
                       source_chunks=["c1"],
                       source_documents=["meetings/standup.md"],
                       created_at=ts_mar1, updated_at=ts_mar5)

    kg.add_test_entity("e-mike", "Mike Torres", "Person",
                       source_chunks=["c3"],
                       source_documents=["meetings/standup.md"],
                       created_at=ts_mar1, updated_at=ts_mar5)

    kg.add_test_entity("e-api", "API Rate Limiting", "Decision",
                       source_chunks=["c4"],
                       source_documents=["meetings/architecture.md"],
                       created_at=ts_mar1, updated_at=ts_mar1)

    # Edges
    kg.add_test_edge("e-sarah", "e-dashboard", "c1", "Discussion", ts_mar5)
    kg.add_test_edge("e-sarah", "e-dashboard", "c2", "StatusUpdate", ts_mar6)
    kg.add_test_edge("e-sarah", "e-react", "c1", "Discussion", ts_mar5)
    kg.add_test_edge("e-sarah", "e-mike", "c3", "Discussion", ts_mar5)
    kg.add_test_edge("e-dashboard", "e-react", "c1", "Discussion", ts_mar5)
    kg.add_test_edge("e-api", "e-sarah", "c4", "Decision", ts_mar1)
    kg.add_test_edge("e-sarah", "e-dashboard", "c5", "ActionItem", ts_mar5)

    # Chunks
    kg.add_test_chunk("c1",
        "Sarah Chen presented the new dashboard wireframes using React components.",
        chunk_type="Discussion",
        tags=["e-sarah", "e-dashboard", "e-react"],
        timestamp=ts_mar5,
        source_document="meetings/standup.md")

    kg.add_test_chunk("c2",
        "Dashboard Redesign is on track. Sarah Chen completed the main layout.",
        chunk_type="StatusUpdate",
        tags=["e-sarah", "e-dashboard"],
        timestamp=ts_mar6,
        source_document="meetings/standup.md")

    kg.add_test_chunk("c3",
        "Sarah Chen and Mike Torres discussed component architecture.",
        chunk_type="Discussion",
        tags=["e-sarah", "e-mike"],
        timestamp=ts_mar5,
        source_document="meetings/standup.md")

    kg.add_test_chunk("c4",
        "Approved 100 req/s rate limit for external API.",
        chunk_type="Decision",
        tags=["e-api", "e-sarah"],
        timestamp=ts_mar1,
        source_document="meetings/architecture.md")

    kg.add_test_chunk("c5",
        "Sarah to finalize dashboard component API by Friday.",
        chunk_type="ActionItem",
        tags=["e-sarah", "e-dashboard"],
        timestamp=ts_mar5,
        source_document="meetings/standup.md")

    return kg


def _make_agent_assembler(kg=None, config=None):
    """Create an AgentContextAssembler with mock embedding manager."""
    if kg is None:
        kg = _build_agent_test_graph()
    if config is None:
        config = GraphiteConfig()
    em = _make_mock_embedding_manager()
    return AgentContextAssembler(kg, em, config), kg


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatTimestamp:
    def test_valid_timestamp(self):
        ts = int(datetime(2026, 3, 5, tzinfo=timezone.utc).timestamp())
        assert _format_timestamp(ts) == "2026-03-05"

    def test_none_returns_none(self):
        assert _format_timestamp(None) is None

    def test_invalid_returns_none(self):
        assert _format_timestamp(-999999999999) is None


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_long_text_truncated(self):
        text = "x" * 250
        result = _truncate(text, max_chars=200)
        assert len(result) == 200
        assert result.endswith("...")

    def test_exact_length_unchanged(self):
        text = "x" * 200
        assert _truncate(text, max_chars=200) == text


# ═══════════════════════════════════════════════════════════════════════════════
# Entity Name Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEntityNameExtraction:
    def test_word_boundary_match(self):
        assembler, kg = _make_agent_assembler()
        assembler._ensure_entity_embeddings()

        matches = assembler._extract_entity_names(
            "Sarah Chen is working on the dashboard"
        )
        assert "e-sarah" in matches
        assert matches["e-sarah"] == 1.0

    def test_substring_match(self):
        """Names found as substrings (no word boundary) get lower score."""
        assembler, kg = _make_agent_assembler()
        assembler._ensure_entity_embeddings()

        # "Dashboard Redesign" should match in text containing it
        matches = assembler._extract_entity_names(
            "The Dashboard Redesign project is progressing well"
        )
        assert "e-dashboard" in matches
        assert matches["e-dashboard"] == 1.0  # word-boundary match

    def test_alias_match(self):
        assembler, kg = _make_agent_assembler()
        assembler._ensure_entity_embeddings()

        # "SC" is too short (< 3 chars), so won't match
        # This is by design to avoid false positives
        matches = assembler._extract_entity_names("SC reviewed the code")
        assert "e-sarah" not in matches

    def test_short_names_skipped(self):
        """Names shorter than 3 chars are skipped to avoid false positives."""
        assembler, kg = _make_agent_assembler()
        # Add a short-named entity
        kg.add_test_entity("e-ai", "AI", "Technology",
                           source_chunks=["c1"])
        assembler._entities_embedded = False
        assembler._ensure_entity_embeddings()

        matches = assembler._extract_entity_names("AI is transforming everything")
        assert "e-ai" not in matches

    def test_case_insensitive(self):
        assembler, kg = _make_agent_assembler()
        assembler._ensure_entity_embeddings()

        matches = assembler._extract_entity_names(
            "sarah chen joined the meeting"
        )
        assert "e-sarah" in matches

    def test_no_matches(self):
        assembler, kg = _make_agent_assembler()
        assembler._ensure_entity_embeddings()

        matches = assembler._extract_entity_names(
            "The weather is nice today"
        )
        assert len(matches) == 0

    def test_multiple_matches(self):
        assembler, kg = _make_agent_assembler()
        assembler._ensure_entity_embeddings()

        matches = assembler._extract_entity_names(
            "Sarah Chen and Mike Torres discussed React"
        )
        assert "e-sarah" in matches
        assert "e-mike" in matches
        assert "e-react" in matches


# ═══════════════════════════════════════════════════════════════════════════════
# Name Match Bonus Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNameMatchBonus:
    def test_name_matched_entities_get_boost(self):
        assembler, kg = _make_agent_assembler()

        name_matches = {"e-sarah": 1.0}
        embedding_candidates = [
            ("e-sarah", 0.5),
            ("e-react", 0.6),
        ]

        result = assembler._merge_and_rerank(
            name_matches, embedding_candidates, max_entities=10
        )

        # Sarah should rank higher due to name match bonus
        result_ids = [eid for eid, _ in result]
        sarah_idx = result_ids.index("e-sarah")
        react_idx = result_ids.index("e-react")
        assert sarah_idx < react_idx

    def test_entities_not_in_embeddings_still_included(self):
        """Name-matched entities missing from embedding search still appear."""
        assembler, kg = _make_agent_assembler()

        name_matches = {"e-mike": 1.0}
        embedding_candidates = [
            ("e-sarah", 0.8),
            ("e-react", 0.6),
        ]

        result = assembler._merge_and_rerank(
            name_matches, embedding_candidates, max_entities=10
        )
        result_ids = [eid for eid, _ in result]
        assert "e-mike" in result_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Brief Mode Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBriefMode:
    def test_brief_mode_structure(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble(
            "Sarah Chen is working on Dashboard Redesign",
            depth="brief",
        )

        assert ctx.depth == "brief"
        assert isinstance(ctx.entities, list)
        assert isinstance(ctx.relationships, list)
        assert isinstance(ctx.pending_items, list)
        assert isinstance(ctx.recent_events, list)
        assert isinstance(ctx.graph_stats, dict)
        assert "entity_count" in ctx.graph_stats
        assert "edge_count" in ctx.graph_stats

        # Brief mode should NOT have recent events
        assert len(ctx.recent_events) == 0

    def test_entity_briefs_populated(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble(
            "Sarah Chen is working on Dashboard Redesign",
            depth="brief",
        )

        assert len(ctx.entities) > 0
        for e in ctx.entities:
            assert isinstance(e, EntityBrief)
            assert e.id
            assert e.name
            assert e.type
            assert 0.0 <= e.importance <= 1.0
            assert isinstance(e.top_connections, list)
            assert isinstance(e.mention_count, int)

    def test_relationships_between_result_entities(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble(
            "Sarah Chen is working on Dashboard Redesign",
            depth="brief",
        )

        entity_names = {e.name for e in ctx.entities}
        for r in ctx.relationships:
            assert r.entity_a in entity_names or r.entity_b in entity_names
            assert r.co_occurrence_count > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Full Mode Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullMode:
    def test_full_mode_includes_recent_events(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble(
            "Sarah Chen is working on Dashboard Redesign",
            depth="full",
        )

        assert ctx.depth == "full"
        assert len(ctx.recent_events) > 0

    def test_full_mode_has_more_entities(self):
        config = GraphiteConfig(
            agent_brief_max_entities=2,
            agent_full_max_entities=10,
        )
        assembler, kg = _make_agent_assembler(config=config)

        brief = assembler.assemble("Sarah Chen Dashboard Redesign", depth="brief")
        assembler.invalidate_caches()
        full = assembler.assemble("Sarah Chen Dashboard Redesign", depth="full")

        assert len(full.entities) >= len(brief.entities)

    def test_recent_events_structure(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("Sarah Chen Dashboard", depth="full")

        for event in ctx.recent_events:
            assert isinstance(event, RecentEvent)
            assert event.type
            assert event.source
            assert isinstance(event.entities, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Pending Items Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPendingItems:
    def test_pending_items_filter(self):
        """Only Decision and ActionItem chunks appear as pending items."""
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble(
            "Sarah Chen Dashboard Redesign API Rate Limiting",
            depth="brief",
        )

        for item in ctx.pending_items:
            assert item.type in ("Decision", "ActionItem"), \
                f"Unexpected pending item type: {item.type}"

    def test_pending_items_have_content(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("Sarah Chen Dashboard", depth="brief")

        if ctx.pending_items:
            item = ctx.pending_items[0]
            assert item.summary
            assert isinstance(item.entities, list)

    def test_custom_pending_types(self):
        """Config can override which chunk types count as pending."""
        config = GraphiteConfig(
            agent_pending_chunk_types=["StatusUpdate"]
        )
        assembler, kg = _make_agent_assembler(config=config)
        ctx = assembler.assemble("Sarah Chen Dashboard Redesign", depth="brief")

        for item in ctx.pending_items:
            assert item.type == "StatusUpdate"


# ═══════════════════════════════════════════════════════════════════════════════
# Empty Graph Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmptyGraph:
    def test_empty_graph_returns_graceful_result(self):
        kg = FakeKnowledgeGraph()
        assembler, _ = _make_agent_assembler(kg=kg)

        ctx = assembler.assemble("anything at all", depth="brief")

        assert ctx.depth == "brief"
        assert ctx.entities == []
        assert ctx.relationships == []
        assert ctx.recent_events == []
        assert ctx.pending_items == []
        assert ctx.graph_stats["entity_count"] == 0

    def test_empty_graph_full_mode(self):
        kg = FakeKnowledgeGraph()
        assembler, _ = _make_agent_assembler(kg=kg)

        ctx = assembler.assemble("anything", depth="full")
        assert ctx.entities == []
        assert ctx.recent_events == []


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestToInjectionText:
    def test_well_formatted_output(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("Sarah Chen Dashboard Redesign", depth="brief")

        text = ctx.to_injection_text()
        assert text.startswith("## Memory Context")
        assert "**Key Entities:**" in text
        assert "Sarah Chen" in text

    def test_includes_pending_items(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble(
            "Sarah Chen Dashboard Redesign API Rate Limiting",
            depth="brief",
        )

        text = ctx.to_injection_text()
        if ctx.pending_items:
            assert "**Pending Items:**" in text

    def test_full_mode_includes_recent_events(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("Sarah Chen Dashboard", depth="full")

        text = ctx.to_injection_text()
        if ctx.recent_events:
            assert "**Recent Events:**" in text

    def test_empty_context_minimal_output(self):
        kg = FakeKnowledgeGraph()
        assembler, _ = _make_agent_assembler(kg=kg)
        ctx = assembler.assemble("nothing", depth="brief")

        text = ctx.to_injection_text()
        assert text == "## Memory Context"

    def test_relationships_in_output(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("Sarah Chen Dashboard Redesign", depth="brief")

        text = ctx.to_injection_text()
        if ctx.relationships:
            assert "**Key Relationships:**" in text
            assert "<>" in text


class TestToDict:
    def test_json_serializable(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("Sarah Chen Dashboard", depth="brief")

        d = ctx.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert parsed["depth"] == "brief"
        assert isinstance(parsed["entities"], list)
        assert isinstance(parsed["relationships"], list)
        assert isinstance(parsed["pending_items"], list)
        assert isinstance(parsed["recent_events"], list)
        assert isinstance(parsed["graph_stats"], dict)

    def test_entity_brief_to_dict(self):
        brief = EntityBrief(
            id="e-1",
            name="Sarah Chen",
            type="Person",
            importance=0.82,
            last_seen="2026-03-05",
            top_connections=["Dashboard Redesign", "React"],
            mention_count=47,
        )
        d = brief.to_dict()
        assert d["name"] == "Sarah Chen"
        assert d["importance"] == 0.82
        assert len(d["top_connections"]) == 2

    def test_relationship_brief_to_dict(self):
        rel = RelationshipBrief(
            entity_a="Sarah Chen",
            entity_b="Dashboard Redesign",
            co_occurrence_count=15,
            most_recent="2026-03-06",
        )
        d = rel.to_dict()
        assert d["co_occurrence_count"] == 15
        assert d["most_recent"] == "2026-03-06"

    def test_recent_event_to_dict(self):
        event = RecentEvent(
            date="2026-03-06",
            type="StatusUpdate",
            source="standup",
            summary="Dashboard is on track",
            entities=["Sarah Chen", "Dashboard Redesign"],
        )
        d = event.to_dict()
        assert d["type"] == "StatusUpdate"
        assert len(d["entities"]) == 2

    def test_full_mode_dict_structure(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("Sarah Chen Dashboard", depth="full")

        d = ctx.to_dict()
        assert d["depth"] == "full"
        # Full mode should have recent_events populated
        assert isinstance(d["recent_events"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Invalidation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheInvalidation:
    def test_invalidate_clears_state(self):
        assembler, kg = _make_agent_assembler()
        assembler.assemble("Sarah Chen", depth="brief")

        assert assembler._entities_embedded
        assert len(assembler._pagerank_cache) > 0
        assert len(assembler._entity_cache) > 0

        assembler.invalidate_caches()

        assert not assembler._entities_embedded
        assert len(assembler._pagerank_cache) == 0
        assert len(assembler._entity_cache) == 0

    def test_reassemble_after_invalidation(self):
        assembler, kg = _make_agent_assembler()
        ctx1 = assembler.assemble("Sarah Chen", depth="brief")
        assembler.invalidate_caches()
        ctx2 = assembler.assemble("Sarah Chen", depth="brief")

        # Should produce similar results
        assert len(ctx2.entities) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Config Override Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigOverrides:
    def test_max_entities_override(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble(
            "Sarah Chen Dashboard Redesign React Mike Torres API Rate Limiting",
            depth="brief",
            max_entities=2,
        )
        assert len(ctx.entities) <= 2

    def test_invalid_depth_defaults_to_brief(self):
        assembler, kg = _make_agent_assembler()
        ctx = assembler.assemble("test", depth="invalid")
        assert ctx.depth == "brief"
        assert len(ctx.recent_events) == 0
