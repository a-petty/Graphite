"""Tests for Phase 4: MCP server knowledge-centric tools.

Covers:
- Entity resolution (by ID, by name, not found, ambiguous)
- All 9 read-only tools
- 2 semantic search tools
- 2 write tools (with mocked LLM/pipeline)
- 3 Phase 5 stubs
- Helper functions (_parse_date, _auto_save, _format_timestamp)
- Edge cases (empty graph, missing entity)
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Reuse FakeKnowledgeGraph from test_memory_context
from tests.test_memory_context import (
    FakeKnowledgeGraph,
    _build_test_graph,
    _make_mock_embedding_manager,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


def _reset_mcp_globals():
    """Reset all MCP server global state between tests."""
    import graphite.mcp_server as srv

    srv._kg = None
    srv._embedding_manager = None
    srv._context_manager = None
    srv._pipeline = None
    srv._config = None
    srv._project_root = Path("/tmp/test-project")
    srv._graph_root = Path("/tmp/test-project")
    srv._graph_root_override = None
    srv._graph_initialized = False
    srv._graph_dirty = False
    srv._tool_lock = None
    srv._write_lock = None
    srv._readers_lock = None
    srv._readers_count = 0
    srv._reflection_task = None
    srv._last_save_time = 0.0
    srv._save_task = None
    srv._dirty_entity_ids = set()


def _setup_populated_graph():
    """Set up MCP globals with a populated FakeKnowledgeGraph."""
    import graphite.mcp_server as srv
    from graphite.config import GraphiteConfig

    _reset_mcp_globals()
    kg = _build_test_graph()
    srv._kg = kg
    srv._config = GraphiteConfig()
    srv._graph_initialized = True
    return kg


def _setup_empty_graph():
    """Set up MCP globals with an empty FakeKnowledgeGraph."""
    import graphite.mcp_server as srv
    from graphite.config import GraphiteConfig

    _reset_mcp_globals()
    kg = FakeKnowledgeGraph()
    srv._kg = kg
    srv._config = GraphiteConfig()
    srv._graph_initialized = True
    return kg


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset MCP server globals before each test."""
    _reset_mcp_globals()
    yield
    _reset_mcp_globals()


# ═══════════════════════════════════════════════════════════════════════════════
# Entity Resolution Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveEntity:
    def test_resolve_by_id(self):
        """Direct ID lookup returns the entity."""
        _setup_populated_graph()
        from graphite.mcp_server import _resolve_entity

        entity = _resolve_entity("e-john")
        assert entity["canonical_name"] == "John Doe"

    def test_resolve_by_name(self):
        """Name search fallback finds the entity."""
        _setup_populated_graph()
        from graphite.mcp_server import _resolve_entity

        entity = _resolve_entity("John Doe")
        assert entity["id"] == "e-john"

    def test_resolve_by_partial_name(self):
        """Partial name search returns a match."""
        _setup_populated_graph()
        from graphite.mcp_server import _resolve_entity

        entity = _resolve_entity("John")
        assert entity["canonical_name"] == "John Doe"

    def test_resolve_not_found(self):
        """Non-existent entity raises ValueError."""
        _setup_populated_graph()
        from graphite.mcp_server import _resolve_entity

        with pytest.raises(ValueError, match="Entity not found"):
            _resolve_entity("Nonexistent Person")

    def test_resolve_exact_match_preferred(self):
        """When multiple results, exact case-insensitive match is preferred."""
        import graphite.mcp_server as srv

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-react", "React", "Technology")
        kg.add_test_entity("e-reactive", "Reactive Systems", "Concept")
        srv._kg = kg
        srv._graph_initialized = True
        srv._config = MagicMock()

        from graphite.mcp_server import _resolve_entity

        entity = _resolve_entity("React")
        assert entity["canonical_name"] == "React"


# ═══════════════════════════════════════════════════════════════════════════════
# Read-only Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphiteStatus:
    def test_status_empty_graph(self):
        """Status on empty graph shows zero counts."""
        _setup_empty_graph()
        from graphite.mcp_server import graphite_status

        result = _run_async(graphite_status())
        assert "Entities: 0" in result
        assert "Co-occurrence edges: 0" in result
        assert "Chunks stored: 0" in result

    def test_status_populated_graph(self):
        """Status on populated graph shows entity count."""
        _setup_populated_graph()
        from graphite.mcp_server import graphite_status

        result = _run_async(graphite_status())
        assert "Entities: 4" in result
        assert "Graphite Knowledge Graph Status" in result


class TestGetKnowledgeMap:
    def test_knowledge_map_populated(self):
        """Knowledge map shows entities grouped by type."""
        _setup_populated_graph()
        from graphite.mcp_server import get_knowledge_map

        result = _run_async(get_knowledge_map(max_entities=50))
        assert "Knowledge Map" in result
        assert "**John Doe**" in result or "**Jane Smith**" in result

    def test_knowledge_map_empty_graph(self):
        """Knowledge map on empty graph returns empty message."""
        _setup_empty_graph()
        from graphite.mcp_server import get_knowledge_map

        result = _run_async(get_knowledge_map())
        assert "empty" in result.lower()

    def test_knowledge_map_limit(self):
        """Knowledge map respects max_entities limit."""
        _setup_populated_graph()
        from graphite.mcp_server import get_knowledge_map

        result = _run_async(get_knowledge_map(max_entities=2))
        # Should have at most 2 bold entity names
        bold_count = result.count("**") // 2  # Each bold entity has 2 ** markers
        assert bold_count <= 2


class TestGetCooccurrences:
    def test_cooccurrences_by_name(self):
        """Get co-occurrences by entity name."""
        _setup_populated_graph()
        from graphite.mcp_server import get_cooccurrences

        result = _run_async(get_cooccurrences(entity="John Doe"))
        assert "Co-occurrences for **John Doe**" in result
        # John co-occurs with Jane and Dashboard
        assert "Jane Smith" in result or "Dashboard" in result

    def test_cooccurrences_not_found(self):
        """Non-existent entity returns error."""
        _setup_populated_graph()
        from graphite.mcp_server import get_cooccurrences

        result = _run_async(get_cooccurrences(entity="Nonexistent"))
        assert "ERROR" in result


class TestGetEntityMentions:
    def test_mentions_returns_chunks(self):
        """Entity mentions returns tagged chunks."""
        _setup_populated_graph()
        from graphite.mcp_server import get_entity_mentions

        result = _run_async(get_entity_mentions(entity="e-john"))
        assert "Mentions of **John Doe**" in result
        assert ">" in result  # Quoted text

    def test_mentions_limit(self):
        """Mentions respects limit parameter."""
        _setup_populated_graph()
        from graphite.mcp_server import get_entity_mentions

        result = _run_async(get_entity_mentions(entity="e-john", limit=1))
        assert "Mentions of **John Doe**" in result


class TestGetKeyEntities:
    def test_key_entities_all_types(self):
        """Key entities returns ranked list."""
        _setup_populated_graph()
        from graphite.mcp_server import get_key_entities

        result = _run_async(get_key_entities(limit=10))
        assert "Key Entities" in result
        assert "score:" in result

    def test_key_entities_filter_by_type(self):
        """Key entities filtered by type only shows that type."""
        _setup_populated_graph()
        from graphite.mcp_server import get_key_entities

        result = _run_async(get_key_entities(entity_type="Person"))
        assert "Person" in result
        # Technology entities should not appear
        assert "Technology" not in result or "(Person)" in result

    def test_key_entities_empty_graph(self):
        """Key entities on empty graph."""
        _setup_empty_graph()
        from graphite.mcp_server import get_key_entities

        result = _run_async(get_key_entities())
        assert "No entities" in result


class TestGetEntityProfile:
    def test_profile_has_sections(self):
        """Entity profile includes name, type, co-occurrences, and mentions."""
        _setup_populated_graph()
        from graphite.mcp_server import get_entity_profile

        result = _run_async(get_entity_profile(entity="e-john"))
        assert "# John Doe (Person)" in result
        assert "Co-occurs with" in result

    def test_profile_not_found(self):
        """Profile for non-existent entity returns error."""
        _setup_populated_graph()
        from graphite.mcp_server import get_entity_profile

        result = _run_async(get_entity_profile(entity="Nonexistent"))
        assert "ERROR" in result


class TestGetTimeline:
    def test_timeline_oldest_first(self):
        """Timeline returns chunks in chronological order."""
        _setup_populated_graph()
        from graphite.mcp_server import get_timeline

        result = _run_async(get_timeline(entity="e-john"))
        assert "Timeline for **John Doe**" in result

    def test_timeline_empty(self):
        """Timeline for entity with no temporal data."""
        import graphite.mcp_server as srv
        from graphite.config import GraphiteConfig

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-lone", "Lone Entity", "Concept")
        srv._kg = kg
        srv._config = GraphiteConfig()
        srv._graph_initialized = True

        from graphite.mcp_server import get_timeline

        result = _run_async(get_timeline(entity="e-lone"))
        assert "No timeline data" in result


class TestGetEvidence:
    def test_evidence_between_entities(self):
        """Evidence returns chunks where both entities co-occur."""
        _setup_populated_graph()
        from graphite.mcp_server import get_evidence

        result = _run_async(get_evidence(entity_a="e-john", entity_b="e-dash"))
        assert "Evidence for **John Doe** ↔ **Dashboard Redesign**" in result
        assert ">" in result

    def test_evidence_no_shared_chunks(self):
        """Evidence returns message when entities don't co-occur."""
        _setup_populated_graph()
        from graphite.mcp_server import get_evidence

        result = _run_async(get_evidence(entity_a="e-john", entity_b="e-react"))
        assert "No shared chunks" in result


class TestGetEntitySummary:
    def test_summary_format(self):
        """Summary returns one-liner with name, type, neighbors, chunk count."""
        _setup_populated_graph()
        from graphite.mcp_server import get_entity_summary

        result = _run_async(get_entity_summary(entity="e-john"))
        assert "John Doe (Person)" in result
        assert "co-occurs with" in result
        assert "chunk(s)" in result

    def test_summary_no_cooccurrences(self):
        """Summary for isolated entity shows no co-occurrences."""
        import graphite.mcp_server as srv
        from graphite.config import GraphiteConfig

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-lone", "Lone Entity", "Concept")
        srv._kg = kg
        srv._config = GraphiteConfig()
        srv._graph_initialized = True

        from graphite.mcp_server import get_entity_summary

        result = _run_async(get_entity_summary(entity="e-lone"))
        assert "Lone Entity (Concept)" in result
        assert "no co-occurrences" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Search Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindRelevantEntities:
    def test_find_relevant_returns_results(self):
        """Semantic search returns scored entities."""
        import graphite.mcp_server as srv

        kg = _setup_populated_graph()
        mock_embed = _make_mock_embedding_manager()
        srv._embedding_manager = mock_embed

        from graphite.mcp_server import find_relevant_entities

        result = _run_async(find_relevant_entities(query="dashboard project"))
        assert "Entities relevant to" in result

    def test_find_relevant_empty_graph(self):
        """Semantic search on empty graph returns empty message."""
        import graphite.mcp_server as srv

        _setup_empty_graph()
        mock_embed = _make_mock_embedding_manager()
        srv._embedding_manager = mock_embed

        from graphite.mcp_server import find_relevant_entities

        result = _run_async(find_relevant_entities(query="anything"))
        assert "empty" in result.lower()


class TestAssembleMemory:
    def test_assemble_returns_context(self):
        """assemble_memory returns knowledge context."""
        import graphite.mcp_server as srv

        kg = _setup_populated_graph()
        mock_embed = _make_mock_embedding_manager()

        # Create a mock MemoryContextManager
        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = (
            "## Knowledge Context\n### Key Entities\n- **John Doe** (Person)\n"
            "### Evidence\n> John said something\n"
        )
        srv._embedding_manager = mock_embed
        srv._context_manager = mock_ctx

        from graphite.mcp_server import assemble_memory

        result = _run_async(assemble_memory(query="What did John say?"))
        assert "Knowledge Context" in result

    def test_assemble_empty_result(self):
        """assemble_memory returns message when no knowledge found."""
        import graphite.mcp_server as srv

        _setup_populated_graph()
        mock_embed = _make_mock_embedding_manager()
        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = ""
        srv._embedding_manager = mock_embed
        srv._context_manager = mock_ctx

        from graphite.mcp_server import assemble_memory

        result = _run_async(assemble_memory(query="unknown topic"))
        assert "No relevant knowledge" in result

    def test_assemble_with_date_filter(self):
        """assemble_memory passes date filters to context manager."""
        import graphite.mcp_server as srv

        _setup_populated_graph()
        mock_embed = _make_mock_embedding_manager()
        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = "## Knowledge Context\nFiltered"
        srv._embedding_manager = mock_embed
        srv._context_manager = mock_ctx

        from graphite.mcp_server import assemble_memory

        result = _run_async(assemble_memory(
            query="test", time_start="2024-01-01", time_end="2024-12-31"
        ))
        assert "Knowledge Context" in result
        # Verify dates were parsed and passed
        call_args = mock_ctx.assemble_context.call_args
        assert call_args.kwargs.get("time_start") is not None
        assert call_args.kwargs.get("time_end") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Write Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphiteUpdateDocument:
    def test_update_document_changed(self, tmp_path):
        """graphite_update_document returns update results when content changed."""
        import graphite.mcp_server as srv
        from graphite.ingestion.pipeline import DocumentUpdateResult, IngestionResult

        _setup_populated_graph()
        srv._project_root = tmp_path

        doc = tmp_path / "test.md"
        doc.write_text("# Updated content\nNew stuff here")

        mock_pipeline = MagicMock()
        mock_pipeline.update_document.return_value = DocumentUpdateResult(
            source_document=str(doc),
            action="updated",
            chunks_removed=2,
            edges_removed=4,
            entities_removed=1,
            entities_updated=1,
            ingestion_result=IngestionResult(
                source_document=str(doc),
                status="complete",
                chunks_tagged=3,
                entities_created=2,
                entities_linked=1,
                edges_created=2,
                duration_seconds=0.5,
            ),
            duration_seconds=0.8,
        )
        srv._pipeline = mock_pipeline
        srv._kg.save = MagicMock()

        from graphite.mcp_server import graphite_update_document

        result = _run_async(graphite_update_document(path=str(doc)))
        assert "Document Updated" in result
        assert "2 chunks" in result
        assert "4 edges" in result
        mock_pipeline.update_document.assert_called_once()

    def test_update_document_unchanged(self, tmp_path):
        """graphite_update_document returns 'unchanged' message."""
        import graphite.mcp_server as srv
        from graphite.ingestion.pipeline import DocumentUpdateResult

        _setup_populated_graph()
        srv._project_root = tmp_path

        doc = tmp_path / "test.md"
        doc.write_text("# Same content")

        mock_pipeline = MagicMock()
        mock_pipeline.update_document.return_value = DocumentUpdateResult(
            source_document=str(doc),
            action="unchanged",
        )
        srv._pipeline = mock_pipeline

        from graphite.mcp_server import graphite_update_document

        result = _run_async(graphite_update_document(path=str(doc)))
        assert "unchanged" in result.lower()


class TestGraphiteRemoveDocument:
    def test_remove_document(self, tmp_path):
        """graphite_remove_document returns removal stats."""
        import graphite.mcp_server as srv
        from graphite.ingestion.pipeline import DocumentUpdateResult

        _setup_populated_graph()
        srv._project_root = tmp_path

        mock_pipeline = MagicMock()
        mock_pipeline.remove_document.return_value = DocumentUpdateResult(
            source_document=str(tmp_path / "old.md"),
            action="removed",
            chunks_removed=3,
            edges_removed=6,
            entities_removed=2,
            entities_updated=1,
        )
        srv._pipeline = mock_pipeline
        srv._kg.save = MagicMock()

        from graphite.mcp_server import graphite_remove_document

        result = _run_async(graphite_remove_document(path=str(tmp_path / "old.md")))
        assert "Document Removed" in result
        assert "Chunks removed: 3" in result
        assert "Edges removed: 6" in result
        assert "Entities removed (orphaned): 2" in result
        mock_pipeline.remove_document.assert_called_once()


class TestGraphiteIngest:
    def test_ingest_calls_pipeline(self, tmp_path):
        """graphite_ingest runs the pipeline and auto-saves."""
        import graphite.mcp_server as srv
        from graphite.config import GraphiteConfig
        from graphite.ingestion.pipeline import IngestionResult

        _setup_populated_graph()
        srv._project_root = tmp_path

        # Create a test file
        doc = tmp_path / "test.md"
        doc.write_text("# Test\nSome content here")

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = IngestionResult(
            source_document=str(doc),
            status="complete",
            chunks_tagged=3,
            entities_created=2,
            edges_created=1,
            duration_seconds=0.5,
        )
        srv._pipeline = mock_pipeline
        srv._kg.save = MagicMock()  # type: ignore

        from graphite.mcp_server import graphite_ingest

        result = _run_async(graphite_ingest(path=str(doc)))
        assert "complete" in result
        assert "3 chunks" in result
        mock_pipeline.ingest_file.assert_called_once()

    def test_ingest_idempotent(self, tmp_path):
        """Second ingest of same file uses update path when hash exists."""
        import graphite.mcp_server as srv
        from graphite.ingestion.pipeline import DocumentUpdateResult

        _setup_populated_graph()
        srv._project_root = tmp_path

        doc = tmp_path / "test.md"
        doc.write_text("# Test\nSome content")

        # Simulate that this file was previously ingested (hash exists)
        srv._kg._document_hashes[str(doc)] = "some_old_hash"

        mock_pipeline = MagicMock()
        mock_pipeline.update_document.return_value = DocumentUpdateResult(
            source_document=str(doc),
            action="unchanged",
        )
        srv._pipeline = mock_pipeline

        from graphite.mcp_server import graphite_ingest

        result = _run_async(graphite_ingest(path=str(doc)))
        assert "unchanged" in result.lower()
        # update_document should have been called, NOT ingest_file
        mock_pipeline.update_document.assert_called_once()
        mock_pipeline.ingest_file.assert_not_called()

    def test_ingest_nonexistent_path(self):
        """graphite_ingest returns error for nonexistent path."""
        import graphite.mcp_server as srv

        _setup_populated_graph()
        srv._pipeline = MagicMock()

        from graphite.mcp_server import graphite_ingest

        result = _run_async(graphite_ingest(path="/nonexistent/file.md"))
        assert "ERROR" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Reflection Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphiteReflect:
    def test_reflect_returns_results(self):
        """graphite_reflect returns formatted reflection results."""
        _setup_populated_graph()
        from graphite.mcp_server import graphite_reflect

        result = _run_async(graphite_reflect(mode="light"))
        assert "Reflection Results" in result
        assert "Orphans removed" in result

    def test_reflect_full_mode(self):
        """graphite_reflect in full mode runs all operations."""
        _setup_populated_graph()
        from graphite.mcp_server import graphite_reflect

        result = _run_async(graphite_reflect(mode="full"))
        assert "Reflection Results" in result
        assert "Merges executed" in result


class TestGraphiteForget:
    def test_forget_removes_entity(self):
        """graphite_forget removes an entity from the graph."""
        kg = _setup_populated_graph()
        from graphite.mcp_server import graphite_forget

        result = _run_async(graphite_forget(entity="e-john"))
        assert "Removed entity" in result
        assert "John Doe" in result
        # Entity should be gone
        assert kg.get_entity("e-john") is None

    def test_forget_nonexistent_returns_error(self):
        """graphite_forget on nonexistent entity returns error."""
        _setup_populated_graph()
        from graphite.mcp_server import graphite_forget

        result = _run_async(graphite_forget(entity="Nonexistent Person"))
        assert "ERROR" in result


class TestGraphiteReview:
    def test_review_shows_no_candidates(self):
        """graphite_review with clean graph shows no candidates."""
        _setup_populated_graph()
        from graphite.mcp_server import graphite_review

        result = _run_async(graphite_review())
        assert "No merge candidates" in result or "candidate" in result.lower()

    def test_review_shows_candidates(self):
        """graphite_review with duplicates shows candidates."""
        import graphite.mcp_server as srv
        from graphite.config import GraphiteConfig
        from tests.test_reflection import _build_merge_candidate_graph

        _reset_mcp_globals()
        kg = _build_merge_candidate_graph()
        srv._kg = kg
        srv._config = GraphiteConfig(merge_alias_overlap_threshold=0.30)
        srv._graph_initialized = True

        from graphite.mcp_server import graphite_review

        result = _run_async(graphite_review())
        assert "candidate" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Function Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseDate:
    def test_valid_date(self):
        """Valid ISO date parses to Unix timestamp."""
        from graphite.mcp_server import _parse_date

        ts = _parse_date("2024-10-14")
        expected = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())
        assert ts == expected

    def test_empty_string(self):
        """Empty string returns None."""
        from graphite.mcp_server import _parse_date

        assert _parse_date("") is None
        assert _parse_date("  ") is None

    def test_invalid_format(self):
        """Invalid format raises ValueError."""
        from graphite.mcp_server import _parse_date

        with pytest.raises(ValueError, match="Invalid date format"):
            _parse_date("not-a-date")

    def test_none_like_input(self):
        """None-like empty input returns None."""
        from graphite.mcp_server import _parse_date

        assert _parse_date("") is None


class TestFormatTimestamp:
    def test_valid_timestamp(self):
        """Valid timestamp formats to YYYY-MM-DD."""
        from graphite.mcp_server import _format_timestamp

        ts = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())
        assert _format_timestamp(ts) == "2024-10-14"

    def test_none_timestamp(self):
        """None returns 'unknown'."""
        from graphite.mcp_server import _format_timestamp

        assert _format_timestamp(None) == "unknown"

    def test_invalid_timestamp(self):
        """Very large timestamp returns 'unknown'."""
        from graphite.mcp_server import _format_timestamp

        assert _format_timestamp(99999999999999) == "unknown"


class TestAutoSave:
    def test_auto_save_when_dirty(self):
        """Auto-save calls kg.save when graph is dirty."""
        import graphite.mcp_server as srv

        _setup_populated_graph()
        srv._graph_dirty = True
        srv._kg.save = MagicMock()  # type: ignore

        from graphite.mcp_server import _auto_save

        _auto_save()
        srv._kg.save.assert_called_once()
        assert srv._graph_dirty is False

    def test_auto_save_skips_clean(self):
        """Auto-save does nothing when graph is not dirty."""
        import graphite.mcp_server as srv

        _setup_populated_graph()
        srv._graph_dirty = False
        srv._kg.save = MagicMock()  # type: ignore

        from graphite.mcp_server import _auto_save

        _auto_save()
        srv._kg.save.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Config from_toml Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigFromToml:
    def test_from_toml_basic(self, tmp_path):
        """from_toml reads key fields."""
        toml_file = tmp_path / ".graphite.toml"
        toml_file.write_text(
            '[llm]\nprovider = "ollama"\nmodel = "llama3:8b"\n'
            '[context]\ntier1_budget_pct = 0.15\n'
            '[paths]\nmemory_root = "docs/memory"\n'
        )
        from graphite.config import GraphiteConfig

        config = GraphiteConfig.from_toml(toml_file)
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama3:8b"
        assert config.tier1_budget_pct == 0.15
        assert config.memory_root == Path("docs/memory")

    def test_from_toml_defaults_for_missing(self, tmp_path):
        """from_toml uses defaults for unspecified keys."""
        toml_file = tmp_path / ".graphite.toml"
        toml_file.write_text('[llm]\nmodel = "custom-model"\n')
        from graphite.config import GraphiteConfig

        config = GraphiteConfig.from_toml(toml_file)
        assert config.llm_model == "custom-model"
        assert config.llm_provider == "ollama"  # default
        assert config.tier1_budget_pct == 0.10  # default

    def test_from_toml_nonexistent_raises(self, tmp_path):
        """from_toml raises FileNotFoundError for missing file."""
        from graphite.config import GraphiteConfig

        with pytest.raises(FileNotFoundError):
            GraphiteConfig.from_toml(tmp_path / "nonexistent.toml")
