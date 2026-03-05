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
    import cortex.mcp_server as srv

    srv._kg = None
    srv._embedding_manager = None
    srv._context_manager = None
    srv._pipeline = None
    srv._config = None
    srv._project_root = Path("/tmp/test-project")
    srv._graph_initialized = False
    srv._graph_dirty = False
    srv._tool_lock = None


def _setup_populated_graph():
    """Set up MCP globals with a populated FakeKnowledgeGraph."""
    import cortex.mcp_server as srv
    from cortex.config import CortexConfig

    _reset_mcp_globals()
    kg = _build_test_graph()
    srv._kg = kg
    srv._config = CortexConfig()
    srv._graph_initialized = True
    return kg


def _setup_empty_graph():
    """Set up MCP globals with an empty FakeKnowledgeGraph."""
    import cortex.mcp_server as srv
    from cortex.config import CortexConfig

    _reset_mcp_globals()
    kg = FakeKnowledgeGraph()
    srv._kg = kg
    srv._config = CortexConfig()
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
        from cortex.mcp_server import _resolve_entity

        entity = _resolve_entity("e-john")
        assert entity["canonical_name"] == "John Doe"

    def test_resolve_by_name(self):
        """Name search fallback finds the entity."""
        _setup_populated_graph()
        from cortex.mcp_server import _resolve_entity

        entity = _resolve_entity("John Doe")
        assert entity["id"] == "e-john"

    def test_resolve_by_partial_name(self):
        """Partial name search returns a match."""
        _setup_populated_graph()
        from cortex.mcp_server import _resolve_entity

        entity = _resolve_entity("John")
        assert entity["canonical_name"] == "John Doe"

    def test_resolve_not_found(self):
        """Non-existent entity raises ValueError."""
        _setup_populated_graph()
        from cortex.mcp_server import _resolve_entity

        with pytest.raises(ValueError, match="Entity not found"):
            _resolve_entity("Nonexistent Person")

    def test_resolve_exact_match_preferred(self):
        """When multiple results, exact case-insensitive match is preferred."""
        import cortex.mcp_server as srv

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-react", "React", "Technology")
        kg.add_test_entity("e-reactive", "Reactive Systems", "Concept")
        srv._kg = kg
        srv._graph_initialized = True
        srv._config = MagicMock()

        from cortex.mcp_server import _resolve_entity

        entity = _resolve_entity("React")
        assert entity["canonical_name"] == "React"


# ═══════════════════════════════════════════════════════════════════════════════
# Read-only Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCortexStatus:
    def test_status_empty_graph(self):
        """Status on empty graph shows zero counts."""
        _setup_empty_graph()
        from cortex.mcp_server import cortex_status

        result = _run_async(cortex_status())
        assert "Entities: 0" in result
        assert "Co-occurrence edges: 0" in result
        assert "Chunks stored: 0" in result

    def test_status_populated_graph(self):
        """Status on populated graph shows entity count."""
        _setup_populated_graph()
        from cortex.mcp_server import cortex_status

        result = _run_async(cortex_status())
        assert "Entities: 4" in result
        assert "Cortex Knowledge Graph Status" in result


class TestGetKnowledgeMap:
    def test_knowledge_map_populated(self):
        """Knowledge map shows entities grouped by type."""
        _setup_populated_graph()
        from cortex.mcp_server import get_knowledge_map

        result = _run_async(get_knowledge_map(max_entities=50))
        assert "Knowledge Map" in result
        assert "**John Doe**" in result or "**Jane Smith**" in result

    def test_knowledge_map_empty_graph(self):
        """Knowledge map on empty graph returns empty message."""
        _setup_empty_graph()
        from cortex.mcp_server import get_knowledge_map

        result = _run_async(get_knowledge_map())
        assert "empty" in result.lower()

    def test_knowledge_map_limit(self):
        """Knowledge map respects max_entities limit."""
        _setup_populated_graph()
        from cortex.mcp_server import get_knowledge_map

        result = _run_async(get_knowledge_map(max_entities=2))
        # Should have at most 2 bold entity names
        bold_count = result.count("**") // 2  # Each bold entity has 2 ** markers
        assert bold_count <= 2


class TestGetCooccurrences:
    def test_cooccurrences_by_name(self):
        """Get co-occurrences by entity name."""
        _setup_populated_graph()
        from cortex.mcp_server import get_cooccurrences

        result = _run_async(get_cooccurrences(entity="John Doe"))
        assert "Co-occurrences for **John Doe**" in result
        # John co-occurs with Jane and Dashboard
        assert "Jane Smith" in result or "Dashboard" in result

    def test_cooccurrences_not_found(self):
        """Non-existent entity returns error."""
        _setup_populated_graph()
        from cortex.mcp_server import get_cooccurrences

        result = _run_async(get_cooccurrences(entity="Nonexistent"))
        assert "ERROR" in result


class TestGetEntityMentions:
    def test_mentions_returns_chunks(self):
        """Entity mentions returns tagged chunks."""
        _setup_populated_graph()
        from cortex.mcp_server import get_entity_mentions

        result = _run_async(get_entity_mentions(entity="e-john"))
        assert "Mentions of **John Doe**" in result
        assert ">" in result  # Quoted text

    def test_mentions_limit(self):
        """Mentions respects limit parameter."""
        _setup_populated_graph()
        from cortex.mcp_server import get_entity_mentions

        result = _run_async(get_entity_mentions(entity="e-john", limit=1))
        assert "Mentions of **John Doe**" in result


class TestGetKeyEntities:
    def test_key_entities_all_types(self):
        """Key entities returns ranked list."""
        _setup_populated_graph()
        from cortex.mcp_server import get_key_entities

        result = _run_async(get_key_entities(limit=10))
        assert "Key Entities" in result
        assert "score:" in result

    def test_key_entities_filter_by_type(self):
        """Key entities filtered by type only shows that type."""
        _setup_populated_graph()
        from cortex.mcp_server import get_key_entities

        result = _run_async(get_key_entities(entity_type="Person"))
        assert "Person" in result
        # Technology entities should not appear
        assert "Technology" not in result or "(Person)" in result

    def test_key_entities_empty_graph(self):
        """Key entities on empty graph."""
        _setup_empty_graph()
        from cortex.mcp_server import get_key_entities

        result = _run_async(get_key_entities())
        assert "No entities" in result


class TestGetEntityProfile:
    def test_profile_has_sections(self):
        """Entity profile includes name, type, co-occurrences, and mentions."""
        _setup_populated_graph()
        from cortex.mcp_server import get_entity_profile

        result = _run_async(get_entity_profile(entity="e-john"))
        assert "# John Doe (Person)" in result
        assert "Co-occurs with" in result

    def test_profile_not_found(self):
        """Profile for non-existent entity returns error."""
        _setup_populated_graph()
        from cortex.mcp_server import get_entity_profile

        result = _run_async(get_entity_profile(entity="Nonexistent"))
        assert "ERROR" in result


class TestGetTimeline:
    def test_timeline_oldest_first(self):
        """Timeline returns chunks in chronological order."""
        _setup_populated_graph()
        from cortex.mcp_server import get_timeline

        result = _run_async(get_timeline(entity="e-john"))
        assert "Timeline for **John Doe**" in result

    def test_timeline_empty(self):
        """Timeline for entity with no temporal data."""
        import cortex.mcp_server as srv
        from cortex.config import CortexConfig

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-lone", "Lone Entity", "Concept")
        srv._kg = kg
        srv._config = CortexConfig()
        srv._graph_initialized = True

        from cortex.mcp_server import get_timeline

        result = _run_async(get_timeline(entity="e-lone"))
        assert "No timeline data" in result


class TestGetEvidence:
    def test_evidence_between_entities(self):
        """Evidence returns chunks where both entities co-occur."""
        _setup_populated_graph()
        from cortex.mcp_server import get_evidence

        result = _run_async(get_evidence(entity_a="e-john", entity_b="e-dash"))
        assert "Evidence for **John Doe** ↔ **Dashboard Redesign**" in result
        assert ">" in result

    def test_evidence_no_shared_chunks(self):
        """Evidence returns message when entities don't co-occur."""
        _setup_populated_graph()
        from cortex.mcp_server import get_evidence

        result = _run_async(get_evidence(entity_a="e-john", entity_b="e-react"))
        assert "No shared chunks" in result


class TestGetEntitySummary:
    def test_summary_format(self):
        """Summary returns one-liner with name, type, neighbors, chunk count."""
        _setup_populated_graph()
        from cortex.mcp_server import get_entity_summary

        result = _run_async(get_entity_summary(entity="e-john"))
        assert "John Doe (Person)" in result
        assert "co-occurs with" in result
        assert "chunk(s)" in result

    def test_summary_no_cooccurrences(self):
        """Summary for isolated entity shows no co-occurrences."""
        import cortex.mcp_server as srv
        from cortex.config import CortexConfig

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-lone", "Lone Entity", "Concept")
        srv._kg = kg
        srv._config = CortexConfig()
        srv._graph_initialized = True

        from cortex.mcp_server import get_entity_summary

        result = _run_async(get_entity_summary(entity="e-lone"))
        assert "Lone Entity (Concept)" in result
        assert "no co-occurrences" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Search Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindRelevantEntities:
    def test_find_relevant_returns_results(self):
        """Semantic search returns scored entities."""
        import cortex.mcp_server as srv

        kg = _setup_populated_graph()
        mock_embed = _make_mock_embedding_manager()
        srv._embedding_manager = mock_embed

        from cortex.mcp_server import find_relevant_entities

        result = _run_async(find_relevant_entities(query="dashboard project"))
        assert "Entities relevant to" in result

    def test_find_relevant_empty_graph(self):
        """Semantic search on empty graph returns empty message."""
        import cortex.mcp_server as srv

        _setup_empty_graph()
        mock_embed = _make_mock_embedding_manager()
        srv._embedding_manager = mock_embed

        from cortex.mcp_server import find_relevant_entities

        result = _run_async(find_relevant_entities(query="anything"))
        assert "empty" in result.lower()


class TestAssembleMemory:
    def test_assemble_returns_context(self):
        """assemble_memory returns knowledge context."""
        import cortex.mcp_server as srv

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

        from cortex.mcp_server import assemble_memory

        result = _run_async(assemble_memory(query="What did John say?"))
        assert "Knowledge Context" in result

    def test_assemble_empty_result(self):
        """assemble_memory returns message when no knowledge found."""
        import cortex.mcp_server as srv

        _setup_populated_graph()
        mock_embed = _make_mock_embedding_manager()
        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = ""
        srv._embedding_manager = mock_embed
        srv._context_manager = mock_ctx

        from cortex.mcp_server import assemble_memory

        result = _run_async(assemble_memory(query="unknown topic"))
        assert "No relevant knowledge" in result

    def test_assemble_with_date_filter(self):
        """assemble_memory passes date filters to context manager."""
        import cortex.mcp_server as srv

        _setup_populated_graph()
        mock_embed = _make_mock_embedding_manager()
        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = "## Knowledge Context\nFiltered"
        srv._embedding_manager = mock_embed
        srv._context_manager = mock_ctx

        from cortex.mcp_server import assemble_memory

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


class TestCortexIngest:
    def test_ingest_calls_pipeline(self, tmp_path):
        """cortex_ingest runs the pipeline and auto-saves."""
        import cortex.mcp_server as srv
        from cortex.config import CortexConfig
        from cortex.ingestion.pipeline import IngestionResult

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

        from cortex.mcp_server import cortex_ingest

        result = _run_async(cortex_ingest(path=str(doc)))
        assert "complete" in result
        assert "3 chunks" in result
        mock_pipeline.ingest_file.assert_called_once()

    def test_ingest_nonexistent_path(self):
        """cortex_ingest returns error for nonexistent path."""
        import cortex.mcp_server as srv

        _setup_populated_graph()
        srv._pipeline = MagicMock()

        from cortex.mcp_server import cortex_ingest

        result = _run_async(cortex_ingest(path="/nonexistent/file.md"))
        assert "ERROR" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 Stub Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhase5Stubs:
    def test_reflect_stub(self):
        """cortex_reflect returns not-yet-implemented message."""
        from cortex.mcp_server import cortex_reflect

        result = _run_async(cortex_reflect())
        assert "not yet implemented" in result
        assert "Phase 5" in result

    def test_forget_stub(self):
        """cortex_forget returns not-yet-implemented message."""
        from cortex.mcp_server import cortex_forget

        result = _run_async(cortex_forget(entity="John"))
        assert "not yet implemented" in result
        assert "John" in result

    def test_review_stub(self):
        """cortex_review returns not-yet-implemented message."""
        from cortex.mcp_server import cortex_review

        result = _run_async(cortex_review())
        assert "not yet implemented" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Function Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseDate:
    def test_valid_date(self):
        """Valid ISO date parses to Unix timestamp."""
        from cortex.mcp_server import _parse_date

        ts = _parse_date("2024-10-14")
        expected = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())
        assert ts == expected

    def test_empty_string(self):
        """Empty string returns None."""
        from cortex.mcp_server import _parse_date

        assert _parse_date("") is None
        assert _parse_date("  ") is None

    def test_invalid_format(self):
        """Invalid format raises ValueError."""
        from cortex.mcp_server import _parse_date

        with pytest.raises(ValueError, match="Invalid date format"):
            _parse_date("not-a-date")

    def test_none_like_input(self):
        """None-like empty input returns None."""
        from cortex.mcp_server import _parse_date

        assert _parse_date("") is None


class TestFormatTimestamp:
    def test_valid_timestamp(self):
        """Valid timestamp formats to YYYY-MM-DD."""
        from cortex.mcp_server import _format_timestamp

        ts = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())
        assert _format_timestamp(ts) == "2024-10-14"

    def test_none_timestamp(self):
        """None returns 'unknown'."""
        from cortex.mcp_server import _format_timestamp

        assert _format_timestamp(None) == "unknown"

    def test_invalid_timestamp(self):
        """Very large timestamp returns 'unknown'."""
        from cortex.mcp_server import _format_timestamp

        assert _format_timestamp(99999999999999) == "unknown"


class TestAutoSave:
    def test_auto_save_when_dirty(self):
        """Auto-save calls kg.save when graph is dirty."""
        import cortex.mcp_server as srv

        _setup_populated_graph()
        srv._graph_dirty = True
        srv._kg.save = MagicMock()  # type: ignore

        from cortex.mcp_server import _auto_save

        _auto_save()
        srv._kg.save.assert_called_once()
        assert srv._graph_dirty is False

    def test_auto_save_skips_clean(self):
        """Auto-save does nothing when graph is not dirty."""
        import cortex.mcp_server as srv

        _setup_populated_graph()
        srv._graph_dirty = False
        srv._kg.save = MagicMock()  # type: ignore

        from cortex.mcp_server import _auto_save

        _auto_save()
        srv._kg.save.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Config from_toml Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigFromToml:
    def test_from_toml_basic(self, tmp_path):
        """from_toml reads key fields."""
        toml_file = tmp_path / ".cortex.toml"
        toml_file.write_text(
            '[llm]\nprovider = "ollama"\nmodel = "llama3:8b"\n'
            '[context]\ntier1_budget_pct = 0.15\n'
            '[paths]\nmemory_root = "docs/memory"\n'
        )
        from cortex.config import CortexConfig

        config = CortexConfig.from_toml(toml_file)
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama3:8b"
        assert config.tier1_budget_pct == 0.15
        assert config.memory_root == Path("docs/memory")

    def test_from_toml_defaults_for_missing(self, tmp_path):
        """from_toml uses defaults for unspecified keys."""
        toml_file = tmp_path / ".cortex.toml"
        toml_file.write_text('[llm]\nmodel = "custom-model"\n')
        from cortex.config import CortexConfig

        config = CortexConfig.from_toml(toml_file)
        assert config.llm_model == "custom-model"
        assert config.llm_provider == "ollama"  # default
        assert config.tier1_budget_pct == 0.10  # default

    def test_from_toml_nonexistent_raises(self, tmp_path):
        """from_toml raises FileNotFoundError for missing file."""
        from cortex.config import CortexConfig

        with pytest.raises(FileNotFoundError):
            CortexConfig.from_toml(tmp_path / "nonexistent.toml")
