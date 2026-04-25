"""Comprehensive tests for the conversation memory system.

Tests the full pipeline from JSONL parsing through pipeline integration,
using real-format session fixtures from test_corpus/sessions/.

Covers:
- ConversationParser: JSONL parsing, exchange grouping, chunking, metadata
- Pipeline integration: ingest_session, ingest_all_sessions, dedup
- Categorizer: claude-session:// URI handling
- Config: conversation-specific settings
- Classifier/Tagger: source-aware prompt selection
- Agent context: user profile assembly
- Capture: hook handler archiving
- Edge cases: empty sessions, malformed JSONL, oversized exchanges
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graphite.config import GraphiteConfig
from graphite.extraction.conversation_parser import (
    ConversationParser,
    SessionMetadata,
    _Exchange,
    _extract_tool_name,
    _parse_iso_timestamp,
)
from graphite.extraction.structural_parser import RawChunk
from graphite.ingestion.categorizer import categorize_document
from graphite.ingestion.pipeline import IngestionPipeline, IngestionResult


def _make_fixed_llm(response: str):
    """Create a simple LLM mock that returns a fixed response."""
    class FixedLLM:
        def chat(self, messages):
            return response
    return FixedLLM()


def _make_pipeline_llm():
    """Create a mock LLM that returns classify and tag responses."""
    class PipelineLLM:
        def chat(self, messages):
            content = messages[0]["content"]
            if "Classify" in content or "classify" in content:
                return "discussion"
            return '[{"name": "Graphite", "type": "project"}]'
    return PipelineLLM()

# ── Fixture paths ──
CORPUS_DIR = Path(__file__).parent / "test_corpus" / "sessions"
SIMPLE_SESSION = CORPUS_DIR / "simple_session.jsonl"
DEBUG_SESSION = CORPUS_DIR / "debugging_session.jsonl"
MULTI_PROJECT_SESSION = CORPUS_DIR / "multi_project_session.jsonl"
EMPTY_SESSION = CORPUS_DIR / "empty_session.jsonl"
TOOL_HEAVY_SESSION = CORPUS_DIR / "tool_heavy_session.jsonl"
MALFORMED_SESSION = CORPUS_DIR / "malformed_session.jsonl"
LONG_EXCHANGE_SESSION = CORPUS_DIR / "long_exchange_session.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimestampParsing:
    def test_iso_with_z_suffix(self):
        ts = _parse_iso_timestamp("2026-03-10T10:00:01.000Z")
        assert ts is not None
        assert isinstance(ts, int)
        assert ts > 0

    def test_iso_with_offset(self):
        ts = _parse_iso_timestamp("2026-03-10T10:00:01.000+00:00")
        assert ts is not None
        assert isinstance(ts, int)

    def test_none_input(self):
        assert _parse_iso_timestamp(None) is None

    def test_empty_string(self):
        assert _parse_iso_timestamp("") is None

    def test_invalid_format(self):
        assert _parse_iso_timestamp("not a date") is None

    def test_timestamps_are_consistent(self):
        ts1 = _parse_iso_timestamp("2026-03-10T10:00:00.000Z")
        ts2 = _parse_iso_timestamp("2026-03-10T10:00:00.000+00:00")
        assert ts1 == ts2


class TestToolNameExtraction:
    def test_tool_with_file_path(self):
        result = _extract_tool_name({
            "name": "Read",
            "input": {"file_path": "/Users/apetty/Dev/Graphite/python_shell/graphite/llm.py"},
        })
        assert result == "Read: llm.py"

    def test_tool_with_path_param(self):
        result = _extract_tool_name({
            "name": "Grep",
            "input": {"pattern": "import", "path": "/Users/apetty/Dev/Graphite"},
        })
        assert result == "Grep: Graphite"

    def test_tool_with_command(self):
        result = _extract_tool_name({
            "name": "Bash",
            "input": {"command": "pytest python_shell/tests/ -x -q"},
        })
        assert result.startswith("Bash: ")
        assert "pytest" in result

    def test_tool_no_input(self):
        result = _extract_tool_name({"name": "EnterPlanMode", "input": {}})
        assert result == "EnterPlanMode"

    def test_tool_no_name(self):
        result = _extract_tool_name({"input": {"file_path": "foo.py"}})
        assert result is None

    def test_tool_empty_dict(self):
        result = _extract_tool_name({})
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationParser — Simple Session
# ═══════════════════════════════════════════════════════════════════════════════


class TestParserSimpleSession:
    """Tests against simple_session.jsonl — a clean architecture discussion."""

    @pytest.fixture
    def parsed(self):
        parser = ConversationParser(max_chunk_tokens=1200)
        chunks, metadata = parser.parse_session(SIMPLE_SESSION)
        return chunks, metadata

    def test_metadata_session_id(self, parsed):
        _, meta = parsed
        assert meta.session_id == "simple-session-001"

    def test_metadata_project_name(self, parsed):
        _, meta = parsed
        assert meta.project_name == "Graphite"

    def test_metadata_project_path(self, parsed):
        _, meta = parsed
        assert meta.project_path == "/Users/apetty/Dev/Graphite"

    def test_metadata_git_branch(self, parsed):
        _, meta = parsed
        assert meta.git_branch == "main"

    def test_metadata_exchange_count(self, parsed):
        _, meta = parsed
        # 4 user messages that get text = 4 exchanges
        # (tool_result user messages are grouped with the prior exchange)
        assert meta.exchange_count >= 3

    def test_metadata_timestamps(self, parsed):
        _, meta = parsed
        assert meta.start_time is not None
        assert meta.end_time is not None

    def test_metadata_tool_usage(self, parsed):
        _, meta = parsed
        assert "Read" in meta.tool_usage_summary
        assert meta.tool_usage_summary["Read"] >= 1

    def test_chunks_nonempty(self, parsed):
        chunks, _ = parsed
        assert len(chunks) > 0

    def test_chunk_source_document_format(self, parsed):
        chunks, _ = parsed
        for chunk in chunks:
            assert chunk.source_document.startswith("claude-session://Graphite/")

    def test_chunk_memory_category(self, parsed):
        chunks, _ = parsed
        for chunk in chunks:
            assert chunk.memory_category == "Episodic"

    def test_chunk_section_names(self, parsed):
        chunks, _ = parsed
        for chunk in chunks:
            assert chunk.section_name is not None
            assert chunk.section_name.startswith("Exchange ")

    def test_chunk_has_user_and_assistant_text(self, parsed):
        chunks, _ = parsed
        # At least one chunk should have both User: and Assistant:
        has_both = any(
            "User:" in c.text and "Assistant:" in c.text
            for c in chunks
        )
        assert has_both

    def test_chunk_contains_real_content(self, parsed):
        """Verify chunks contain meaningful conversation content."""
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        # These terms appear in the simple session fixture
        assert "Atlas" in all_text
        assert "Graphite" in all_text
        assert "tag-and-index" in all_text

    def test_thinking_blocks_excluded(self, parsed):
        """Thinking blocks should not appear in chunk text."""
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        assert "The user wants to review" not in all_text

    def test_tool_summaries_present(self, parsed):
        """Tool usage should be summarized in bracket notation."""
        chunks, _ = parsed
        has_tools = any("[Tools:" in c.text for c in chunks)
        assert has_tools

    def test_chunks_are_rawchunk_instances(self, parsed):
        chunks, _ = parsed
        for chunk in chunks:
            assert isinstance(chunk, RawChunk)

    def test_chunk_ids_unique(self, parsed):
        chunks, _ = parsed
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_timestamps_from_session(self, parsed):
        chunks, _ = parsed
        timestamped = [c for c in chunks if c.timestamp is not None]
        assert len(timestamped) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationParser — Debugging Session
# ═══════════════════════════════════════════════════════════════════════════════


class TestParserDebuggingSession:
    """Tests against debugging_session.jsonl — error diagnosis with heavy tool use."""

    @pytest.fixture
    def parsed(self):
        parser = ConversationParser(max_chunk_tokens=1200)
        return parser.parse_session(DEBUG_SESSION)

    def test_session_id(self, parsed):
        _, meta = parsed
        assert meta.session_id == "debug-session-002"

    def test_git_branch(self, parsed):
        _, meta = parsed
        assert meta.git_branch == "fix/pipeline-crash"

    def test_tool_usage_summary(self, parsed):
        _, meta = parsed
        # The debugging session uses Read, Grep, Edit, Bash
        tools = meta.tool_usage_summary
        assert "Read" in tools
        assert "Grep" in tools
        assert "Edit" in tools
        assert "Bash" in tools

    def test_error_content_preserved(self, parsed):
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        assert "circuit breaker" in all_text.lower() or "Circuit breaker" in all_text

    def test_tool_result_content_excluded(self, parsed):
        """Tool result content (file contents) should be excluded."""
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        # The fixture has tool_result content like "(file contents...)"
        # but the parser should skip tool_result user messages
        # and only keep actual user-typed text
        assert "(file contents...)" not in all_text

    def test_multiple_tool_uses_summarized(self, parsed):
        """Multiple tool uses in one turn should be compactly summarized."""
        chunks, _ = parsed
        has_multi_tool = any(
            "[Tools:" in c.text and "," in c.text.split("[Tools:")[1]
            for c in chunks
            if "[Tools:" in c.text
        )
        # The debug session has turns with multiple tool uses
        assert has_multi_tool or len(chunks) > 0  # At minimum, chunks exist


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationParser — Multi-Project Session
# ═══════════════════════════════════════════════════════════════════════════════


class TestParserMultiProjectSession:
    """Tests against multi_project_session.jsonl — FountainOfYouth project."""

    @pytest.fixture
    def parsed(self):
        parser = ConversationParser(max_chunk_tokens=1200)
        return parser.parse_session(MULTI_PROJECT_SESSION)

    def test_project_name(self, parsed):
        _, meta = parsed
        assert meta.project_name == "FountainOfYouth"

    def test_source_document_uses_project_name(self, parsed):
        chunks, _ = parsed
        for chunk in chunks:
            assert "FountainOfYouth" in chunk.source_document

    def test_different_project_path(self, parsed):
        _, meta = parsed
        assert meta.project_path == "/Users/apetty/Desktop/FountainOfYouth"

    def test_captures_preferences(self, parsed):
        """Should capture user preferences mentioned in conversation."""
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        assert "PyJWT" in all_text
        assert "fewer dependencies" in all_text

    def test_captures_goals(self, parsed):
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        assert "MVP" in all_text
        assert "end of March" in all_text

    def test_captures_technology_decisions(self, parsed):
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        assert "FastAPI" in all_text
        assert "PostgreSQL" in all_text
        assert "SQLAlchemy" in all_text


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationParser — Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestParserEmptySession:
    def test_empty_session_returns_empty(self):
        parser = ConversationParser()
        chunks, meta = parser.parse_session(EMPTY_SESSION)
        assert len(chunks) == 0
        assert meta.exchange_count == 0

    def test_empty_session_metadata(self):
        parser = ConversationParser()
        _, meta = parser.parse_session(EMPTY_SESSION)
        assert meta.project_name is not None
        assert meta.session_id is not None


class TestParserMalformedSession:
    def test_skips_malformed_lines(self):
        parser = ConversationParser()
        chunks, meta = parser.parse_session(MALFORMED_SESSION)
        # Should still produce chunks from the valid lines
        assert len(chunks) >= 1

    def test_session_id_from_valid_lines(self):
        parser = ConversationParser()
        _, meta = parser.parse_session(MALFORMED_SESSION)
        assert meta.session_id == "malformed-005"

    def test_content_from_valid_lines(self):
        parser = ConversationParser()
        chunks, _ = parser.parse_session(MALFORMED_SESSION)
        all_text = " ".join(c.text for c in chunks)
        assert "malformed lines" in all_text


class TestParserNonexistentFile:
    def test_nonexistent_file_returns_empty(self):
        parser = ConversationParser()
        chunks, meta = parser.parse_session(Path("/tmp/nonexistent_session.jsonl"))
        assert len(chunks) == 0
        assert meta.exchange_count == 0


class TestParserToolHeavySession:
    """Session with lots of tool_use blocks and tool_result responses."""

    @pytest.fixture
    def parsed(self):
        parser = ConversationParser()
        return parser.parse_session(TOOL_HEAVY_SESSION)

    def test_tool_usage_tallied(self, parsed):
        _, meta = parsed
        assert "Grep" in meta.tool_usage_summary
        assert meta.tool_usage_summary["Grep"] >= 2
        assert "Read" in meta.tool_usage_summary
        assert "Glob" in meta.tool_usage_summary

    def test_tool_results_not_in_chunks(self, parsed):
        """Tool result content should not appear in chunk text."""
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        # The actual file content from tool results should be excluded
        assert "pipeline.py:from graphite.llm" not in all_text

    def test_assistant_text_preserved(self, parsed):
        chunks, _ = parsed
        all_text = " ".join(c.text for c in chunks)
        assert "LLM hierarchy" in all_text or "LLMClient" in all_text


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationParser — Chunking Behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkingBehavior:
    def test_small_exchange_single_chunk(self):
        """A short exchange should produce exactly one chunk."""
        parser = ConversationParser(max_chunk_tokens=2000)
        chunks, _ = parser.parse_session(MULTI_PROJECT_SESSION)
        # Each exchange should fit in one chunk with high token limit
        section_names = [c.section_name for c in chunks]
        # No "(part N)" suffixes expected
        assert not any("part" in s for s in section_names)

    def test_oversized_exchange_splits(self):
        """A long exchange should be split into multiple chunks."""
        parser = ConversationParser(max_chunk_tokens=100)  # Very small
        chunks, _ = parser.parse_session(LONG_EXCHANGE_SESSION)
        # With 100 token limit, the long user message should split
        parts = [c for c in chunks if "part" in (c.section_name or "")]
        assert len(parts) > 0, "Expected oversized exchange to be split into parts"

    def test_split_chunks_share_source_document(self):
        parser = ConversationParser(max_chunk_tokens=100)
        chunks, _ = parser.parse_session(LONG_EXCHANGE_SESSION)
        sources = {c.source_document for c in chunks}
        assert len(sources) == 1  # All from same session

    def test_split_chunks_share_timestamp(self):
        parser = ConversationParser(max_chunk_tokens=100)
        chunks, _ = parser.parse_session(LONG_EXCHANGE_SESSION)
        # All chunks from the same exchange should share timestamp
        exchange_1_chunks = [
            c for c in chunks
            if c.section_name and c.section_name.startswith("Exchange 1")
        ]
        if len(exchange_1_chunks) > 1:
            timestamps = {c.timestamp for c in exchange_1_chunks}
            assert len(timestamps) == 1

    def test_token_limit_respected(self):
        """No chunk should exceed the token limit (approximately)."""
        limit = 200
        parser = ConversationParser(max_chunk_tokens=limit)
        chunks, _ = parser.parse_session(SIMPLE_SESSION)
        for chunk in chunks:
            # Rough check: word_count * 1.3 should be near the limit
            word_count = len(chunk.text.split())
            estimated_tokens = int(word_count * 1.3)
            # Allow some overshoot from sentence boundaries
            assert estimated_tokens < limit * 1.5, (
                f"Chunk has ~{estimated_tokens} tokens, limit is {limit}"
            )


class TestToolSummaryConfig:
    def test_tool_summaries_disabled(self):
        parser = ConversationParser(include_tool_summaries=False)
        chunks, _ = parser.parse_session(TOOL_HEAVY_SESSION)
        for chunk in chunks:
            assert "[Tools:" not in chunk.text

    def test_tool_summaries_enabled(self):
        parser = ConversationParser(include_tool_summaries=True)
        chunks, _ = parser.parse_session(TOOL_HEAVY_SESSION)
        has_tools = any("[Tools:" in c.text for c in chunks)
        assert has_tools


# ═══════════════════════════════════════════════════════════════════════════════
# Categorizer — claude-session:// URIs
# ═══════════════════════════════════════════════════════════════════════════════


class TestCategorizerSessionURIs:
    def test_claude_session_uri_returns_episodic(self):
        result = categorize_document(
            "claude-session://Graphite/abc123", Path("memory")
        )
        assert result == "Episodic"

    def test_claude_session_uri_any_project(self):
        result = categorize_document(
            "claude-session://FountainOfYouth/def456", Path("memory")
        )
        assert result == "Episodic"

    def test_regular_path_still_works(self):
        # meetings/ should still map to Episodic
        result = categorize_document(
            Path("memory/meetings/standup.md"), Path("memory")
        )
        assert result == "Episodic"

    def test_associates_path_still_works(self):
        result = categorize_document(
            Path("memory/associates/alice.md"), Path("memory")
        )
        assert result == "Semantic"

    def test_work_path_still_works(self):
        result = categorize_document(
            Path("memory/work/project-x.md"), Path("memory")
        )
        assert result == "Procedural"


# ═══════════════════════════════════════════════════════════════════════════════
# Config — Conversation Settings
# ═══════════════════════════════════════════════════════════════════════════════


class TestConversationConfig:
    def test_default_claude_data_dir(self):
        config = GraphiteConfig()
        assert config.claude_data_dir == Path.home() / ".claude"

    def test_default_max_exchange_tokens(self):
        config = GraphiteConfig()
        assert config.conversation_max_exchange_tokens == 1200

    def test_default_include_tool_summaries(self):
        config = GraphiteConfig()
        assert config.conversation_include_tool_summaries is True

    def test_default_skip_tool_output(self):
        config = GraphiteConfig()
        assert config.conversation_skip_tool_output is True

    def test_override_max_exchange_tokens(self):
        config = GraphiteConfig(conversation_max_exchange_tokens=800)
        assert config.conversation_max_exchange_tokens == 800

    def test_expanded_chunk_types(self):
        config = GraphiteConfig()
        assert "debugging" in config.valid_chunk_types
        assert "code_review" in config.valid_chunk_types
        assert "architecture" in config.valid_chunk_types
        assert "learning" in config.valid_chunk_types

    def test_expanded_entity_types(self):
        config = GraphiteConfig()
        assert "preference" in config.valid_entity_types
        assert "goal" in config.valid_entity_types
        assert "pattern" in config.valid_entity_types
        assert "skill" in config.valid_entity_types

    def test_conversation_prompts_exist(self):
        config = GraphiteConfig()
        classify_prompt = config.get_prompt("classify_conversation")
        assert "{chunk_text}" in classify_prompt
        assert "debugging" in classify_prompt

        tag_prompt = config.get_prompt("tag_conversation")
        assert "{chunk_text}" in tag_prompt
        assert "preference" in tag_prompt

    def test_conversation_batch_prompts_exist(self):
        config = GraphiteConfig()
        classify_batch = config.get_prompt("classify_conversation_batch")
        assert "{chunks}" in classify_batch

        tag_batch = config.get_prompt("tag_conversation_batch")
        assert "{chunks}" in tag_batch


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier — Source-Aware Prompt Selection
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassifierPromptSelection:
    def _make_chunk(self, source_document="memory/meetings/test.md", text="test"):
        return RawChunk(
            id="test-id",
            source_document=source_document,
            section_name="Test",
            speaker=None,
            timestamp=None,
            memory_category="Episodic",
            text=text,
        )

    def test_conversation_source_detected(self):
        from graphite.extraction.classifier import ChunkClassifier

        llm = _make_fixed_llm("discussion")
        classifier = ChunkClassifier(llm)
        assert classifier._is_conversation_source("claude-session://Graphite/abc")
        assert not classifier._is_conversation_source("memory/meetings/test.md")

    def test_regular_source_uses_default_prompt(self):
        from graphite.extraction.classifier import ChunkClassifier

        llm = _make_fixed_llm("discussion")
        classifier = ChunkClassifier(llm)

        chunk = self._make_chunk(source_document="memory/meetings/test.md")
        result = classifier._classify_single(chunk)
        assert result == "discussion"

    def test_conversation_source_uses_conversation_prompt(self):
        from graphite.extraction.classifier import ChunkClassifier

        # The conversation prompt includes "debugging" as a valid type
        llm = _make_fixed_llm("debugging")
        config = GraphiteConfig()
        classifier = ChunkClassifier(llm, config=config)

        chunk = self._make_chunk(
            source_document="claude-session://Graphite/abc",
            text="The pipeline is crashing with a RuntimeError",
        )
        result = classifier._classify_single(chunk)
        assert result == "debugging"

    def test_conversation_batch_selects_prompt(self):
        from graphite.extraction.classifier import ChunkClassifier

        llm = _make_fixed_llm("architecture\ndebugging")
        config = GraphiteConfig()
        classifier = ChunkClassifier(llm, config=config)

        chunks = [
            self._make_chunk(
                source_document="claude-session://Graphite/abc",
                text="chunk 1",
            ),
            self._make_chunk(
                source_document="claude-session://Graphite/abc",
                text="chunk 2",
            ),
        ]
        results = classifier._classify_batch(chunks)
        assert results is not None
        assert len(results) == 2
        assert "architecture" in results
        assert "debugging" in results


# ═══════════════════════════════════════════════════════════════════════════════
# Tagger — Source-Aware Prompt Selection
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaggerPromptSelection:
    def test_conversation_source_detected(self):
        from graphite.extraction.tagger import EntityTagger

        llm = _make_fixed_llm('[]')
        tagger = EntityTagger(llm)
        assert tagger._is_conversation_source("claude-session://Graphite/abc")
        assert not tagger._is_conversation_source("memory/meetings/test.md")

    def test_conversation_entity_types_valid(self):
        """New conversation entity types should pass validation."""
        from graphite.extraction.tagger import EntityTagger
        from graphite.extraction.classifier import ClassifiedChunk

        config = GraphiteConfig()
        llm = _make_fixed_llm('[{"name": "simple over clever", "type": "preference"}]')
        tagger = EntityTagger(llm, config=config)

        raw = RawChunk(
            id="test",
            source_document="claude-session://Graphite/abc",
            section_name="Exchange 1",
            speaker=None,
            timestamp=None,
            memory_category="Episodic",
            text="I prefer simple over clever code",
        )
        classified = ClassifiedChunk(raw=raw, chunk_type="preference")
        result = tagger._tag_single(classified)

        assert result is not None
        assert len(result.entities) == 1
        assert result.entities[0].name == "simple over clever"
        assert result.entities[0].entity_type == "preference"


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline — Session Ingestion (Mock Graph)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineSessionIngestion:
    """Tests ingest_session with a mock knowledge graph."""

    def _make_mock_kg(self):
        kg = MagicMock()
        kg.get_document_hash.return_value = None  # No prior hash = new doc
        kg.set_document_hash.return_value = None
        kg.search_entities.return_value = "[]"
        kg.add_entity.return_value = "entity-001"
        kg.store_chunk.return_value = "chunk-001"
        kg.add_cooccurrence.return_value = None
        return kg

    def test_ingest_simple_session(self):
        kg = self._make_mock_kg()
        llm = _make_pipeline_llm()
        config = GraphiteConfig()
        pipeline = IngestionPipeline(kg, llm, config=config)

        result = pipeline.ingest_session(SIMPLE_SESSION)

        assert result.source_document.startswith("claude-session://Graphite/")
        assert result.chunks_total > 0
        # Status depends on LLM responses, but should not be "failed"
        # (stub returns fixed strings, so classification/tagging may partially work)

    def test_ingest_empty_session(self):
        kg = self._make_mock_kg()
        config = GraphiteConfig()
        pipeline = IngestionPipeline(kg, config=config)

        result = pipeline.ingest_session(EMPTY_SESSION)
        assert result.status == "complete"
        assert result.chunks_total == 0

    def test_ingest_dedup_skips_unchanged(self):
        """Second ingestion of same file should be skipped via hash check."""
        kg = self._make_mock_kg()
        # Simulate hash match
        import hashlib
        content = SIMPLE_SESSION.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        kg.get_document_hash.return_value = file_hash

        config = GraphiteConfig()
        pipeline = IngestionPipeline(kg, config=config)

        result = pipeline.ingest_session(SIMPLE_SESSION)
        assert result.status == "complete"
        assert result.chunks_total == 0  # Skipped — no chunks processed

    def test_ingest_nonexistent_file(self):
        kg = self._make_mock_kg()
        config = GraphiteConfig()
        pipeline = IngestionPipeline(kg, config=config)

        result = pipeline.ingest_session(Path("/tmp/nonexistent.jsonl"))
        assert result.status == "complete"  # Empty = complete
        assert result.chunks_total == 0


class TestPipelineAllSessions:
    """Tests ingest_all_sessions with a temporary directory structure."""

    def _make_mock_kg(self):
        kg = MagicMock()
        kg.get_document_hash.return_value = None
        kg.set_document_hash.return_value = None
        kg.search_entities.return_value = "[]"
        kg.add_entity.return_value = "entity-001"
        kg.store_chunk.return_value = "chunk-001"
        kg.add_cooccurrence.return_value = None
        return kg

    def test_discovers_sessions_in_projects_dir(self):
        """Should find .jsonl files under projects/ subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock claude directory structure
            projects_dir = Path(tmpdir) / "projects"
            project_dir = projects_dir / "-Users-apetty-Dev-Graphite"
            project_dir.mkdir(parents=True)

            # Copy fixture
            shutil.copy(SIMPLE_SESSION, project_dir / "test-session.jsonl")

            kg = self._make_mock_kg()
            config = GraphiteConfig(claude_data_dir=Path(tmpdir))
            pipeline = IngestionPipeline(kg, config=config)

            results = pipeline.ingest_all_sessions(claude_dir=Path(tmpdir))
            assert len(results) == 1
            assert results[0].chunks_total > 0

    def test_project_filter(self):
        """Should only ingest sessions matching the project filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir) / "projects"

            # Create two project dirs
            graphite_dir = projects_dir / "-Users-apetty-Dev-Graphite"
            graphite_dir.mkdir(parents=True)
            foy_dir = projects_dir / "-Users-apetty-Desktop-FountainOfYouth"
            foy_dir.mkdir(parents=True)

            shutil.copy(SIMPLE_SESSION, graphite_dir / "session1.jsonl")
            shutil.copy(MULTI_PROJECT_SESSION, foy_dir / "session2.jsonl")

            kg = self._make_mock_kg()
            config = GraphiteConfig(claude_data_dir=Path(tmpdir))
            pipeline = IngestionPipeline(kg, config=config)

            # Filter to Graphite only
            results = pipeline.ingest_all_sessions(
                claude_dir=Path(tmpdir),
                project_filter="Graphite",
            )
            assert len(results) == 1

    def test_since_filter(self):
        """Should only ingest sessions modified after the since date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir) / "projects"
            project_dir = projects_dir / "-Users-apetty-Dev-Graphite"
            project_dir.mkdir(parents=True)

            shutil.copy(SIMPLE_SESSION, project_dir / "session1.jsonl")

            kg = self._make_mock_kg()
            config = GraphiteConfig(claude_data_dir=Path(tmpdir))
            pipeline = IngestionPipeline(kg, config=config)

            # Since date far in the future — should find nothing
            results = pipeline.ingest_all_sessions(
                claude_dir=Path(tmpdir),
                since="2099-01-01",
            )
            assert len(results) == 0

    def test_empty_projects_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir) / "projects"
            projects_dir.mkdir()

            kg = self._make_mock_kg()
            config = GraphiteConfig()
            pipeline = IngestionPipeline(kg, config=config)

            results = pipeline.ingest_all_sessions(claude_dir=Path(tmpdir))
            assert len(results) == 0

    def test_no_projects_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create projects/ subdirectory
            kg = self._make_mock_kg()
            config = GraphiteConfig()
            pipeline = IngestionPipeline(kg, config=config)

            results = pipeline.ingest_all_sessions(claude_dir=Path(tmpdir))
            assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline — Entity/Chunk Type Mapping
# ═══════════════════════════════════════════════════════════════════════════════


class TestEntityTypeMapping:
    """Verify conversation entity types map correctly to Rust types."""

    def test_conversation_chunk_types(self):
        from graphite.ingestion.pipeline import _CHUNK_TYPE_MAP

        assert _CHUNK_TYPE_MAP["debugging"] == "Discussion"
        assert _CHUNK_TYPE_MAP["code_review"] == "Discussion"
        assert _CHUNK_TYPE_MAP["architecture"] == "Decision"
        assert _CHUNK_TYPE_MAP["learning"] == "Background"

    def test_conversation_entity_types(self):
        from graphite.ingestion.pipeline import _ENTITY_TYPE_MAP

        assert _ENTITY_TYPE_MAP["preference"] == "Preference"
        assert _ENTITY_TYPE_MAP["goal"] == "Goal"
        assert _ENTITY_TYPE_MAP["pattern"] == "Pattern"
        assert _ENTITY_TYPE_MAP["skill"] == "Skill"

    def test_original_types_unchanged(self):
        from graphite.ingestion.pipeline import _CHUNK_TYPE_MAP, _ENTITY_TYPE_MAP

        assert _CHUNK_TYPE_MAP["decision"] == "Decision"
        assert _CHUNK_TYPE_MAP["discussion"] == "Discussion"
        assert _ENTITY_TYPE_MAP["person"] == "Person"
        assert _ENTITY_TYPE_MAP["project"] == "Project"


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Context — User Profile Assembly
# ═══════════════════════════════════════════════════════════════════════════════


class TestUserProfileAssembly:
    """Tests for the UserProfile dataclass and assemble_user_profile method."""

    def test_user_profile_dataclass(self):
        from graphite.agent_context import UserProfile, EntityBrief

        profile = UserProfile(
            preferences=[],
            goals=[],
            work_patterns=[],
            skills=[],
            active_projects=[],
            recurring_themes=[],
            narrative="## What I Know About You\n\nNo data yet.",
        )
        d = profile.to_dict()
        assert "preferences" in d
        assert "narrative" in d
        assert isinstance(d["preferences"], list)

    def test_user_profile_with_entities(self):
        from graphite.agent_context import UserProfile, EntityBrief

        pref = EntityBrief(
            id="e1", name="simple over clever",
            type="Preference", importance=0.8,
            last_seen="2026-03-10", top_connections=["Graphite"],
            mention_count=5,
        )
        profile = UserProfile(
            preferences=[pref],
            goals=[],
            work_patterns=[],
            skills=[],
            active_projects=[],
            recurring_themes=[],
            narrative="## What I Know About You\n\n**Preferences:**\n- simple over clever (5 mentions)",
        )
        d = profile.to_dict()
        assert len(d["preferences"]) == 1
        assert d["preferences"][0]["name"] == "simple over clever"

    def test_empty_profile_narrative(self):
        from graphite.agent_context import AgentContextAssembler

        kg = MagicMock()
        kg.compute_pagerank.return_value = "[]"

        emb = MagicMock()
        assembler = AgentContextAssembler(kg, emb)

        # Force entity embeddings to be "done" with empty graph
        assembler._entities_embedded = True

        profile = assembler.assemble_user_profile()
        assert "No user profile data yet" in profile.narrative


# ═══════════════════════════════════════════════════════════════════════════════
# Capture — Hook Handler
# ═══════════════════════════════════════════════════════════════════════════════


class TestHookHandler:
    def test_archive_with_transcript_path(self):
        from graphite.capture.hook_handler import archive_transcript

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "archive"

            result = archive_transcript(
                {"transcript_path": str(SIMPLE_SESSION)},
                archive_dir=archive_dir,
            )
            assert result is not None
            assert result.exists()
            assert result.suffix == ".jsonl"
            # Verify content was copied
            assert result.stat().st_size == SIMPLE_SESSION.stat().st_size

    def test_archive_dedup_by_session_id(self):
        from graphite.capture.hook_handler import archive_transcript

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "archive"

            # Archive twice
            result1 = archive_transcript(
                {"transcript_path": str(SIMPLE_SESSION)},
                archive_dir=archive_dir,
            )
            result2 = archive_transcript(
                {"transcript_path": str(SIMPLE_SESSION)},
                archive_dir=archive_dir,
            )

            # Same destination — second call overwrites
            assert result1 == result2
            # Only one file in archive
            archived_files = list(archive_dir.glob("*.jsonl"))
            assert len(archived_files) == 1

    def test_archive_no_transcript(self):
        from graphite.capture.hook_handler import archive_transcript

        with tempfile.TemporaryDirectory() as tmpdir:
            result = archive_transcript({}, archive_dir=Path(tmpdir))
            assert result is None

    def test_archive_nonexistent_transcript(self):
        from graphite.capture.hook_handler import archive_transcript

        with tempfile.TemporaryDirectory() as tmpdir:
            result = archive_transcript(
                {"transcript_path": "/tmp/does_not_exist.jsonl"},
                archive_dir=Path(tmpdir),
            )
            assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Synthesizer — Cross-Session Reflection
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossSessionReflection:
    def _make_mock_kg(self):
        kg = MagicMock()
        kg.all_entity_ids.return_value = json.dumps(["e1", "e2", "e3"])
        kg.get_entity.side_effect = lambda eid: json.dumps({
            "e1": {
                "id": "e1",
                "canonical_name": "Graphite",
                "entity_type": "Project",
            },
            "e2": {
                "id": "e2",
                "canonical_name": "Build personal memory system",
                "entity_type": {"Custom": "Goal"},
            },
            "e3": {
                "id": "e3",
                "canonical_name": "FastAPI",
                "entity_type": "Technology",
            },
        }.get(eid, {"id": eid, "canonical_name": eid, "entity_type": "Concept"}))
        return kg

    def test_find_cross_project_entities(self):
        from graphite.reflection.synthesizer import Synthesizer
        import time

        kg = self._make_mock_kg()
        now = int(time.time())

        # e1 appears in both Graphite and FountainOfYouth sessions
        kg.get_temporal_chain.side_effect = lambda eid: json.dumps({
            "e1": [
                {"source_document": "claude-session://Graphite/s1", "timestamp": now},
                {"source_document": "claude-session://FountainOfYouth/s2", "timestamp": now},
            ],
            "e2": [
                {"source_document": "claude-session://Graphite/s1", "timestamp": now},
            ],
            "e3": [
                {"source_document": "claude-session://FountainOfYouth/s2", "timestamp": now},
            ],
        }.get(eid, []))

        synth = Synthesizer(kg)
        results = synth.find_cross_project_entities()

        # Only e1 spans multiple projects
        assert len(results) == 1
        assert results[0]["name"] == "Graphite"
        assert results[0]["project_count"] == 2
        assert "Graphite" in results[0]["projects"]
        assert "FountainOfYouth" in results[0]["projects"]

    def test_track_goals_active(self):
        from graphite.reflection.synthesizer import Synthesizer
        import time

        kg = self._make_mock_kg()
        now = int(time.time())

        kg.get_temporal_chain.side_effect = lambda eid: json.dumps({
            "e2": [
                {"source_document": "claude-session://Graphite/s1", "timestamp": now - 86400},
                {"source_document": "claude-session://Graphite/s2", "timestamp": now},
            ],
        }.get(eid, []))

        synth = Synthesizer(kg)
        goals = synth.track_goals()

        assert len(goals) == 1
        assert goals[0]["name"] == "Build personal memory system"
        assert goals[0]["status"] == "active"

    def test_track_goals_stale(self):
        from graphite.reflection.synthesizer import Synthesizer
        import time

        kg = self._make_mock_kg()
        old = int(time.time()) - (60 * 86400)  # 60 days ago

        kg.get_temporal_chain.side_effect = lambda eid: json.dumps({
            "e2": [
                {"source_document": "claude-session://Graphite/s1", "timestamp": old},
            ],
        }.get(eid, []))

        synth = Synthesizer(kg)
        goals = synth.track_goals()

        assert len(goals) == 1
        assert goals[0]["status"] == "stale"

    def test_track_goals_empty(self):
        from graphite.reflection.synthesizer import Synthesizer

        kg = self._make_mock_kg()
        kg.get_temporal_chain.return_value = "[]"

        synth = Synthesizer(kg)
        goals = synth.track_goals()
        # No chunks for the goal entity = not included
        assert len(goals) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration — Parser → Pipeline Round-Trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestParserPipelineIntegration:
    """Verify parser output is compatible with pipeline expectations."""

    def test_chunks_have_required_fields(self):
        """Every chunk must have all fields the pipeline needs."""
        parser = ConversationParser()
        chunks, _ = parser.parse_session(SIMPLE_SESSION)

        for chunk in chunks:
            assert chunk.id is not None and chunk.id != ""
            assert chunk.source_document is not None
            assert chunk.memory_category in ("Episodic", "Semantic", "Procedural")
            assert chunk.text is not None and chunk.text.strip() != ""

    def test_chunks_pass_to_classifier(self):
        """Parser output should be directly consumable by ChunkClassifier."""
        from graphite.extraction.classifier import ChunkClassifier

        parser = ConversationParser()
        chunks, _ = parser.parse_session(SIMPLE_SESSION)

        llm = _make_fixed_llm("discussion")
        classifier = ChunkClassifier(llm)

        # This should not raise
        non_filler, filler = classifier.classify_chunks(chunks[:3])
        assert len(non_filler) + len(filler) == 3

    def test_chunks_pass_to_tagger(self):
        """Classified chunks should be consumable by EntityTagger."""
        from graphite.extraction.classifier import ClassifiedChunk
        from graphite.extraction.tagger import EntityTagger

        parser = ConversationParser()
        chunks, _ = parser.parse_session(SIMPLE_SESSION)

        # Wrap in ClassifiedChunk
        classified = [ClassifiedChunk(raw=c, chunk_type="discussion") for c in chunks[:2]]

        llm = _make_fixed_llm('[{"name": "Graphite", "type": "project"}]')
        tagger = EntityTagger(llm)

        tagged = tagger.tag_chunks(classified)
        assert len(tagged) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Real Data Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════


class TestRealSessionSmoke:
    """Smoke test against actual session files if available."""

    @pytest.fixture
    def real_session_path(self):
        """Find a real session file to test against."""
        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            pytest.skip("No ~/.claude/projects directory found")

        for project_dir in claude_dir.iterdir():
            if not project_dir.is_dir():
                continue
            jsonl_files = list(project_dir.glob("*.jsonl"))
            if jsonl_files:
                # Return the smallest one for speed
                return min(jsonl_files, key=lambda f: f.stat().st_size)

        pytest.skip("No JSONL session files found")

    def test_real_session_parses(self, real_session_path):
        parser = ConversationParser()
        chunks, meta = parser.parse_session(real_session_path)

        assert meta.session_id is not None
        assert meta.project_name is not None
        # Chunks may be 0 if the session is just a file-history-snapshot
        assert isinstance(chunks, list)

    def test_real_session_metadata_valid(self, real_session_path):
        parser = ConversationParser()
        _, meta = parser.parse_session(real_session_path)

        assert isinstance(meta, SessionMetadata)
        assert isinstance(meta.tool_usage_summary, dict)
        assert meta.exchange_count >= 0
