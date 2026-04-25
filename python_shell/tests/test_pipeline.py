"""Tests for the Phase 2 three-pass tag-and-index extraction pipeline.

Covers:
- GraphiteConfig defaults and overrides
- Document categorization
- Structural parser (all document types)
- Chunk classifier (with mock LLM)
- Entity tagger (JSON parsing, hallucination, disambiguation, circuit breaker)
- Pipeline orchestrator (end-to-end with mocks)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graphite.config import GraphiteConfig
from graphite.extraction.classifier import ChunkClassifier, ClassifiedChunk
from graphite.extraction.structural_parser import RawChunk, StructuralParser
from graphite.extraction.tagger import (
    EntityTagger,
    ExtractedEntity,
    TaggedChunk,
)
from graphite.ingestion.categorizer import categorize_document
from graphite.ingestion.pipeline import IngestionPipeline, IngestionResult
from graphite.llm import StubClient


# ═══════════════════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphiteConfig:
    def test_defaults(self):
        config = GraphiteConfig()
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama3.3:70b"
        assert config.llm_temperature == 0.1
        assert config.llm_max_tokens == 4096
        assert config.max_chunk_tokens == 800
        assert config.chunk_overlap_tokens == 100
        assert config.disambiguation_auto_merge_threshold == 0.85
        assert config.disambiguation_review_threshold == 0.70
        assert config.circuit_breaker_failure_rate == 0.50
        assert "decision" in config.valid_chunk_types
        assert "filler" in config.valid_chunk_types
        assert "person" in config.valid_entity_types
        assert config.default_chunk_type == "background"

    def test_overrides(self):
        config = GraphiteConfig(
            llm_model="llama3.1:8b",
            max_chunk_tokens=500,
            disambiguation_auto_merge_threshold=0.90,
        )
        assert config.llm_model == "llama3.1:8b"
        assert config.max_chunk_tokens == 500
        assert config.disambiguation_auto_merge_threshold == 0.90
        # Other defaults still hold
        assert config.llm_provider == "ollama"

    def test_prompts_dir_resolves(self):
        config = GraphiteConfig()
        assert config.prompts_dir.exists()
        assert (config.prompts_dir / "classify.txt").exists()
        assert (config.prompts_dir / "tag.txt").exists()

    def test_get_prompt(self):
        config = GraphiteConfig()
        classify_prompt = config.get_prompt("classify")
        assert "{chunk_text}" in classify_prompt
        tag_prompt = config.get_prompt("tag")
        assert "{chunk_text}" in tag_prompt

    def test_get_prompt_missing(self):
        config = GraphiteConfig()
        with pytest.raises(FileNotFoundError):
            config.get_prompt("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════════
# Categorizer Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCategorizer:
    def test_meetings_episodic(self, tmp_path):
        memory_root = tmp_path / "memory"
        meetings = memory_root / "meetings"
        meetings.mkdir(parents=True)
        doc = meetings / "standup.md"
        doc.touch()
        assert categorize_document(doc, memory_root) == "Episodic"

    def test_associates_semantic(self, tmp_path):
        memory_root = tmp_path / "memory"
        associates = memory_root / "associates"
        associates.mkdir(parents=True)
        doc = associates / "john.md"
        doc.touch()
        assert categorize_document(doc, memory_root) == "Semantic"

    def test_work_procedural(self, tmp_path):
        memory_root = tmp_path / "memory"
        work = memory_root / "work"
        work.mkdir(parents=True)
        doc = work / "project.md"
        doc.touch()
        assert categorize_document(doc, memory_root) == "Procedural"

    def test_outside_memory_root(self, tmp_path):
        memory_root = tmp_path / "memory"
        memory_root.mkdir()
        doc = tmp_path / "random.md"
        doc.touch()
        assert categorize_document(doc, memory_root) == "Episodic"

    def test_unknown_subdirectory(self, tmp_path):
        memory_root = tmp_path / "memory"
        other = memory_root / "other"
        other.mkdir(parents=True)
        doc = other / "file.md"
        doc.touch()
        assert categorize_document(doc, memory_root) == "Episodic"

    def test_nested_subdirectory(self, tmp_path):
        memory_root = tmp_path / "memory"
        nested = memory_root / "meetings" / "2024" / "q3"
        nested.mkdir(parents=True)
        doc = nested / "review.md"
        doc.touch()
        assert categorize_document(doc, memory_root) == "Episodic"


# ═══════════════════════════════════════════════════════════════════════════════
# Structural Parser Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructuralParser:
    def setup_method(self):
        self.parser = StructuralParser()

    def test_empty_document(self):
        chunks = self.parser.parse("", "test.md", "Episodic")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = self.parser.parse("   \n\n  ", "test.md", "Episodic")
        assert chunks == []

    def test_detect_meeting_transcript_bold(self):
        text = "**Alice:** Hello\n**Bob:** Hi"
        assert self.parser._detect_document_type(text) == "meeting_transcript"

    def test_detect_meeting_transcript_speaker_label(self):
        text = "Speaker 1: Hello\nSpeaker 2: Hi"
        assert self.parser._detect_document_type(text) == "meeting_transcript"

    def test_detect_markdown(self):
        text = "# Title\n\n## Section One\n\nContent here."
        assert self.parser._detect_document_type(text) == "markdown"

    def test_detect_date_structured(self):
        text = "2024-01-15\nSome content\n\n2024-01-16\nMore content"
        assert self.parser._detect_document_type(text) == "date_structured"

    def test_detect_plain_text(self):
        text = "Just some regular text with no special markers."
        assert self.parser._detect_document_type(text) == "plain_text"

    def test_parse_meeting_transcript(self):
        text = (
            "# Meeting — 2024-09-15\n\n"
            "**Alice:** I think we should use React.\n\n"
            "**Bob:** I agree. The component library looks solid.\n\n"
            "**Alice:** Great, let's move forward."
        )
        chunks = self.parser.parse(text, "meeting.md", "Episodic")
        assert len(chunks) >= 2
        speakers = {c.speaker for c in chunks if c.speaker}
        assert "Alice" in speakers
        assert "Bob" in speakers
        assert all(c.memory_category == "Episodic" for c in chunks)
        assert all(c.source_document == "meeting.md" for c in chunks)

    def test_parse_markdown_sections(self):
        text = (
            "## Introduction\n\n"
            "This is the intro paragraph.\n\n"
            "## Technical Details\n\n"
            "Here are the technical details.\n\n"
            "More details in this paragraph."
        )
        chunks = self.parser.parse(text, "doc.md", "Procedural")
        assert len(chunks) >= 2
        sections = {c.section_name for c in chunks if c.section_name}
        assert "Introduction" in sections
        assert "Technical Details" in sections

    def test_parse_date_structured(self):
        text = (
            "2024-01-15\n"
            "Had a productive meeting about the project roadmap.\n\n"
            "2024-01-16\n"
            "Reviewed the design specs with the team."
        )
        chunks = self.parser.parse(text, "log.md", "Episodic")
        assert len(chunks) >= 2
        # Timestamps should be extracted
        timestamps = {c.timestamp for c in chunks if c.timestamp}
        assert len(timestamps) >= 1

    def test_parse_plain_text(self):
        text = (
            "First paragraph about React development.\n\n"
            "Second paragraph about testing strategies.\n\n"
            "Third paragraph about deployment."
        )
        chunks = self.parser.parse(text, "notes.md", "Procedural")
        assert len(chunks) == 3

    def test_oversized_chunk_splitting(self):
        # Create a chunk that exceeds the token limit
        # Use 200 words with max_chunk_tokens=100 (100/1.3 ≈ 77 words per chunk)
        long_text = " ".join([f"word{i}" for i in range(200)])
        parser = StructuralParser(GraphiteConfig(max_chunk_tokens=100))
        chunks = parser.parse(long_text, "big.md", "Episodic")
        assert len(chunks) > 1
        # Each chunk should be under the limit (with some slack for overlap)
        for chunk in chunks:
            assert parser._estimate_tokens(chunk.text) <= 200  # allow some slack

    def test_timestamp_extraction_iso(self):
        text = "Meeting on 2024-09-15 about the project."
        ts = self.parser._extract_timestamp(text)
        assert ts is not None

    def test_timestamp_extraction_written(self):
        text = "Meeting on November 18, 2024 about the project."
        ts = self.parser._extract_timestamp(text)
        assert ts is not None

    def test_timestamp_extraction_none(self):
        text = "Just some text without any dates."
        ts = self.parser._extract_timestamp(text)
        assert ts is None

    def test_chunk_ids_unique(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = self.parser.parse(text, "test.md", "Episodic")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_meeting_preamble_preserved(self):
        text = (
            "# Q3 Planning — 2024-09-15\n\n"
            "## Attendees\nAlice, Bob, Carol\n\n"
            "**Alice:** Let's begin."
        )
        chunks = self.parser.parse(text, "meeting.md", "Episodic")
        # Should have preamble + at least one speaker turn
        assert len(chunks) >= 1

    def test_test_corpus_meeting(self):
        """Parse a real test corpus file to verify it produces reasonable chunks."""
        corpus_file = Path(__file__).parent.parent.parent / "test_corpus" / "meetings" / "q3-design-review.md"
        if not corpus_file.exists():
            pytest.skip("Test corpus not found")
        text = corpus_file.read_text()
        chunks = self.parser.parse(text, str(corpus_file), "Episodic")
        assert len(chunks) >= 4  # At least a few speaker turns
        assert all(c.text.strip() for c in chunks)  # No empty chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier Tests
# ═══════════════════════════════════════════════════════════════════════════════


class MockClassifierLLM:
    """Mock LLM that returns predefined classifications.

    Handles both batch and individual calls:
    - Batch call (prompt contains '---CHUNK'): returns all remaining
      responses as newline-separated lines.
    - Individual call: returns responses one at a time.
    """

    def __init__(self, responses):
        self.responses = responses
        self.call_index = 0

    def chat(self, messages):
        prompt = messages[0]["content"] if messages else ""

        # Detect batch call
        if "---CHUNK" in prompt:
            remaining = self.responses[self.call_index:]
            self.call_index = len(self.responses)
            return "\n".join(remaining)

        # Individual call
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
        else:
            response = "background"
        self.call_index += 1
        return response


class TestClassifier:
    def _make_chunk(self, text="Some meeting discussion.", chunk_id="test-1"):
        return RawChunk(
            id=chunk_id,
            source_document="test.md",
            section_name=None,
            speaker=None,
            timestamp=None,
            memory_category="Episodic",
            text=text,
        )

    def test_valid_classifications(self):
        llm = MockClassifierLLM(["decision", "discussion", "action_item"])
        classifier = ChunkClassifier(llm)
        chunks = [self._make_chunk(f"text {i}") for i in range(3)]
        non_filler, filler = classifier.classify_chunks(chunks)
        assert len(non_filler) == 3
        assert len(filler) == 0
        assert non_filler[0].chunk_type == "decision"
        assert non_filler[1].chunk_type == "discussion"
        assert non_filler[2].chunk_type == "action_item"

    def test_filler_detection(self):
        llm = MockClassifierLLM(["discussion", "filler", "background"])
        classifier = ChunkClassifier(llm)
        chunks = [self._make_chunk(f"text {i}") for i in range(3)]
        non_filler, filler = classifier.classify_chunks(chunks)
        assert len(non_filler) == 2
        assert len(filler) == 1

    def test_invalid_classification_defaults(self):
        llm = MockClassifierLLM(["totally_invalid_type"])
        classifier = ChunkClassifier(llm)
        chunks = [self._make_chunk()]
        non_filler, filler = classifier.classify_chunks(chunks)
        assert len(non_filler) == 1
        assert non_filler[0].chunk_type == "background"

    def test_messy_llm_output(self):
        llm = MockClassifierLLM(["  Decision. ", "DISCUSSION\n", "  action_item  "])
        classifier = ChunkClassifier(llm)
        chunks = [self._make_chunk(f"text {i}") for i in range(3)]
        non_filler, _ = classifier.classify_chunks(chunks)
        assert non_filler[0].chunk_type == "decision"
        assert non_filler[1].chunk_type == "discussion"
        assert non_filler[2].chunk_type == "action_item"

    def test_llm_error_defaults(self):
        class ErrorLLM:
            def chat(self, messages):
                raise RuntimeError("Connection failed")

        classifier = ChunkClassifier(ErrorLLM())
        chunks = [self._make_chunk()]
        non_filler, filler = classifier.classify_chunks(chunks)
        assert len(non_filler) == 1
        assert non_filler[0].chunk_type == "background"


# ═══════════════════════════════════════════════════════════════════════════════
# Tagger Tests
# ═══════════════════════════════════════════════════════════════════════════════


class MockTaggerLLM:
    """Mock LLM that returns predefined JSON entity lists."""

    def __init__(self, responses):
        self.responses = responses
        self.call_index = 0

    def chat(self, messages):
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
        else:
            response = "[]"
        self.call_index += 1
        return response


class TestTagger:
    def _make_classified_chunk(self, text="Sarah Chen discussed the React project.", chunk_type="discussion"):
        raw = RawChunk(
            id="test-chunk-1",
            source_document="test.md",
            section_name=None,
            speaker=None,
            timestamp=None,
            memory_category="Episodic",
            text=text,
        )
        return ClassifiedChunk(raw=raw, chunk_type=chunk_type)

    def test_clean_json_parsing(self):
        response = '[{"name": "Sarah Chen", "type": "person"}, {"name": "React", "type": "technology"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 2
        assert results[0].entities[0].name == "Sarah Chen"
        assert results[0].entities[1].name == "React"

    def test_json_with_markdown_fences(self):
        response = '```json\n[{"name": "Sarah Chen", "type": "person"}]\n```'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_json_with_preamble(self):
        response = 'Here are the entities I found:\n[{"name": "Sarah Chen", "type": "person"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_json_with_trailing_comma(self):
        response = '[{"name": "Sarah Chen", "type": "person"},]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_hallucination_detection(self):
        """Entity names that don't appear in the chunk text should be removed."""
        response = '[{"name": "Sarah Chen", "type": "person"}, {"name": "Nonexistent Person", "type": "person"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        names = [e.name for e in results[0].entities]
        assert "Sarah Chen" in names
        assert "Nonexistent Person" not in names

    def test_invalid_entity_type(self):
        """Entities with invalid types should be removed."""
        response = '[{"name": "Sarah Chen", "type": "person"}, {"name": "React", "type": "framework"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        names = [e.name for e in results[0].entities]
        assert "Sarah Chen" in names
        assert "React" not in names  # "framework" is not a valid type

    def test_empty_json_response(self):
        response = "[]"
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 0

    def test_completely_broken_json(self):
        """Malformed JSON that can't be repaired should fail gracefully."""
        response = "This is not JSON at all, just random text."
        # Two responses: original + retry
        llm = MockTaggerLLM([response, response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        # Should get 0 results (chunk failed)
        assert len(results) == 0

    def test_circuit_breaker(self):
        """If >50% of chunks fail, should abort early."""
        broken_response = "not json"
        # 4 chunks, all with broken JSON (x2 for retries)
        llm = MockTaggerLLM([broken_response] * 20)
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk(f"text {i}") for i in range(4)]
        results = tagger.tag_chunks(chunks)
        # Circuit breaker should trip — not all chunks processed
        assert len(results) < 4

    def test_case_insensitive_validation(self):
        """Entity validation should be case-insensitive."""
        response = '[{"name": "sarah chen", "type": "person"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunk = self._make_classified_chunk("Sarah Chen discussed the project.")
        results = tagger.tag_chunks([chunk])
        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_xml_wrapper_tags(self):
        """LLM wrapping JSON in XML-like tags."""
        response = '<answer>[{"name": "Sarah Chen", "type": "person"}]</answer>'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 1
        assert results[0].entities[0].name == "Sarah Chen"

    def test_wrapper_object(self):
        """LLM returning {"entities": [...]} instead of bare array."""
        response = '{"entities": [{"name": "Sarah Chen", "type": "person"}, {"name": "React", "type": "technology"}]}'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 2

    def test_smart_quotes(self):
        """LLM using curly/smart quotes instead of straight quotes."""
        response = '[\u201c{\u201cname\u201d: \u201cSarah Chen\u201d, \u201ctype\u201d: \u201cperson\u201d}]'
        # Simpler version — just the values in smart quotes
        response = '[{"name": \u201cSarah Chen\u201d, "type": \u201cperson\u201d}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert results[0].entities[0].name == "Sarah Chen"

    def test_js_style_comments(self):
        """LLM adding // comments in JSON output."""
        response = (
            '[\n'
            '  {"name": "Sarah Chen", "type": "person"}, // main speaker\n'
            '  {"name": "React", "type": "technology"} // frontend framework\n'
            ']'
        )
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 2

    def test_numbered_list_format(self):
        """LLM returning numbered list instead of JSON array."""
        response = (
            '1. {"name": "Sarah Chen", "type": "person"}\n'
            '2. {"name": "React", "type": "technology"}'
        )
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 2

    def test_inconsistent_key_names(self):
        """LLM using 'entity'/'category' instead of 'name'/'type'."""
        response = '[{"entity": "Sarah Chen", "category": "person"}, {"entity_name": "React", "label": "technology"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 2
        assert results[0].entities[0].name == "Sarah Chen"
        assert results[0].entities[1].name == "React"

    def test_single_quotes_json(self):
        """LLM returning JSON with single quotes throughout."""
        response = "[{'name': 'Sarah Chen', 'type': 'person'}]"
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_unquoted_keys(self):
        """LLM returning JSON with unquoted keys."""
        response = '[{name: "Sarah Chen", type: "person"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_response_tag_wrapper(self):
        """LLM wrapping in <response> tags."""
        response = '<response>\n[{"name": "Sarah Chen", "type": "person"}]\n</response>'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert len(results) == 1
        assert len(results[0].entities) == 1

    def test_disambiguation_no_graph(self):
        """Without a knowledge graph, all entities should be new."""
        response = '[{"name": "Sarah Chen", "type": "person"}]'
        llm = MockTaggerLLM([response])
        tagger = EntityTagger(llm, knowledge_graph=None)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert results[0].entities[0].is_new is True

    def test_disambiguation_exact_match(self):
        """Exact name match should link to existing entity."""
        response = '[{"name": "Sarah Chen", "type": "person"}]'
        llm = MockTaggerLLM([response])

        # Mock knowledge graph
        mock_kg = MagicMock()
        mock_kg.search_entities.return_value = json.dumps([
            {"id": "existing-123", "canonical_name": "Sarah Chen", "aliases": []}
        ])

        tagger = EntityTagger(llm, knowledge_graph=mock_kg)
        chunks = [self._make_classified_chunk()]
        results = tagger.tag_chunks(chunks)
        assert results[0].entities[0].is_new is False
        assert results[0].entities[0].existing_entity_id == "existing-123"


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Orchestrator Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipeline:
    def _make_mock_llm(self):
        """Create a mock LLM that returns sensible responses for both classify and tag."""
        call_count = {"n": 0}

        class PipelineMockLLM:
            def chat(self, messages):
                call_count["n"] += 1
                content = messages[0]["content"]
                if "Classify" in content:
                    return "discussion"
                elif "List every" in content:
                    # Tag response — extract entity names from the text
                    return '[{"name": "Sarah Chen", "type": "person"}, {"name": "React", "type": "technology"}]'
                return "background"

        return PipelineMockLLM()

    def test_ingest_file(self, tmp_path):
        """End-to-end pipeline test with a simple document."""
        from graphite.semantic_engine import PyKnowledgeGraph

        # Create test document
        doc = tmp_path / "memory" / "meetings" / "test.md"
        doc.parent.mkdir(parents=True)
        doc.write_text(
            "**Sarah Chen:** I think we should use React for the Dashboard Redesign.\n\n"
            "**Marcus Johnson:** I agree, React with TypeScript is the right choice."
        )

        kg = PyKnowledgeGraph(str(tmp_path))
        config = GraphiteConfig(memory_root=tmp_path / "memory")
        llm = self._make_mock_llm()

        pipeline = IngestionPipeline(
            knowledge_graph=kg,
            llm_client=llm,
            config=config,
        )

        result = pipeline.ingest_file(doc)
        assert result.status in ("complete", "partial")
        assert result.chunks_total >= 2
        assert result.chunks_tagged >= 1
        assert result.entities_created >= 1

    def test_ingest_empty_file(self, tmp_path):
        """Empty files should be handled gracefully."""
        from graphite.semantic_engine import PyKnowledgeGraph

        doc = tmp_path / "empty.md"
        doc.write_text("")

        kg = PyKnowledgeGraph(str(tmp_path))
        llm = self._make_mock_llm()
        pipeline = IngestionPipeline(kg, llm)

        result = pipeline.ingest_file(doc)
        assert result.status == "complete"
        assert result.chunks_total == 0

    def test_ingest_nonexistent_file(self, tmp_path):
        """Missing files should return failed status."""
        from graphite.semantic_engine import PyKnowledgeGraph

        kg = PyKnowledgeGraph(str(tmp_path))
        llm = self._make_mock_llm()
        pipeline = IngestionPipeline(kg, llm)

        result = pipeline.ingest_file(tmp_path / "nonexistent.md")
        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_ingest_directory(self, tmp_path):
        """Directory ingestion should process all .md files."""
        from graphite.semantic_engine import PyKnowledgeGraph

        # Create multiple test documents
        meetings = tmp_path / "memory" / "meetings"
        meetings.mkdir(parents=True)
        (meetings / "doc1.md").write_text(
            "**Sarah Chen:** React is great for building dashboards."
        )
        (meetings / "doc2.md").write_text(
            "**Marcus Johnson:** Python works well for backend services."
        )

        kg = PyKnowledgeGraph(str(tmp_path))
        config = GraphiteConfig(memory_root=tmp_path / "memory")

        # Create a mock LLM that handles both classify and tag calls for both docs
        class MultiDocMockLLM:
            def chat(self, messages):
                content = messages[0]["content"]
                if "Classify" in content:
                    return "discussion"
                elif "List every" in content:
                    if "Sarah Chen" in content:
                        return '[{"name": "Sarah Chen", "type": "person"}, {"name": "React", "type": "technology"}]'
                    elif "Marcus Johnson" in content:
                        return '[{"name": "Marcus Johnson", "type": "person"}, {"name": "Python", "type": "technology"}]'
                    return "[]"
                return "background"

        pipeline = IngestionPipeline(kg, MultiDocMockLLM(), config=config)
        results = pipeline.ingest_directory(meetings)
        assert len(results) == 2

    def test_graph_writes(self, tmp_path):
        """Verify entities, chunks, and edges are written to the graph."""
        from graphite.semantic_engine import PyKnowledgeGraph

        # Use markdown format (not meeting transcript) so all entity names
        # appear in the chunk text for hallucination validation to pass
        doc = tmp_path / "test.md"
        doc.write_text(
            "Sarah Chen discussed React and TypeScript with Marcus Johnson "
            "during the weekly planning session."
        )

        kg = PyKnowledgeGraph(str(tmp_path))

        class DetailedMockLLM:
            def chat(self, messages):
                content = messages[0]["content"]
                if "Classify" in content:
                    return "discussion"
                elif "List every" in content:
                    return json.dumps([
                        {"name": "Sarah Chen", "type": "person"},
                        {"name": "React", "type": "technology"},
                        {"name": "TypeScript", "type": "technology"},
                        {"name": "Marcus Johnson", "type": "person"},
                    ])
                return "background"

        pipeline = IngestionPipeline(kg, DetailedMockLLM())
        result = pipeline.ingest_file(doc)

        assert result.entities_created == 4
        # 4 entities → C(4,2) = 6 co-occurrence edges
        assert result.edges_created == 6

        # Verify entities exist in graph
        stats_json = kg.get_statistics()
        stats = json.loads(stats_json)
        assert stats["entity_count"] == 4

    def test_no_llm_client(self, tmp_path):
        """Pipeline without LLM should fail gracefully."""
        from graphite.semantic_engine import PyKnowledgeGraph

        doc = tmp_path / "test.md"
        doc.write_text("Some content here.")

        kg = PyKnowledgeGraph(str(tmp_path))
        pipeline = IngestionPipeline(kg, llm_client=None)
        result = pipeline.ingest_file(doc)
        assert result.status == "failed"
        assert any("LLM" in e for e in result.errors)

    def test_save_graph(self, tmp_path):
        """Verify graph persistence after ingestion."""
        from graphite.semantic_engine import PyKnowledgeGraph

        # Use plain text (not meeting transcript) so entity names stay in chunk text
        doc = tmp_path / "test.md"
        doc.write_text(
            "Sarah Chen says React is great for building dashboards."
        )

        kg = PyKnowledgeGraph(str(tmp_path))

        class SimpleMockLLM:
            def chat(self, messages):
                content = messages[0]["content"]
                if "Classify" in content:
                    return "discussion"
                elif "List every" in content:
                    return '[{"name": "Sarah Chen", "type": "person"}, {"name": "React", "type": "technology"}]'
                return "background"

        pipeline = IngestionPipeline(kg, SimpleMockLLM())
        pipeline.ingest_file(doc)

        # save_path is the directory; GraphStore appends .graphite/graph.msgpack
        save_path = str(tmp_path / "save_dir")
        pipeline.save_graph(save_path)

        # Load and verify
        loaded_kg = PyKnowledgeGraph.from_path(save_path)
        stats_json = loaded_kg.get_statistics()
        stats = json.loads(stats_json)
        assert stats["entity_count"] == 2
