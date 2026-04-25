"""Tests for Phase 7: Evaluation framework.

Covers:
- TestQueryLoader: load, validate, filter by type
- SimpleRAGBaseline: build_index, retrieve, empty corpus
- Metrics: each metric function with synthetic/mock data
- ReportFormatter: JSON roundtrip, console output
- DegradedLLMClient: failure injection
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graphite.config import GraphiteConfig
from graphite.evaluation.queries import TestQuery, TestQueryLoader
from graphite.evaluation.baseline_rag import SimpleRAGBaseline
from graphite.evaluation.metrics import (
    MetricResult,
    DegradedLLMClient,
    evaluate_cooccurrence_accuracy,
    evaluate_context_efficiency,
    evaluate_entity_tagging_accuracy,
    evaluate_multihop_reasoning,
    evaluate_retrieval_precision_at_k,
    evaluate_temporal_reasoning,
    _chunk_contains_keywords,
    _count_keyword_hits,
)
from graphite.evaluation.runner import EvalReport
from graphite.evaluation.report import ReportFormatter

# Import shared test infrastructure
from tests.test_memory_context import (
    FakeKnowledgeGraph,
    _build_test_graph,
    _make_mock_embedding_manager,
    _deterministic_embedding,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


def _make_sample_queries():
    """Create a small set of test queries."""
    return [
        TestQuery(
            id="retrieval_01",
            type="retrieval",
            question="What technologies does John Doe work with?",
            expected_entities=["John Doe", "React", "Dashboard Redesign"],
            expected_answer_keywords=["React", "Dashboard"],
        ),
        TestQuery(
            id="retrieval_02",
            type="retrieval",
            question="Who is working on the Dashboard Redesign?",
            expected_entities=["John Doe", "Jane Smith", "Dashboard Redesign"],
            expected_answer_keywords=["John", "Jane", "Dashboard"],
        ),
        TestQuery(
            id="temporal_01",
            type="temporal",
            question="What happened in October 2024?",
            expected_entities=["John Doe", "Dashboard Redesign"],
            expected_answer_keywords=["Dashboard", "John"],
            time_context={"start": 1727740800, "end": 1730419200},
        ),
        TestQuery(
            id="multihop_01",
            type="multi_hop",
            question="What technologies is John exposed to through projects?",
            expected_entities=["John Doe", "Dashboard Redesign", "React"],
            expected_answer_keywords=["React", "Dashboard"],
            hops_required=2,
        ),
    ]


def _make_corpus_dir_with_expected(tmp_path: Path) -> Path:
    """Create a temporary corpus directory with .expected.json files."""
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir(parents=True)

    # Write a simple document
    doc = meetings_dir / "test-meeting.md"
    doc.write_text(
        "# Test Meeting\n\n"
        "**John Doe:** We should use React for the Dashboard Redesign.\n\n"
        "**Jane Smith:** I agree. React with TypeScript is the way to go.\n"
    )

    # Write expected.json
    expected = {
        "entities": [
            {"name": "John Doe", "type": "person"},
            {"name": "Jane Smith", "type": "person"},
            {"name": "React", "type": "technology"},
            {"name": "Dashboard Redesign", "type": "project"},
        ],
        "cooccurrences": [
            {"entity_a": "John Doe", "entity_b": "React"},
            {"entity_a": "John Doe", "entity_b": "Dashboard Redesign"},
        ],
        "expected_chunks_min": 2,
        "expected_chunks_max": 5,
        "expected_filler_chunks": 0,
    }
    expected_file = meetings_dir / "test-meeting.expected.json"
    expected_file.write_text(json.dumps(expected))

    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# TestQueryLoader
# ═══════════════════════════════════════════════════════════════════════════════


class TestTestQueryLoader:
    """Tests for TestQueryLoader."""

    def test_load_valid_queries(self, tmp_path):
        queries = [
            {
                "id": "retrieval_01",
                "type": "retrieval",
                "question": "What tech does John use?",
                "expected_entities": ["John Doe"],
                "expected_answer_keywords": ["React"],
            },
            {
                "id": "temporal_01",
                "type": "temporal",
                "question": "What happened in Oct?",
                "expected_entities": ["Dashboard"],
                "expected_answer_keywords": ["Dashboard"],
                "time_context": {"start": 100, "end": 200},
            },
        ]
        path = tmp_path / "queries.json"
        path.write_text(json.dumps(queries))

        result = TestQueryLoader.load(path)
        assert len(result) == 2
        assert result[0].id == "retrieval_01"
        assert result[0].type == "retrieval"
        assert result[1].time_context == {"start": 100, "end": 200}

    def test_load_missing_field_raises(self, tmp_path):
        queries = [{"id": "bad_01", "type": "retrieval"}]  # missing required fields
        path = tmp_path / "queries.json"
        path.write_text(json.dumps(queries))

        with pytest.raises(ValueError, match="missing required field"):
            TestQueryLoader.load(path)

    def test_load_invalid_type_raises(self, tmp_path):
        queries = [{
            "id": "bad_02",
            "type": "invalid_type",
            "question": "?",
            "expected_entities": ["X"],
            "expected_answer_keywords": ["Y"],
        }]
        path = tmp_path / "queries.json"
        path.write_text(json.dumps(queries))

        with pytest.raises(ValueError, match="invalid type"):
            TestQueryLoader.load(path)

    def test_load_empty_entities_raises(self, tmp_path):
        queries = [{
            "id": "bad_03",
            "type": "retrieval",
            "question": "?",
            "expected_entities": [],
            "expected_answer_keywords": ["Y"],
        }]
        path = tmp_path / "queries.json"
        path.write_text(json.dumps(queries))

        with pytest.raises(ValueError, match="empty expected_entities"):
            TestQueryLoader.load(path)

    def test_filter_by_type(self):
        queries = _make_sample_queries()
        retrieval = TestQueryLoader.filter_by_type(queries, "retrieval")
        assert len(retrieval) == 2
        assert all(q.type == "retrieval" for q in retrieval)

        temporal = TestQueryLoader.filter_by_type(queries, "temporal")
        assert len(temporal) == 1

        multihop = TestQueryLoader.filter_by_type(queries, "multi_hop")
        assert len(multihop) == 1

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TestQueryLoader.load(tmp_path / "nonexistent.json")


# ═══════════════════════════════════════════════════════════════════════════════
# SimpleRAGBaseline
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaselineRAG:
    """Tests for SimpleRAGBaseline."""

    def test_empty_corpus(self, tmp_path):
        baseline = SimpleRAGBaseline(tmp_path)
        baseline.build_index()
        assert len(baseline.chunks) == 0
        results = baseline.retrieve("test query")
        assert results == []

    def test_build_index_parses_files(self, tmp_path):
        meetings = tmp_path / "meetings"
        meetings.mkdir()
        doc = meetings / "test.md"
        doc.write_text(
            "# Test Meeting\n\n"
            "**Alice:** We need to use Python.\n\n"
            "**Bob:** I agree, Python is great.\n"
        )

        # Mock the embedding manager to avoid loading the real model
        baseline = SimpleRAGBaseline(tmp_path)
        mock_emb = MagicMock()
        mock_emb.generate_embedding.side_effect = lambda texts: [
            _deterministic_embedding(t) for t in texts
        ]
        baseline._embedding_manager = mock_emb

        baseline.build_index()
        assert len(baseline.chunks) > 0
        assert len(baseline.chunk_embeddings) == len(baseline.chunks)

    def test_retrieve_returns_scored_results(self, tmp_path):
        meetings = tmp_path / "meetings"
        meetings.mkdir()
        doc = meetings / "test.md"
        doc.write_text(
            "# Meeting\n\n"
            "**Alice:** Python is great for backend.\n\n"
            "**Bob:** JavaScript is better for frontend.\n"
        )

        baseline = SimpleRAGBaseline(tmp_path)
        mock_emb = MagicMock()
        mock_emb.generate_embedding.side_effect = lambda texts: [
            _deterministic_embedding(t) for t in texts
        ]
        baseline._embedding_manager = mock_emb

        baseline.build_index()
        results = baseline.retrieve("Python backend", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        # Results are (chunk, score) tuples
        for chunk, score in results:
            assert hasattr(chunk, "text")
            assert isinstance(score, float)

        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_similarity(self):
        # Identical vectors
        v = np.array([1.0, 0.0, 0.0])
        assert SimpleRAGBaseline._cosine_similarity(v, v) == pytest.approx(1.0)

        # Orthogonal vectors
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert SimpleRAGBaseline._cosine_similarity(v1, v2) == pytest.approx(0.0)

        # Zero vector
        v0 = np.array([0.0, 0.0])
        assert SimpleRAGBaseline._cosine_similarity(v0, v1) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricHelpers:
    """Tests for metric helper functions."""

    def test_chunk_contains_keywords(self):
        text = "John Doe is using React for the Dashboard Redesign project."
        assert _chunk_contains_keywords(text, ["React"])
        assert _chunk_contains_keywords(text, ["Dashboard"])
        assert _chunk_contains_keywords(text, ["react"])  # case-insensitive
        assert not _chunk_contains_keywords(text, ["Python"])
        assert _chunk_contains_keywords(text, ["Python", "React"])  # any match

    def test_count_keyword_hits(self):
        text = "John Doe is using React and TypeScript for the Dashboard."
        assert _count_keyword_hits(text, ["React", "TypeScript", "Python"]) == 2
        assert _count_keyword_hits(text, ["Vue"]) == 0
        assert _count_keyword_hits(text, []) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestEntityTaggingAccuracy:
    """Tests for metric 1: Entity Tagging Accuracy."""

    def test_perfect_extraction(self, tmp_path):
        corpus_dir = _make_corpus_dir_with_expected(tmp_path)

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-john", "John Doe", "Person",
                          source_documents=["meetings/test-meeting.md"])
        kg.add_test_entity("e-jane", "Jane Smith", "Person",
                          source_documents=["meetings/test-meeting.md"])
        kg.add_test_entity("e-react", "React", "Technology",
                          source_documents=["meetings/test-meeting.md"])
        kg.add_test_entity("e-dash", "Dashboard Redesign", "Project",
                          source_documents=["meetings/test-meeting.md"])

        result = evaluate_entity_tagging_accuracy(kg, corpus_dir)
        assert result.name == "Entity Tagging Accuracy"
        assert result.score == pytest.approx(1.0)
        assert result.details["precision"] == pytest.approx(1.0)
        assert result.details["recall"] == pytest.approx(1.0)

    def test_partial_extraction(self, tmp_path):
        corpus_dir = _make_corpus_dir_with_expected(tmp_path)

        kg = FakeKnowledgeGraph()
        # Only extract 2 of 4 expected entities
        kg.add_test_entity("e-john", "John Doe", "Person",
                          source_documents=["meetings/test-meeting.md"])
        kg.add_test_entity("e-react", "React", "Technology",
                          source_documents=["meetings/test-meeting.md"])

        result = evaluate_entity_tagging_accuracy(kg, corpus_dir)
        assert result.details["recall"] == pytest.approx(0.5)
        assert result.details["precision"] == pytest.approx(1.0)

    def test_no_expected_files(self, tmp_path):
        result = evaluate_entity_tagging_accuracy(FakeKnowledgeGraph(), tmp_path)
        assert result.score == 0.0


class TestCooccurrenceAccuracy:
    """Tests for metric 2: Co-occurrence Accuracy."""

    def test_all_cooccurrences_found(self, tmp_path):
        corpus_dir = _make_corpus_dir_with_expected(tmp_path)

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-john", "John Doe")
        kg.add_test_entity("e-react", "React", "Technology")
        kg.add_test_entity("e-dash", "Dashboard Redesign", "Project")

        kg.add_test_edge("e-john", "e-react", "c1")
        kg.add_test_edge("e-john", "e-dash", "c2")

        result = evaluate_cooccurrence_accuracy(kg, corpus_dir)
        assert result.name == "Co-occurrence Accuracy"
        assert result.score == pytest.approx(1.0)

    def test_missing_cooccurrences(self, tmp_path):
        corpus_dir = _make_corpus_dir_with_expected(tmp_path)

        kg = FakeKnowledgeGraph()
        kg.add_test_entity("e-john", "John Doe")
        kg.add_test_entity("e-react", "React", "Technology")
        kg.add_test_entity("e-dash", "Dashboard Redesign", "Project")

        # Only one of two expected co-occurrences
        kg.add_test_edge("e-john", "e-react", "c1")

        result = evaluate_cooccurrence_accuracy(kg, corpus_dir)
        assert result.score == pytest.approx(0.5)


class TestRetrievalPrecision:
    """Tests for metric 3: Retrieval Precision @k."""

    def test_with_mock_context_and_baseline(self):
        queries = _make_sample_queries()

        # Mock context manager that returns text with keywords
        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = (
            "John Doe works with React on the Dashboard Redesign project."
        )

        # Mock baseline that returns chunks with fewer keywords
        mock_baseline = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Some general text about projects."
        mock_baseline.retrieve.return_value = [(mock_chunk, 0.8)]

        result = evaluate_retrieval_precision_at_k(
            mock_ctx, mock_baseline, queries, k=5
        )

        assert result.name == "Retrieval Precision @5"
        assert result.queries_evaluated == 2  # only retrieval queries
        # Graphite should score higher since its context has more keywords
        assert result.score > result.baseline_score

    def test_empty_context(self):
        queries = _make_sample_queries()

        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = ""

        mock_baseline = MagicMock()
        mock_baseline.retrieve.return_value = []

        result = evaluate_retrieval_precision_at_k(
            mock_ctx, mock_baseline, queries, k=5
        )
        assert result.score == 0.0
        assert result.baseline_score == 0.0


class TestTemporalReasoning:
    """Tests for metric 4: Temporal Reasoning."""

    def test_temporal_queries_evaluated(self):
        queries = _make_sample_queries()

        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = "Dashboard John Doe October event"

        mock_baseline = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Some text"
        mock_baseline.retrieve.return_value = [(mock_chunk, 0.7)]

        result = evaluate_temporal_reasoning(mock_ctx, mock_baseline, queries)

        assert result.name == "Temporal Reasoning"
        assert result.queries_evaluated == 1  # only temporal queries

    def test_passes_time_context(self):
        queries = _make_sample_queries()
        temporal_q = TestQueryLoader.filter_by_type(queries, "temporal")[0]

        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = "Dashboard"

        mock_baseline = MagicMock()
        mock_baseline.retrieve.return_value = []

        evaluate_temporal_reasoning(mock_ctx, mock_baseline, queries)

        # Verify time_start and time_end were passed
        call_args = mock_ctx.assemble_context.call_args
        assert call_args.kwargs.get("time_start") == temporal_q.time_context["start"]
        assert call_args.kwargs.get("time_end") == temporal_q.time_context["end"]


class TestMultihopReasoning:
    """Tests for metric 5: Multi-hop Reasoning."""

    def test_multihop_with_graph(self):
        queries = _make_sample_queries()
        kg = _build_test_graph()

        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = (
            "John Doe works on Dashboard Redesign using React"
        )

        mock_baseline = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "John"
        mock_baseline.retrieve.return_value = [(mock_chunk, 0.5)]

        result = evaluate_multihop_reasoning(
            kg, mock_ctx, mock_baseline, queries
        )

        assert result.name == "Multi-hop Reasoning"
        assert result.queries_evaluated == 1
        assert result.score > result.baseline_score


class TestContextEfficiency:
    """Tests for metric 6: Context Efficiency."""

    def test_efficiency_calculation(self):
        queries = _make_sample_queries()

        # Graphite: short text with many keywords = high efficiency
        mock_ctx = MagicMock()
        mock_ctx.assemble_context.return_value = "John React Dashboard"

        # Baseline: long text with few keywords = low efficiency
        mock_baseline = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = ("Lorem ipsum dolor sit amet " * 20 + "John")
        mock_baseline.retrieve.return_value = [(mock_chunk, 0.5)]

        result = evaluate_context_efficiency(
            mock_ctx, mock_baseline, queries
        )

        assert result.name == "Context Efficiency"
        assert result.score > 0.0
        assert result.baseline_score is not None


# ═══════════════════════════════════════════════════════════════════════════════
# DegradedLLMClient
# ═══════════════════════════════════════════════════════════════════════════════


class TestDegradedLLMClient:
    """Tests for the DegradedLLMClient wrapper."""

    def test_passes_through_normally(self):
        base = MagicMock()
        base.chat.return_value = "normal response"

        degraded = DegradedLLMClient(base, failure_rate=0.0, malformed_rate=0.0)
        result = degraded.chat([{"role": "user", "content": "test"}])

        assert result == "normal response"
        assert degraded.call_count == 1
        assert degraded.failures_injected == 0
        assert degraded.malformed_injected == 0

    def test_injects_failures(self):
        base = MagicMock()
        base.chat.return_value = "ok"

        # 100% failure rate
        degraded = DegradedLLMClient(base, failure_rate=1.0, malformed_rate=0.0)

        with pytest.raises(ConnectionError, match="Simulated"):
            degraded.chat([])

        assert degraded.failures_injected == 1

    def test_injects_malformed(self):
        base = MagicMock()
        base.chat.return_value = "ok"

        # 0% failure, 100% malformed
        degraded = DegradedLLMClient(base, failure_rate=0.0, malformed_rate=1.0)
        result = degraded.chat([])

        assert "malformed" in result
        assert degraded.malformed_injected == 1

    def test_deterministic_with_seed(self):
        base = MagicMock()
        base.chat.return_value = "ok"

        results1 = []
        d1 = DegradedLLMClient(base, failure_rate=0.2, malformed_rate=0.2, seed=123)
        for _ in range(20):
            try:
                r = d1.chat([])
                results1.append(("ok" if r == "ok" else "malformed"))
            except ConnectionError:
                results1.append("fail")

        results2 = []
        d2 = DegradedLLMClient(base, failure_rate=0.2, malformed_rate=0.2, seed=123)
        for _ in range(20):
            try:
                r = d2.chat([])
                results2.append(("ok" if r == "ok" else "malformed"))
            except ConnectionError:
                results2.append("fail")

        assert results1 == results2


# ═══════════════════════════════════════════════════════════════════════════════
# EvalReport + ReportFormatter
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvalReport:
    """Tests for EvalReport serialization."""

    def test_to_dict_roundtrip(self):
        report = EvalReport(
            corpus_dir="/tmp/test",
            document_count=10,
            entity_count=50,
            edge_count=100,
            chunk_count=200,
            metrics=[
                MetricResult(
                    name="Test Metric",
                    score=0.85,
                    baseline_score=0.60,
                    details={"key": "value"},
                    queries_evaluated=20,
                ),
            ],
            mode="graph_only",
            duration_seconds=12.5,
        )

        d = report.to_dict()
        assert d["document_count"] == 10
        assert d["entity_count"] == 50
        assert len(d["metrics"]) == 1
        assert d["metrics"][0]["score"] == 0.85
        assert d["metrics"][0]["baseline_score"] == 0.60

        # Verify JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed == d


class TestReportFormatter:
    """Tests for ReportFormatter."""

    def test_save_json(self, tmp_path):
        report = EvalReport(
            corpus_dir="/tmp/test",
            document_count=5,
            entity_count=10,
            edge_count=20,
            chunk_count=30,
            metrics=[
                MetricResult(name="M1", score=0.8, details={}, queries_evaluated=5),
            ],
            mode="graph_only",
            duration_seconds=1.0,
        )

        output = tmp_path / "report.json"
        ReportFormatter.save_json(report, output)

        assert output.exists()
        with open(output) as f:
            loaded = json.load(f)
        assert loaded["document_count"] == 5
        assert loaded["metrics"][0]["name"] == "M1"

    def test_print_console_no_error(self):
        """Verify console printing doesn't crash."""
        from rich.console import Console
        from io import StringIO

        report = EvalReport(
            corpus_dir="/tmp/test",
            document_count=5,
            entity_count=10,
            edge_count=20,
            chunk_count=30,
            metrics=[
                MetricResult(name="Retrieval Precision @5", score=0.82,
                           baseline_score=0.61, details={}, queries_evaluated=10),
                MetricResult(name="Co-occurrence Accuracy", score=0.88,
                           details={}, queries_evaluated=50),
                MetricResult(name="Context Efficiency", score=0.91,
                           baseline_score=0.43, details={}, queries_evaluated=20),
            ],
            mode="graph_only",
            duration_seconds=5.3,
        )

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        ReportFormatter.print_console(report, console)

        output = buf.getvalue()
        assert "Graphite Evaluation Report" in output
        assert "Retrieval Precision @5" in output


# ═══════════════════════════════════════════════════════════════════════════════
# MetricResult dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_defaults(self):
        m = MetricResult(name="Test", score=0.5, details={})
        assert m.queries_evaluated == 0
        assert m.errors == []
        assert m.baseline_score is None

    def test_with_all_fields(self):
        m = MetricResult(
            name="Test",
            score=0.75,
            details={"key": "val"},
            queries_evaluated=10,
            errors=["err1"],
            baseline_score=0.50,
        )
        assert m.score == 0.75
        assert m.baseline_score == 0.50
        assert len(m.errors) == 1
