"""Tests for the spool batch extractor (PR F of Phase 2b).

Three contracts to lock down:

  1. **Per-source grouping.** Fragments sharing a ``source_id`` are
     concatenated into one synthetic document, distinct sources go
     through separate ``ingest_text`` calls. No bleed.
  2. **Failure isolation.** A pipeline exception on one source does not
     prevent the extractor from processing other sources in the same
     batch.
  3. **Spool state machine respect.** Successful groups land
     ``extracted``; failed groups land ``failed``; the spool's
     ``claim_batch`` atomicity guarantees nothing is processed twice.

The extractor is decoupled from the real ``IngestionPipeline`` via a
``pipeline_factory`` callable; tests inject a fake pipeline that
records calls and returns canned ``IngestionResult``-like objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pytest

from graphite.spool import (
    STATUS_EXTRACTED,
    STATUS_FAILED,
    STATUS_PENDING,
    Spool,
)
from graphite.spool_extractor import BatchExtractor


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------
@dataclass
class FakeResult:
    """Stand-in for ingestion.pipeline.IngestionResult."""
    status: str = "complete"
    chunks_total: int = 1
    chunks_tagged: int = 1
    entities_created: int = 1
    entities_linked: int = 0
    edges_created: int = 1
    duration_seconds: float = 0.01
    errors: list = field(default_factory=list)


class FakePipeline:
    """Records every ingest_text call and returns a configured result."""

    def __init__(self, default_result: FakeResult | None = None):
        self.calls: List[Tuple[str, str, str]] = []  # (text, source_id, category)
        self._default_result = default_result or FakeResult()
        self._per_source: dict = {}  # source_id -> result | callable raising

    def configure(self, source_id: str, result_or_exc):
        self._per_source[source_id] = result_or_exc

    def ingest_text(self, text: str, source_id: str, memory_category: str) -> FakeResult:
        self.calls.append((text, source_id, memory_category))
        rule = self._per_source.get(source_id, self._default_result)
        if isinstance(rule, BaseException):
            raise rule
        return rule


@pytest.fixture
def spool(tmp_path: Path):
    s = Spool(tmp_path / "spool.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------
class TestEmptyCases:
    def test_empty_spool_returns_zero_summary(self, spool):
        pipeline = FakePipeline()
        extractor = BatchExtractor(spool, lambda: pipeline)
        summary = extractor.extract_batch()
        assert summary == {
            "claimed": 0, "groups": 0, "succeeded": 0, "failed": 0, "errors": [],
        }
        assert pipeline.calls == []


# ---------------------------------------------------------------------------
# Successful path
# ---------------------------------------------------------------------------
class TestHappyPath:
    def test_single_source_two_fragments_one_pipeline_call(self, spool):
        spool.add("first thought", source_id="remember://x")
        spool.add("second thought", source_id="remember://x")
        pipeline = FakePipeline()
        extractor = BatchExtractor(spool, lambda: pipeline)

        summary = extractor.extract_batch()

        assert summary["claimed"] == 2
        assert summary["groups"] == 1
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0

        # One pipeline call with concatenated text + correct source_id.
        assert len(pipeline.calls) == 1
        text, source_id, category = pipeline.calls[0]
        assert source_id == "remember://x"
        assert category == "Episodic"
        assert "first thought" in text
        assert "second thought" in text
        # Blank-line separator preserves structural-parser semantics.
        assert "first thought\n\nsecond thought" == text

        counts = spool.status_counts()
        assert counts[STATUS_EXTRACTED] == 2
        assert counts[STATUS_PENDING] == 0

    def test_multiple_sources_one_call_per_source(self, spool):
        spool.add("a1", source_id="A")
        spool.add("a2", source_id="A")
        spool.add("b1", source_id="B")
        spool.add("c1", source_id="C")

        pipeline = FakePipeline()
        extractor = BatchExtractor(spool, lambda: pipeline)
        summary = extractor.extract_batch()

        assert summary["groups"] == 3
        assert summary["succeeded"] == 4
        # One pipeline call per source.
        sources_called = sorted(call[1] for call in pipeline.calls)
        assert sources_called == ["A", "B", "C"]

    def test_assigns_unique_batch_id_per_group(self, spool):
        spool.add("a", source_id="A")
        spool.add("b", source_id="B")

        pipeline = FakePipeline()
        extractor = BatchExtractor(spool, lambda: pipeline)
        extractor.extract_batch()

        # Each source gets its own batch_id.
        cur = spool._conn.execute(
            "SELECT source_id, batch_id FROM fragments ORDER BY source_id"
        )
        rows = cur.fetchall()
        assert len(rows) == 2
        assert rows[0]["batch_id"] != rows[1]["batch_id"]
        assert all(r["batch_id"].startswith("batch-") for r in rows)

    def test_partial_status_still_marks_extracted(self, spool):
        """``partial`` means some chunks failed to tag but the document
        itself was processed. We don't want to retry on that — record the
        partial result and move on."""
        spool.add("partial-source", source_id="P")
        pipeline = FakePipeline(
            default_result=FakeResult(status="partial", errors=["one chunk failed"]),
        )
        extractor = BatchExtractor(spool, lambda: pipeline)
        summary = extractor.extract_batch()

        assert summary["succeeded"] == 1
        assert summary["failed"] == 0
        assert spool.status_counts()[STATUS_EXTRACTED] == 1


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------
class TestFailureIsolation:
    def test_pipeline_exception_marks_only_that_source_failed(self, spool):
        spool.add("good", source_id="G")
        spool.add("bad", source_id="B")
        spool.add("also-good", source_id="G2")

        pipeline = FakePipeline()
        pipeline.configure("B", RuntimeError("boom"))
        extractor = BatchExtractor(spool, lambda: pipeline)
        summary = extractor.extract_batch()

        assert summary["succeeded"] == 2  # G + G2
        assert summary["failed"] == 1     # B
        assert "B: boom" in summary["errors"][0]

        counts = spool.status_counts()
        assert counts[STATUS_EXTRACTED] == 2
        assert counts[STATUS_FAILED] == 1

        # The failed fragment carries the error string.
        failed = spool.get_failed()
        assert len(failed) == 1
        assert failed[0]["source_id"] == "B"
        assert "boom" in failed[0]["error"]

    def test_status_failed_result_marks_failed(self, spool):
        spool.add("bad-status", source_id="X")
        pipeline = FakePipeline(
            default_result=FakeResult(status="failed", errors=["pass 2 broke"]),
        )
        extractor = BatchExtractor(spool, lambda: pipeline)
        summary = extractor.extract_batch()

        assert summary["failed"] == 1
        assert summary["succeeded"] == 0
        failed = spool.get_failed()
        assert "pass 2 broke" in failed[0]["error"]

    def test_pipeline_factory_failure_bounces_all_to_failed(self, spool):
        spool.add("a", source_id="A")
        spool.add("b", source_id="B")

        def _failing_factory():
            raise RuntimeError("LLM not configured")

        extractor = BatchExtractor(spool, _failing_factory)
        summary = extractor.extract_batch()

        assert summary["claimed"] == 2
        assert summary["failed"] == 2
        assert summary["succeeded"] == 0
        # Every claimed fragment is now failed with the construction error.
        failed = spool.get_failed()
        assert len(failed) == 2
        assert all("LLM not configured" in f["error"] for f in failed)


# ---------------------------------------------------------------------------
# Source filtering
# ---------------------------------------------------------------------------
class TestSourceFilter:
    def test_only_filtered_source_processed(self, spool):
        spool.add("a1", source_id="A")
        spool.add("a2", source_id="A")
        spool.add("b1", source_id="B")

        pipeline = FakePipeline()
        extractor = BatchExtractor(spool, lambda: pipeline)
        summary = extractor.extract_batch(source_filter="A")

        assert summary["claimed"] == 2
        # B fragment still pending — wasn't claimed.
        counts = spool.status_counts()
        assert counts[STATUS_EXTRACTED] == 2
        assert counts[STATUS_PENDING] == 1
        assert pipeline.calls[0][1] == "A"  # only A was called


# ---------------------------------------------------------------------------
# Limit handling
# ---------------------------------------------------------------------------
class TestBatchSizeLimit:
    def test_limit_caps_claimed_fragments(self, spool):
        for i in range(5):
            spool.add(f"frag-{i}", source_id=f"src-{i}")

        pipeline = FakePipeline()
        extractor = BatchExtractor(spool, lambda: pipeline)
        summary = extractor.extract_batch(batch_size_limit=3)

        assert summary["claimed"] == 3
        # Two left pending for the next batch.
        assert spool.pending_count() == 2
