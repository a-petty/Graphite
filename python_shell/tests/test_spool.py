"""Tests for the SQLite-backed spool (PR E of Phase 2b).

Two contracts to lock down:

  1. **Atomic claim_batch.** A daemon crash, or a second claimer, must not
     produce duplicate work. Once a row is in ``extracting``, only one
     caller can move it forward.
  2. **Crash recovery.** ``reset_stale_extracting`` is the only mechanism
     that recovers from a crash mid-batch. It must move every
     ``extracting`` row back to ``pending`` so the next worker run picks
     them up.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from graphite.spool import (
    STATUS_EXTRACTED,
    STATUS_EXTRACTING,
    STATUS_FAILED,
    STATUS_PENDING,
    Spool,
)


# ---------------------------------------------------------------------------
# Schema lifecycle
# ---------------------------------------------------------------------------
class TestSchema:
    def test_creates_db_file_and_schema(self, tmp_path: Path):
        db = tmp_path / "spool.db"
        assert not db.exists()
        s = Spool(db)
        try:
            assert db.exists()
            # Schema row should be present.
            cur = s._conn.execute("SELECT value FROM schema_meta WHERE key='version'")
            assert cur.fetchone()["value"] == "1"
        finally:
            s.close()

    def test_creates_parent_dir(self, tmp_path: Path):
        db = tmp_path / "nested" / "deep" / "spool.db"
        s = Spool(db)
        try:
            assert db.exists()
        finally:
            s.close()

    def test_refuses_to_open_future_schema(self, tmp_path: Path):
        db = tmp_path / "spool.db"
        # Plant a future-version schema_meta row.
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta (key, value) VALUES ('version', '99');
            CREATE TABLE fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'Episodic',
                project TEXT,
                entity_hints TEXT,
                created_at INTEGER NOT NULL,
                extracted_at INTEGER,
                extraction_status TEXT NOT NULL DEFAULT 'pending',
                batch_id TEXT,
                error TEXT
            );
        """)
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="schema version mismatch"):
            Spool(db)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------
class TestAdd:
    def test_adds_row_returns_id(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            fid = s.add("hello", source_id="remember://x/1")
            assert isinstance(fid, int) and fid >= 1
            assert s.pending_count() == 1
        finally:
            s.close()

    def test_rejects_empty_text(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            with pytest.raises(ValueError):
                s.add("", source_id="x")
            with pytest.raises(ValueError):
                s.add("hi", source_id="")
        finally:
            s.close()

    def test_serializes_entity_hints_as_json(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            fid = s.add(
                "Sarah is always late.",
                source_id="remember://x/2",
                entity_hints=["Sarah", "Meetings"],
            )
            row = s._conn.execute(
                "SELECT entity_hints FROM fragments WHERE id=?", (fid,)
            ).fetchone()
            assert row["entity_hints"] == '["Sarah", "Meetings"]'

            # Round-trip via claim_batch
            batch = s.claim_batch()
            assert batch[0].entity_hints == ["Sarah", "Meetings"]
        finally:
            s.close()

    def test_default_status_is_pending(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("x", source_id="src")
            row = s._conn.execute(
                "SELECT extraction_status FROM fragments LIMIT 1"
            ).fetchone()
            assert row["extraction_status"] == STATUS_PENDING
        finally:
            s.close()


# ---------------------------------------------------------------------------
# claim_batch atomicity
# ---------------------------------------------------------------------------
class TestClaimBatch:
    def test_returns_fragments_in_source_then_time_order(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            # Interleave sources to verify batching groups them.
            s.add("a-first", source_id="A")
            s.add("b-first", source_id="B")
            s.add("a-second", source_id="A")
            s.add("b-second", source_id="B")

            batch = s.claim_batch(limit=10)
            assert [f.source_id for f in batch] == ["A", "A", "B", "B"]
            # Within a source, time order.
            assert batch[0].text == "a-first"
            assert batch[1].text == "a-second"
        finally:
            s.close()

    def test_marks_claimed_rows_as_extracting(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("x", source_id="src")
            s.add("y", source_id="src")
            batch = s.claim_batch(limit=5)
            assert len(batch) == 2
            counts = s.status_counts()
            assert counts[STATUS_PENDING] == 0
            assert counts[STATUS_EXTRACTING] == 2
        finally:
            s.close()

    def test_second_claim_sees_no_overlap(self, tmp_path: Path):
        """Two consecutive claim_batch calls without intervening
        mark_extracted/mark_failed must not return the same rows. This is
        the contract that prevents duplicate work."""
        s = Spool(tmp_path / "spool.db")
        try:
            for i in range(5):
                s.add(f"f{i}", source_id="src")

            first = s.claim_batch(limit=3)
            second = s.claim_batch(limit=3)

            first_ids = {f.id for f in first}
            second_ids = {f.id for f in second}
            assert first_ids.isdisjoint(second_ids)
            # Total claimed = 5 (3 + 2 leftover; second batch is short).
            assert len(first) + len(second) == 5
        finally:
            s.close()

    def test_source_filter_narrows_to_one_source(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("a1", source_id="A")
            s.add("a2", source_id="A")
            s.add("b1", source_id="B")

            batch = s.claim_batch(limit=10, source_filter="A")
            assert {f.source_id for f in batch} == {"A"}
            assert len(batch) == 2
            # B fragment is still pending.
            assert s.pending_count() == 1
        finally:
            s.close()

    def test_limit_zero_returns_empty_no_state_change(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("x", source_id="src")
            assert s.claim_batch(limit=0) == []
            assert s.pending_count() == 1
        finally:
            s.close()


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------
class TestStateTransitions:
    def test_mark_extracted_sets_batch_id_and_timestamp(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("x", source_id="src")
            batch = s.claim_batch()
            ids = [f.id for f in batch]

            n = s.mark_extracted(ids, batch_id="batch-42")
            assert n == 1

            row = s._conn.execute(
                "SELECT extraction_status, batch_id, extracted_at, error FROM fragments LIMIT 1"
            ).fetchone()
            assert row["extraction_status"] == STATUS_EXTRACTED
            assert row["batch_id"] == "batch-42"
            assert row["extracted_at"] is not None
            assert row["error"] is None
        finally:
            s.close()

    def test_mark_failed_records_error(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("x", source_id="src")
            batch = s.claim_batch()
            n = s.mark_failed([f.id for f in batch], error="LLM timeout", batch_id="b-x")
            assert n == 1
            row = s._conn.execute(
                "SELECT extraction_status, error FROM fragments LIMIT 1"
            ).fetchone()
            assert row["extraction_status"] == STATUS_FAILED
            assert row["error"] == "LLM timeout"
        finally:
            s.close()

    def test_reset_stale_extracting_recovers_from_crash(self, tmp_path: Path):
        """Daemon crashes mid-batch, leaving rows in ``extracting``. On
        startup, ``reset_stale_extracting`` bounces them back to
        ``pending``; nothing is lost."""
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("x", source_id="src")
            s.add("y", source_id="src")
            batch = s.claim_batch()
            assert s.status_counts()[STATUS_EXTRACTING] == 2

            # Simulate a crash: forget the batch handle, no mark_extracted.
            del batch

            # On startup the daemon would call this:
            bounced = s.reset_stale_extracting()
            assert bounced == 2

            counts = s.status_counts()
            assert counts[STATUS_PENDING] == 2
            assert counts[STATUS_EXTRACTING] == 0
        finally:
            s.close()

    def test_retry_failed_moves_failed_back_to_pending(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("x", source_id="src")
            batch = s.claim_batch()
            s.mark_failed([f.id for f in batch], error="boom")

            n = s.retry_failed()
            assert n == 1
            counts = s.status_counts()
            assert counts[STATUS_PENDING] == 1
            assert counts[STATUS_FAILED] == 0
            # Error cleared on retry.
            row = s._conn.execute("SELECT error FROM fragments LIMIT 1").fetchone()
            assert row["error"] is None
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------
class TestCleanup:
    def test_deletes_old_extracted_only(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            # Three rows: one old extracted, one fresh extracted, one failed.
            s.add("old", source_id="src")
            s.add("fresh", source_id="src")
            s.add("failed", source_id="src")

            batch = s.claim_batch(limit=10)
            old_id, fresh_id, failed_id = [f.id for f in batch]

            s.mark_extracted([old_id, fresh_id], batch_id="b1")
            s.mark_failed([failed_id], error="boom", batch_id="b2")

            # Backdate the "old" extracted row to 31 days ago.
            old_ts = int(time.time()) - (31 * 86400)
            s._conn.execute(
                "UPDATE fragments SET extracted_at=? WHERE id=?",
                (old_ts, old_id),
            )

            removed = s.cleanup_old(retain_days=30)
            assert removed == 1

            counts = s.status_counts()
            assert counts[STATUS_EXTRACTED] == 1  # fresh kept
            assert counts[STATUS_FAILED] == 1     # failed kept regardless of age
        finally:
            s.close()

    def test_cleanup_old_rejects_nonpositive_retain(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            with pytest.raises(ValueError):
                s.cleanup_old(retain_days=0)
            with pytest.raises(ValueError):
                s.cleanup_old(retain_days=-1)
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
class TestObservability:
    def test_recent_batches_orders_by_extracted_at_desc(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            for i in range(3):
                s.add(f"f{i}", source_id=f"src{i}")
            batch = s.claim_batch(limit=10)
            ids_per_batch = [[batch[0].id], [batch[1].id], [batch[2].id]]
            for i, ids in enumerate(ids_per_batch):
                s.mark_extracted(ids, batch_id=f"batch-{i}")
                # Advance time so each batch has a distinct extracted_at.
                s._conn.execute(
                    "UPDATE fragments SET extracted_at=? WHERE id=?",
                    (1_000_000 + i, ids[0]),
                )

            recent = s.recent_batches(limit=5)
            assert len(recent) == 3
            assert [b["batch_id"] for b in recent] == ["batch-2", "batch-1", "batch-0"]
            assert all(b["fragment_count"] == 1 for b in recent)
        finally:
            s.close()

    def test_get_failed_returns_recent_first(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            for i in range(3):
                s.add(f"f{i}", source_id="src")
            batch = s.claim_batch(limit=10)
            s.mark_failed([f.id for f in batch], error="boom", batch_id="b1")

            failed = s.get_failed(limit=2)
            assert len(failed) == 2
            # Most recent (highest id) first.
            assert failed[0]["id"] > failed[1]["id"]
            assert failed[0]["error"] == "boom"
        finally:
            s.close()

    def test_status_counts_includes_total(self, tmp_path: Path):
        s = Spool(tmp_path / "spool.db")
        try:
            s.add("a", source_id="src")
            s.add("b", source_id="src")
            counts = s.status_counts()
            assert counts["total"] == 2
            assert counts[STATUS_PENDING] == 2
            for k in (STATUS_EXTRACTING, STATUS_EXTRACTED, STATUS_FAILED):
                assert counts[k] == 0
        finally:
            s.close()
