"""SQLite-backed durable spool for fast `remember()` capture.

Phase 2b's central data structure. ``remember(text)`` writes one row here
and returns within milliseconds; the batch extractor (PR F) drains pending
rows asynchronously and runs them through the LLM pipeline.

Design notes:

  * **Single writer.** The graphited daemon owns this file. Concurrent
    access from the daemon's async handlers and its ingest worker thread
    is serialized via the ``threading.Lock`` on the ``Spool`` instance.
    SQLite is opened with ``check_same_thread=False`` to allow that
    cross-thread sharing.
  * **WAL mode.** Reads (status queries) don't block writes (``add``).
  * **No migrations yet.** A ``schema_meta`` row records the current
    schema version; if Phase 4 needs to evolve the schema, the bump goes
    here.
  * **Failure modes are first-class.** Fragments cycle through
    ``pending → extracting → extracted | failed``. A daemon crash leaves
    rows stuck in ``extracting``; ``reset_stale_extracting`` (called on
    startup) bounces them back to ``pending`` so they get picked up
    again.
  * **Retention.** Extracted rows linger 30 days by default — they're a
    cheap audit trail for "which prompt produced which entity" debugging
    and let us re-extract with a smarter prompt later. ``cleanup_old``
    purges them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_VERSION = "1"

# Status values. The state machine is:
#   add()                                  -> pending
#   claim_batch()       pending           -> extracting
#   mark_extracted()    extracting        -> extracted
#   mark_failed()       extracting        -> failed
#   reset_stale_extracting() extracting   -> pending
#   retry_failed()      failed            -> pending
STATUS_PENDING = "pending"
STATUS_EXTRACTING = "extracting"
STATUS_EXTRACTED = "extracted"
STATUS_FAILED = "failed"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fragments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id           TEXT NOT NULL,
    text                TEXT NOT NULL,
    category            TEXT NOT NULL DEFAULT 'Episodic',
    project             TEXT,
    entity_hints        TEXT,
    created_at          INTEGER NOT NULL,
    extracted_at        INTEGER,
    extraction_status   TEXT NOT NULL DEFAULT 'pending',
    batch_id            TEXT,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_status_created ON fragments(extraction_status, created_at);
CREATE INDEX IF NOT EXISTS idx_source         ON fragments(source_id);
CREATE INDEX IF NOT EXISTS idx_batch          ON fragments(batch_id);
"""


@dataclass(frozen=True)
class Fragment:
    id: int
    source_id: str
    text: str
    category: str
    project: Optional[str]
    entity_hints: Optional[list[str]]
    created_at: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "text": self.text,
            "category": self.category,
            "project": self.project,
            "entity_hints": self.entity_hints,
            "created_at": self.created_at,
        }


class Spool:
    """Thread-safe SQLite-backed fragment spool."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False because the daemon's asyncio handlers and
        # its ingest worker thread will both hit this connection. We
        # serialize access ourselves via _lock.
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions manually
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()

        self._init_schema()

    # -- internals --------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            cur = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            )
            row = cur.fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
                    (SCHEMA_VERSION,),
                )
            elif row["value"] != SCHEMA_VERSION:
                # Phase 4 will plug a real migration here. For now the only
                # version is "1"; a mismatch means the user pointed us at a
                # spool from a future Graphite. Refuse to clobber.
                raise RuntimeError(
                    f"Spool schema version mismatch: got {row['value']!r}, "
                    f"this build understands {SCHEMA_VERSION!r}. Refusing to "
                    f"open {self.db_path}."
                )

    @staticmethod
    def _row_to_fragment(row: sqlite3.Row) -> Fragment:
        hints_raw = row["entity_hints"]
        hints: Optional[list[str]] = None
        if hints_raw:
            try:
                parsed = json.loads(hints_raw)
                if isinstance(parsed, list):
                    hints = [str(x) for x in parsed]
            except json.JSONDecodeError:
                hints = None
        return Fragment(
            id=row["id"],
            source_id=row["source_id"],
            text=row["text"],
            category=row["category"],
            project=row["project"],
            entity_hints=hints,
            created_at=row["created_at"],
        )

    # -- public API -------------------------------------------------------

    def add(
        self,
        text: str,
        source_id: str,
        *,
        category: str = "Episodic",
        project: Optional[str] = None,
        entity_hints: Optional[Iterable[str]] = None,
    ) -> int:
        """Insert one fragment. Returns the assigned fragment id."""
        if not text:
            raise ValueError("Cannot spool an empty fragment")
        if not source_id:
            raise ValueError("Cannot spool a fragment with empty source_id")

        hints_json = None
        if entity_hints is not None:
            hints_list = [str(h) for h in entity_hints if h]
            if hints_list:
                hints_json = json.dumps(hints_list)

        now = int(time.time())
        with self._lock, self._conn:
            cur = self._conn.execute(
                """INSERT INTO fragments
                       (source_id, text, category, project, entity_hints, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (source_id, text, category, project, hints_json, now),
            )
            return int(cur.lastrowid)

    def pending_count(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS c FROM fragments WHERE extraction_status = ?",
                (STATUS_PENDING,),
            )
            return int(cur.fetchone()["c"])

    def status_counts(self) -> dict:
        """Histogram of fragments by extraction_status."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT extraction_status AS s, COUNT(*) AS c
                       FROM fragments GROUP BY extraction_status"""
            )
            counts = {STATUS_PENDING: 0, STATUS_EXTRACTING: 0, STATUS_EXTRACTED: 0, STATUS_FAILED: 0}
            for row in cur.fetchall():
                counts[row["s"]] = int(row["c"])
            counts["total"] = sum(counts.values())
            return counts

    def claim_batch(
        self,
        limit: int = 50,
        source_filter: Optional[str] = None,
    ) -> list[Fragment]:
        """Atomically take up to ``limit`` ``pending`` rows and mark them
        ``extracting``. Caller owns those rows until they're either marked
        extracted or marked failed. Crash leaves them in ``extracting`` —
        ``reset_stale_extracting`` bounces them back on next startup.

        Within a claim, rows come back ordered by ``source_id`` then
        ``created_at`` so the batch extractor can group consecutive
        same-source rows into one synthetic document.
        """
        if limit <= 0:
            return []

        with self._lock, self._conn:
            # Begin an explicit transaction so the SELECT + UPDATE pair are
            # atomic w.r.t. concurrent claimers.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if source_filter is not None:
                    cur = self._conn.execute(
                        """SELECT * FROM fragments
                            WHERE extraction_status = ? AND source_id = ?
                            ORDER BY source_id, created_at, id
                            LIMIT ?""",
                        (STATUS_PENDING, source_filter, limit),
                    )
                else:
                    cur = self._conn.execute(
                        """SELECT * FROM fragments
                            WHERE extraction_status = ?
                            ORDER BY source_id, created_at, id
                            LIMIT ?""",
                        (STATUS_PENDING, limit),
                    )
                rows = cur.fetchall()
                if not rows:
                    self._conn.execute("COMMIT")
                    return []

                ids = [row["id"] for row in rows]
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"""UPDATE fragments
                            SET extraction_status = ?
                          WHERE id IN ({placeholders})""",
                    (STATUS_EXTRACTING, *ids),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return [self._row_to_fragment(row) for row in rows]

    def mark_extracted(self, ids: Iterable[int], batch_id: Optional[str] = None) -> int:
        ids_list = [int(i) for i in ids]
        if not ids_list:
            return 0
        if batch_id is None:
            batch_id = uuid.uuid4().hex[:12]
        now = int(time.time())
        placeholders = ",".join("?" for _ in ids_list)
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"""UPDATE fragments
                        SET extraction_status = ?, extracted_at = ?, batch_id = ?, error = NULL
                      WHERE id IN ({placeholders})""",
                (STATUS_EXTRACTED, now, batch_id, *ids_list),
            )
            return cur.rowcount

    def mark_failed(self, ids: Iterable[int], error: str, batch_id: Optional[str] = None) -> int:
        ids_list = [int(i) for i in ids]
        if not ids_list:
            return 0
        placeholders = ",".join("?" for _ in ids_list)
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"""UPDATE fragments
                        SET extraction_status = ?, error = ?, batch_id = ?
                      WHERE id IN ({placeholders})""",
                (STATUS_FAILED, error, batch_id, *ids_list),
            )
            return cur.rowcount

    def reset_stale_extracting(self) -> int:
        """Bounce any rows stuck in ``extracting`` back to ``pending``.
        Called on daemon startup to recover from a crash mid-batch."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                """UPDATE fragments
                       SET extraction_status = ?
                     WHERE extraction_status = ?""",
                (STATUS_PENDING, STATUS_EXTRACTING),
            )
            return cur.rowcount

    def retry_failed(self) -> int:
        """Move all failed fragments back to pending. Triage tool — surface
        through ``graphite spool retry-failed``."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                """UPDATE fragments
                       SET extraction_status = ?, error = NULL
                     WHERE extraction_status = ?""",
                (STATUS_PENDING, STATUS_FAILED),
            )
            return cur.rowcount

    def cleanup_old(self, retain_days: int = 30) -> int:
        """Delete extracted fragments older than ``retain_days``. Failed
        rows are kept indefinitely so the user can triage them."""
        if retain_days <= 0:
            raise ValueError("retain_days must be positive")
        cutoff = int(time.time()) - retain_days * 86400
        with self._lock, self._conn:
            cur = self._conn.execute(
                """DELETE FROM fragments
                    WHERE extraction_status = ?
                      AND extracted_at IS NOT NULL
                      AND extracted_at < ?""",
                (STATUS_EXTRACTED, cutoff),
            )
            return cur.rowcount

    def get_failed(self, limit: int = 50) -> list[dict]:
        """Most recently failed fragments — for triage UI."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT id, source_id, text, error, created_at
                       FROM fragments
                      WHERE extraction_status = ?
                      ORDER BY id DESC
                      LIMIT ?""",
                (STATUS_FAILED, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def recent_batches(self, limit: int = 10) -> list[dict]:
        """Most recent batches by max extracted_at. Returns batch_id,
        fragment count, sources (capped to 5), and the most recent
        extraction timestamp."""
        with self._lock:
            cur = self._conn.execute(
                """SELECT batch_id,
                          MAX(extracted_at) AS extracted_at,
                          COUNT(*)         AS fragment_count
                     FROM fragments
                    WHERE batch_id IS NOT NULL
                      AND extraction_status = ?
                 GROUP BY batch_id
                 ORDER BY extracted_at DESC
                    LIMIT ?""",
                (STATUS_EXTRACTED, limit),
            )
            batches = [dict(row) for row in cur.fetchall()]

            for b in batches:
                src_cur = self._conn.execute(
                    """SELECT DISTINCT source_id FROM fragments
                            WHERE batch_id = ? LIMIT 5""",
                    (b["batch_id"],),
                )
                b["sources"] = [r["source_id"] for r in src_cur.fetchall()]
            return batches

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
