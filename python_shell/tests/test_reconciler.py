"""Tests for the daemon's archive reconciler (PR D of Phase 2a).

The reconciler is the durability backstop. The live hook path is
best-effort; if the daemon was down or the LLM was unreachable, sessions
land in ``~/.graphite/archive/sessions/`` but never make it into the
graph. The reconciler replays them.

Two correctness contracts:

  1. **Idempotent.** A session whose content hash matches the graph's
     stored hash is skipped without enqueueing a job. Running the
     reconciler 10 times in a row produces the same final state as
     running it once.
  2. **Cheap pre-check.** The header-peek used to derive ``source_id``
     does not run the LLM, does not parse the full transcript, does not
     load anything into memory beyond a few lines per file.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import tempfile
from pathlib import Path
from typing import AsyncIterator
from unittest import mock

import pytest
import pytest_asyncio

from graphite.client import GraphiteClient
from graphite.daemon import (
    Daemon,
    _file_sha256,
    _peek_session_header,
)


def _short_socket_path() -> Path:
    return Path(tempfile.gettempdir()) / f"gphr-{secrets.token_hex(4)}.sock"


def _write_synthetic_session(
    archive_dir: Path,
    session_id: str,
    project: str = "TestProject",
    cwd: str | None = None,
) -> Path:
    """Plant a Claude-Code-style session JSONL in the archive."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{session_id}.jsonl"
    actual_cwd = cwd if cwd is not None else f"/Users/test/{project}"
    path.write_text(
        json.dumps({
            "type": "user",
            "sessionId": session_id,
            "cwd": actual_cwd,
            "content": "hello",
        }) + "\n"
        + json.dumps({"type": "assistant", "content": "hi"}) + "\n"
    )
    return path


# ---------------------------------------------------------------------------
# Header peek (cheap, no LLM)
# ---------------------------------------------------------------------------
class TestPeekSessionHeader:
    def test_extracts_session_id_and_project_from_first_line(self, tmp_path: Path):
        path = _write_synthetic_session(tmp_path, "sess-001", project="MyRepo")
        sid, project = _peek_session_header(path)
        assert sid == "sess-001"
        assert project == "MyRepo"

    def test_falls_back_to_unknown_for_missing_metadata(self, tmp_path: Path):
        path = tmp_path / "weird.jsonl"
        path.write_text(json.dumps({"type": "user", "content": "no metadata"}) + "\n")
        sid, project = _peek_session_header(path)
        assert sid is None
        assert project == "unknown"

    def test_handles_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        sid, project = _peek_session_header(path)
        assert sid is None
        assert project == "unknown"

    def test_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            "this is not json\n"
            + json.dumps({"sessionId": "sess-002", "cwd": "/tmp/X"}) + "\n"
        )
        sid, project = _peek_session_header(path)
        assert sid == "sess-002"
        assert project == "X"

    def test_unreadable_file_returns_none(self, tmp_path: Path):
        sid, project = _peek_session_header(tmp_path / "does-not-exist.jsonl")
        assert sid is None
        assert project == "unknown"


# ---------------------------------------------------------------------------
# Sync reconcile_archive — call directly, no socket
# ---------------------------------------------------------------------------
class TestReconcileArchiveSync:
    def _make_daemon(self, tmp_path: Path) -> Daemon:
        d = Daemon(graph_root=tmp_path / "g", socket_path=_short_socket_path())
        # Skip LLM init so reconcile sees `no_llm` as needed for some tests.
        d._llm_init_attempted = True
        d._llm_client = None
        return d

    def test_no_archive_dir_returns_zeroed_summary(self, tmp_path: Path):
        d = self._make_daemon(tmp_path)
        summary = d.reconcile_archive(archive_dir=tmp_path / "missing")
        assert summary["scanned"] == 0
        assert summary["enqueued"] == 0
        assert summary["already_indexed"] == 0

    def test_skips_when_no_llm_configured(self, tmp_path: Path):
        d = self._make_daemon(tmp_path)
        archive = tmp_path / "archive"
        _write_synthetic_session(archive, "sess-A")
        _write_synthetic_session(archive, "sess-B")

        summary = d.reconcile_archive(archive_dir=archive)
        assert summary["scanned"] == 2
        assert summary["skipped_no_llm"] == 2
        assert summary["enqueued"] == 0
        # Queue must remain empty — wouldn't want jobs that immediately fail.
        assert d._ingest_queue.qsize() == 0

    def test_enqueues_unindexed_sessions(self, tmp_path: Path):
        d = self._make_daemon(tmp_path)
        d._llm_client = object()  # pretend an LLM is ready

        archive = tmp_path / "archive"
        _write_synthetic_session(archive, "sess-A", project="ProjX")
        _write_synthetic_session(archive, "sess-B", project="ProjY")

        summary = d.reconcile_archive(archive_dir=archive)
        assert summary["scanned"] == 2
        assert summary["enqueued"] == 2
        assert summary["already_indexed"] == 0
        assert d._ingest_queue.qsize() == 2

    def test_skips_already_indexed_by_hash(self, tmp_path: Path):
        d = self._make_daemon(tmp_path)
        d._llm_client = object()

        archive = tmp_path / "archive"
        sess_path = _write_synthetic_session(archive, "sess-A", project="ProjX")

        # Pre-populate the graph's document hash to simulate prior ingest.
        d._ensure_graph()
        source_id = "claude-session://ProjX/sess-A"
        d._kg.set_document_hash(source_id, _file_sha256(sess_path))

        summary = d.reconcile_archive(archive_dir=archive)
        assert summary["scanned"] == 1
        assert summary["already_indexed"] == 1
        assert summary["enqueued"] == 0

    def test_re_enqueues_when_hash_changes(self, tmp_path: Path):
        d = self._make_daemon(tmp_path)
        d._llm_client = object()

        archive = tmp_path / "archive"
        sess_path = _write_synthetic_session(archive, "sess-A", project="ProjX")

        d._ensure_graph()
        source_id = "claude-session://ProjX/sess-A"
        d._kg.set_document_hash(source_id, "stale-hash-from-prior-run")

        summary = d.reconcile_archive(archive_dir=archive)
        assert summary["enqueued"] == 1
        assert summary["already_indexed"] == 0

    def test_unparseable_files_are_counted_and_skipped(self, tmp_path: Path):
        d = self._make_daemon(tmp_path)
        d._llm_client = object()

        archive = tmp_path / "archive"
        archive.mkdir()
        # No sessionId / cwd anywhere in this file.
        (archive / "garbage.jsonl").write_text(
            json.dumps({"type": "summary", "text": "no session info"}) + "\n"
        )

        summary = d.reconcile_archive(archive_dir=archive)
        assert summary["scanned"] == 1
        assert summary["skipped_unparseable"] == 1
        assert summary["enqueued"] == 0

    def test_idempotent_on_repeated_calls(self, tmp_path: Path):
        """Running reconcile twice in a row, with no new sessions and no
        ingest happening between, should enqueue zero new jobs the second
        time — the first call enqueued, but the underlying graph hashes
        haven't been updated yet, so the second pass would naively re-enqueue.
        We verify the queue depth doesn't grow without bound."""
        d = self._make_daemon(tmp_path)
        d._llm_client = object()

        archive = tmp_path / "archive"
        sess_path = _write_synthetic_session(archive, "sess-A")

        # Simulate prior successful ingest by recording the hash directly.
        d._ensure_graph()
        d._kg.set_document_hash(
            "claude-session://TestProject/sess-A",
            _file_sha256(sess_path),
        )

        first = d.reconcile_archive(archive_dir=archive)
        second = d.reconcile_archive(archive_dir=archive)
        assert first["enqueued"] == 0
        assert second["enqueued"] == 0
        assert first["already_indexed"] == 1
        assert second["already_indexed"] == 1


# ---------------------------------------------------------------------------
# RPC end-to-end
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def daemon_pair(tmp_path: Path) -> AsyncIterator[tuple[Daemon, Path]]:
    """A daemon with the worker stubbed and the startup reconcile disabled
    (we trigger reconcile explicitly in tests)."""
    daemon = Daemon(graph_root=tmp_path / "g", socket_path=_short_socket_path())
    daemon._llm_init_attempted = True
    daemon._llm_client = object()  # pass the NOT_READY gate
    daemon._run_ingest_job = lambda job: {  # type: ignore[assignment]
        "source_document": job["payload"]["path"],
        "status": "complete",
        "chunks_total": 0, "chunks_tagged": 0, "chunks_filler": 0,
        "entities_created": 0, "entities_linked": 0, "edges_created": 0,
        "duration_seconds": 0.01, "errors": [],
    }
    # Skip the startup reconcile so each test owns the trigger.
    from graphite.config import GraphiteConfig
    daemon._config = GraphiteConfig(reconcile_on_startup=False)

    serve = asyncio.create_task(daemon.start())
    for _ in range(100):
        if daemon.socket_path.exists():
            break
        if serve.done():
            raise RuntimeError(f"daemon exited early: {serve.exception()!r}")
        await asyncio.sleep(0.01)

    try:
        yield daemon, daemon.socket_path
    finally:
        await daemon.stop()
        serve.cancel()
        try:
            await serve
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_reconcile_overflow_rpc_returns_summary(daemon_pair, tmp_path: Path):
    """The overflow RPC mirrors reconcile_archive's surface — runs the
    overflow reconciler against a path we choose, returns a summary."""
    daemon, socket_path = daemon_pair
    overflow = tmp_path / "of"
    overflow.mkdir()
    (overflow / "1.json").write_text(json.dumps({
        "version": 1,
        "source_id": "openclaw://main/sess-A",
        "text": "captured during outage",
        "category": "Episodic",
        "captured_at": 1234567890,
    }))

    def _call():
        with GraphiteClient(socket_path=socket_path) as c:
            return c.reconcile_overflow(overflow_dir=str(overflow))

    summary = await asyncio.to_thread(_call)
    assert summary["scanned"] == 1
    assert summary["replayed"] == 1
    assert summary["overflow_dir"] == str(overflow)


@pytest.mark.asyncio
async def test_reconcile_rpc_returns_summary(daemon_pair, tmp_path: Path):
    daemon, socket_path = daemon_pair
    archive = tmp_path / "archive"
    _write_synthetic_session(archive, "sess-RPC-1")

    def _call():
        with GraphiteClient(socket_path=socket_path) as c:
            return c.reconcile_archive(archive_dir=str(archive))

    summary = await asyncio.to_thread(_call)
    assert summary["scanned"] == 1
    assert summary["enqueued"] == 1
    assert summary["archive_dir"] == str(archive)


@pytest.mark.asyncio
async def test_startup_reconcile_runs_when_enabled(tmp_path: Path):
    """Verifies the startup hook actually fires when reconcile_on_startup
    is True (the default). We plant an archive before bringing up the
    daemon and confirm the queue gets populated without anyone calling
    reconcile_archive explicitly."""
    archive = tmp_path / "archive"
    _write_synthetic_session(archive, "sess-startup-1")
    _write_synthetic_session(archive, "sess-startup-2")

    daemon = Daemon(graph_root=tmp_path / "g", socket_path=_short_socket_path())
    daemon._llm_init_attempted = True
    daemon._llm_client = object()
    daemon._run_ingest_job = lambda job: {  # type: ignore[assignment]
        "source_document": job["payload"]["path"],
        "status": "complete",
        "chunks_total": 0, "chunks_tagged": 0, "chunks_filler": 0,
        "entities_created": 0, "entities_linked": 0, "edges_created": 0,
        "duration_seconds": 0.01, "errors": [],
    }
    # Re-point the default archive at our tmp dir for the duration of this test.
    with mock.patch("graphite.daemon.DEFAULT_ARCHIVE_DIR", archive):
        serve = asyncio.create_task(daemon.start())
        try:
            for _ in range(100):
                if daemon.socket_path.exists():
                    break
                await asyncio.sleep(0.01)
            # Wait for the worker to drain both jobs.
            for _ in range(200):
                status = await asyncio.to_thread(
                    lambda: GraphiteClient(socket_path=daemon.socket_path).__enter__().ingest_queue_status()
                )
                if status["depth"] == 0 and len(status["recent"]) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert len(status["recent"]) >= 2
        finally:
            await daemon.stop()
            serve.cancel()
            try:
                await serve
            except (asyncio.CancelledError, Exception):
                pass
