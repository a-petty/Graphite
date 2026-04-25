"""Tests for the daemon's ingest queue + worker (PR A of Phase 2a).

Unit layer: validates the queue mechanics, status reporting, lifecycle, and
failure handling. We monkey-patch ``Daemon._run_ingest_job`` so the tests
don't need a live LLM; actual end-to-end pipeline correctness is covered by
the ingestion test suite and by manual integration.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import tempfile
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from graphite.client import DaemonError, GraphiteClient
from graphite.daemon import Daemon
from graphite.protocol import ErrorCode


def _short_socket_path() -> Path:
    """Keep sockets under macOS's ~104-char sun_path limit (see test_daemon.py)."""
    return Path(tempfile.gettempdir()) / f"gphi-{secrets.token_hex(4)}.sock"


def _client_call(socket_path: Path, method: str, params=None):
    with GraphiteClient(socket_path=socket_path) as c:
        return c.call(method, params or {})


@pytest_asyncio.fixture
async def daemon_pair(tmp_path: Path) -> AsyncIterator[tuple[Daemon, Path]]:
    """Spin up a Daemon with monkey-patched ingest so tests don't hit an LLM."""
    graph_root = tmp_path / "graph_home"
    socket_path = _short_socket_path()
    daemon = Daemon(graph_root=graph_root, socket_path=socket_path)

    # Stub out the sync pipeline body with something that doesn't need an LLM.
    # Tests that want to exercise the LLM-missing path clear this override.
    def _fake_run(job: dict) -> dict:
        payload = job["payload"]
        if payload.get("simulate_failure"):
            raise RuntimeError(payload.get("failure_msg", "simulated failure"))
        return {
            "source_document": payload.get("path", "synthetic://test"),
            "status": "complete",
            "chunks_total": 3,
            "chunks_tagged": 2,
            "chunks_filler": 1,
            "entities_created": 2,
            "entities_linked": 0,
            "edges_created": 1,
            "duration_seconds": 0.01,
            "errors": [],
        }

    daemon._run_ingest_job = _fake_run  # type: ignore[assignment]

    serve = asyncio.create_task(daemon.start())
    for _ in range(100):
        if socket_path.exists():
            break
        if serve.done():
            raise RuntimeError(f"daemon exited early: {serve.exception()!r}")
        await asyncio.sleep(0.01)

    try:
        yield daemon, socket_path
    finally:
        await daemon.stop()
        serve.cancel()
        try:
            await serve
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Queue status reporting
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_queue_status_on_empty_daemon(daemon_pair):
    _, socket_path = daemon_pair
    status = await asyncio.to_thread(_client_call, socket_path, "ingest_queue_status")
    assert status["depth"] == 0
    assert status["current"] is None
    assert status["recent"] == []
    # LLM may or may not be configured depending on env; it's a boolean either way.
    assert isinstance(status["llm_configured"], bool)


@pytest.mark.asyncio
async def test_queue_status_reports_llm_provider_from_config(daemon_pair):
    daemon, socket_path = daemon_pair
    # _ensure_config has already run in start() — llm_provider defaults to "ollama".
    status = await asyncio.to_thread(_client_call, socket_path, "ingest_queue_status")
    assert status["llm_provider"] == daemon._config.llm_provider


# ---------------------------------------------------------------------------
# Enqueue-level gating
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_enqueue_rejected_without_llm_unless_forced(daemon_pair):
    daemon, socket_path = daemon_pair
    # Force the "no LLM available" state even if ollama is nominally reachable,
    # so the test is deterministic across dev environments.
    daemon._llm_client = None
    daemon._llm_init_attempted = True  # skip actual construction

    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(
            _client_call, socket_path, "enqueue_session_ingest",
            {"path": "/tmp/doesnt-matter.jsonl"},
        )
    assert ei.value.code == int(ErrorCode.NOT_READY)
    assert "no llm" in ei.value.message.lower()


@pytest.mark.asyncio
async def test_enqueue_with_force_bypasses_llm_check(daemon_pair):
    daemon, socket_path = daemon_pair
    daemon._llm_client = None
    daemon._llm_init_attempted = True

    result = await asyncio.to_thread(
        _client_call, socket_path, "enqueue_session_ingest",
        {"path": "/tmp/session.jsonl", "force": True},
    )
    assert "job_id" in result
    assert result["queue_position"] >= 1


@pytest.mark.asyncio
async def test_enqueue_requires_path_string(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(
            _client_call, socket_path, "enqueue_session_ingest", {},
        )
    assert ei.value.code == int(ErrorCode.INVALID_PARAMS)


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------
async def _wait_for_recent(socket_path: Path, job_id: str, timeout_s: float = 2.0) -> dict:
    """Poll ``ingest_queue_status`` until the given job shows up in ``recent``."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        status = await asyncio.to_thread(_client_call, socket_path, "ingest_queue_status")
        for job in status["recent"]:
            if job["id"] == job_id:
                return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} never showed up in recent within {timeout_s}s")


@pytest.mark.asyncio
async def test_worker_drains_single_job_and_records_completion(daemon_pair):
    daemon, socket_path = daemon_pair
    # Fixture stubbed _run_ingest_job to succeed; force past the LLM gate.
    daemon._llm_client = object()  # any non-None passes the NOT_READY check
    daemon._llm_init_attempted = True

    enq = await asyncio.to_thread(
        _client_call, socket_path, "enqueue_session_ingest",
        {"path": "/tmp/synthetic.jsonl"},
    )
    job_id = enq["job_id"]

    job = await _wait_for_recent(socket_path, job_id)
    assert job["status"] == "complete"
    assert job["result"]["entities_created"] == 2
    assert job["error"] is None
    assert job["started_at"] is not None
    assert job["finished_at"] is not None


@pytest.mark.asyncio
async def test_worker_records_failed_jobs_without_crashing(daemon_pair):
    daemon, socket_path = daemon_pair
    daemon._llm_client = object()
    daemon._llm_init_attempted = True

    # Inject a failure job directly (bypasses enqueue validation), since the
    # public socket surface only accepts path/project/force and shouldn't grow
    # a back-door for test hooks.
    failing = daemon.enqueue_ingest_job("session", {
        "path": "/tmp/will-fail.jsonl",
        "simulate_failure": True,
        "failure_msg": "boom",
    })
    job = await _wait_for_recent(socket_path, failing["job_id"])
    assert job["status"] == "failed"
    assert "boom" in job["error"]

    # Worker must still be alive after a failure — enqueue another job and
    # verify it completes.
    enq2 = await asyncio.to_thread(
        _client_call, socket_path, "enqueue_session_ingest",
        {"path": "/tmp/next.jsonl"},
    )
    job2 = await _wait_for_recent(socket_path, enq2["job_id"])
    assert job2["status"] == "complete"


@pytest.mark.asyncio
async def test_worker_drains_jobs_serially(daemon_pair):
    """The worker must not run two ingest jobs in parallel — local LLMs
    don't tolerate concurrency well, and canon resolution wants one session
    at a time."""
    daemon, socket_path = daemon_pair
    daemon._llm_client = object()
    daemon._llm_init_attempted = True

    seen_concurrent = 0
    in_flight = 0
    ran = asyncio.Event()

    def _slow_run(job: dict) -> dict:
        nonlocal in_flight, seen_concurrent
        in_flight += 1
        if in_flight > 1:
            seen_concurrent += 1
        # Busy-wait briefly so a second job has a chance to race if parallelism
        # were permitted. No asyncio here — this runs on an executor thread.
        import time
        time.sleep(0.05)
        in_flight -= 1
        ran.set()
        return {
            "source_document": job["payload"]["path"],
            "status": "complete",
            "chunks_total": 0, "chunks_tagged": 0, "chunks_filler": 0,
            "entities_created": 0, "entities_linked": 0, "edges_created": 0,
            "duration_seconds": 0.05, "errors": [],
        }

    daemon._run_ingest_job = _slow_run  # type: ignore[assignment]

    ids = []
    for i in range(3):
        enq = await asyncio.to_thread(
            _client_call, socket_path, "enqueue_session_ingest",
            {"path": f"/tmp/j{i}.jsonl"},
        )
        ids.append(enq["job_id"])

    for job_id in ids:
        await _wait_for_recent(socket_path, job_id, timeout_s=3.0)

    assert seen_concurrent == 0, "worker ran jobs in parallel — should be serial"


@pytest.mark.asyncio
async def test_remember_writes_to_spool_and_returns_immediately(daemon_pair):
    daemon, socket_path = daemon_pair

    def _call():
        with GraphiteClient(socket_path=socket_path) as c:
            return c.remember(text="I prefer pnpm over npm.")

    import time as _time
    start = _time.perf_counter()
    result = await asyncio.to_thread(_call)
    elapsed = _time.perf_counter() - start

    assert "fragment_id" in result
    assert result["pending_count"] >= 1
    # Tight bound — the spool RPC should be ~milliseconds, not seconds.
    # Allow generous headroom for slow CI without losing the contract.
    assert elapsed < 1.0, f"remember() took {elapsed:.3f}s — should be near-instant"


@pytest.mark.asyncio
async def test_remember_synthesizes_source_id_when_omitted(daemon_pair):
    _, socket_path = daemon_pair
    result = await asyncio.to_thread(
        _client_call, socket_path, "remember",
        {"text": "no source given"},
    )
    assert result["source_id"].startswith("remember://")


@pytest.mark.asyncio
async def test_remember_rejects_empty_text(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(
            _client_call, socket_path, "remember", {"text": ""},
        )
    assert ei.value.code == int(ErrorCode.INVALID_PARAMS)


@pytest.mark.asyncio
async def test_remember_rejects_bad_category(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(
            _client_call, socket_path, "remember",
            {"text": "x", "category": "Bogus"},
        )
    assert ei.value.code == int(ErrorCode.INVALID_PARAMS)


@pytest.mark.asyncio
async def test_spool_status_reports_counts(daemon_pair):
    _, socket_path = daemon_pair
    # Empty spool.
    status = await asyncio.to_thread(_client_call, socket_path, "spool_status")
    counts = status["counts"]
    assert counts == {
        "pending": 0, "extracting": 0, "extracted": 0, "failed": 0, "total": 0,
    }

    # Add three fragments.
    for i in range(3):
        await asyncio.to_thread(
            _client_call, socket_path, "remember",
            {"text": f"fragment {i}", "source_id": "remember://test"},
        )

    status = await asyncio.to_thread(_client_call, socket_path, "spool_status")
    assert status["counts"]["pending"] == 3
    assert status["counts"]["total"] == 3
    assert status["recent_batches"] == []  # nothing extracted yet


@pytest.mark.asyncio
async def test_remember_does_not_require_llm(daemon_pair):
    """``remember`` must work even with no LLM configured — that's the
    whole point of decoupling capture from extraction."""
    daemon, socket_path = daemon_pair
    daemon._llm_client = None
    daemon._llm_init_attempted = True

    result = await asyncio.to_thread(
        _client_call, socket_path, "remember",
        {"text": "captured without LLM"},
    )
    assert "fragment_id" in result


# ---------------------------------------------------------------------------
# PR F — auto-trigger and flush_spool
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_remember_auto_triggers_batch_at_threshold(daemon_pair):
    """When ``pending_count`` reaches ``spool_size_threshold``, the next
    ``remember`` call must enqueue a ``spool_batch`` job. We lower the
    threshold to 3 to keep the test fast."""
    daemon, socket_path = daemon_pair
    daemon._config.spool_size_threshold = 3

    # Capture spool_batch enqueues by patching _run_ingest_job.
    seen_kinds: list[str] = []
    original = daemon._run_ingest_job

    def _spy(job: dict) -> dict:
        seen_kinds.append(job["kind"])
        return original(job)
    daemon._run_ingest_job = _spy  # type: ignore[assignment]

    # 3 remember calls should be enough to cross the threshold.
    for i in range(3):
        await asyncio.to_thread(
            _client_call, socket_path, "remember",
            {"text": f"fact {i}", "source_id": f"remember://test/{i}"},
        )

    # Wait briefly for the worker to pick up the auto-triggered batch.
    for _ in range(50):
        if "spool_batch" in seen_kinds:
            break
        await asyncio.sleep(0.02)

    assert "spool_batch" in seen_kinds, (
        f"expected spool_batch auto-trigger after threshold; saw kinds: {seen_kinds}"
    )


@pytest.mark.asyncio
async def test_remember_does_not_auto_trigger_below_threshold(daemon_pair):
    daemon, socket_path = daemon_pair
    daemon._config.spool_size_threshold = 100  # never reached in this test

    seen_kinds: list[str] = []
    original = daemon._run_ingest_job
    def _spy(job): seen_kinds.append(job["kind"]); return original(job)
    daemon._run_ingest_job = _spy  # type: ignore[assignment]

    for i in range(3):
        await asyncio.to_thread(
            _client_call, socket_path, "remember", {"text": f"fact {i}"},
        )
    await asyncio.sleep(0.1)
    assert "spool_batch" not in seen_kinds


@pytest.mark.asyncio
async def test_remember_threshold_zero_disables_auto_trigger(daemon_pair):
    """Setting threshold to 0 should disable auto-triggering entirely."""
    daemon, socket_path = daemon_pair
    daemon._config.spool_size_threshold = 0

    seen_kinds: list[str] = []
    original = daemon._run_ingest_job
    def _spy(job): seen_kinds.append(job["kind"]); return original(job)
    daemon._run_ingest_job = _spy  # type: ignore[assignment]

    for i in range(20):
        await asyncio.to_thread(
            _client_call, socket_path, "remember", {"text": f"fact {i}"},
        )
    await asyncio.sleep(0.1)
    assert "spool_batch" not in seen_kinds


@pytest.mark.asyncio
async def test_flush_spool_enqueues_batch_job(daemon_pair):
    daemon, socket_path = daemon_pair
    daemon._config.spool_size_threshold = 0  # disable auto so we control when batches fire

    # Capture jobs.
    seen: list[dict] = []
    original = daemon._run_ingest_job
    def _spy(job): seen.append(job); return original(job)
    daemon._run_ingest_job = _spy  # type: ignore[assignment]

    # Add a fragment, then explicitly flush.
    await asyncio.to_thread(
        _client_call, socket_path, "remember",
        {"text": "explicit flush target", "source_id": "remember://flush"},
    )
    result = await asyncio.to_thread(
        _client_call, socket_path, "flush_spool", {},
    )
    assert "job_id" in result

    for _ in range(50):
        if any(j["kind"] == "spool_batch" for j in seen):
            break
        await asyncio.sleep(0.02)
    assert any(j["kind"] == "spool_batch" for j in seen)


@pytest.mark.asyncio
async def test_flush_spool_with_source_filter_passes_through(daemon_pair):
    daemon, socket_path = daemon_pair
    daemon._config.spool_size_threshold = 0

    seen_payloads: list[dict] = []
    original = daemon._run_ingest_job
    def _spy(job):
        seen_payloads.append(job["payload"])
        return original(job)
    daemon._run_ingest_job = _spy  # type: ignore[assignment]

    await asyncio.to_thread(
        _client_call, socket_path, "flush_spool",
        {"source_filter": "session://specific", "limit": 25},
    )
    await asyncio.sleep(0.1)

    assert any(
        p.get("source_filter") == "session://specific" and p.get("limit") == 25
        for p in seen_payloads
    )


@pytest.mark.asyncio
async def test_flush_spool_rejects_bad_limit(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(
            _client_call, socket_path, "flush_spool", {"limit": 0},
        )
    assert ei.value.code == int(ErrorCode.INVALID_PARAMS)


# ---------------------------------------------------------------------------
# PR G — spool_retry_failed and spool_cleanup RPCs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spool_retry_failed_resets_failed_to_pending(daemon_pair):
    daemon, socket_path = daemon_pair
    daemon._ensure_spool()
    fid = daemon._spool.add("trial", source_id="src")
    # Move it directly to failed state.
    daemon._spool.claim_batch()
    daemon._spool.mark_failed([fid], error="prior failure")

    result = await asyncio.to_thread(_client_call, socket_path, "spool_retry_failed")
    assert result["reset"] == 1

    counts = daemon._spool.status_counts()
    assert counts["pending"] == 1
    assert counts["failed"] == 0


@pytest.mark.asyncio
async def test_spool_cleanup_rejects_nonpositive_retain_days(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(
            _client_call, socket_path, "spool_cleanup", {"retain_days": 0},
        )
    assert ei.value.code == int(ErrorCode.INVALID_PARAMS)


@pytest.mark.asyncio
async def test_spool_status_includes_failed_sample(daemon_pair):
    daemon, socket_path = daemon_pair
    daemon._ensure_spool()
    fid = daemon._spool.add("borked", source_id="src://x")
    daemon._spool.claim_batch()
    daemon._spool.mark_failed([fid], error="something went wrong")

    status = await asyncio.to_thread(_client_call, socket_path, "spool_status")
    sample = status["failed_sample"]
    assert len(sample) == 1
    assert sample[0]["error"] == "something went wrong"


@pytest.mark.asyncio
async def test_shutdown_cancels_worker_cleanly(tmp_path: Path):
    """Stopping the daemon while jobs are queued must not hang shutdown."""
    daemon = Daemon(graph_root=tmp_path / "g", socket_path=_short_socket_path())

    # Pipeline that takes long enough to still be running when we call stop().
    def _slow(job):
        import time
        time.sleep(0.3)
        return {"source_document": "x", "status": "complete",
                "chunks_total": 0, "chunks_tagged": 0, "chunks_filler": 0,
                "entities_created": 0, "entities_linked": 0, "edges_created": 0,
                "duration_seconds": 0.3, "errors": []}

    daemon._run_ingest_job = _slow  # type: ignore[assignment]
    daemon._llm_client = object()
    daemon._llm_init_attempted = True

    serve = asyncio.create_task(daemon.start())
    for _ in range(100):
        if daemon.socket_path.exists():
            break
        await asyncio.sleep(0.01)

    # Enqueue one job, then immediately shut down while it's processing.
    daemon.enqueue_ingest_job("session", {"path": "/tmp/x.jsonl"})
    await asyncio.sleep(0.05)  # let worker pick it up
    await asyncio.wait_for(daemon.stop(), timeout=2.0)
    serve.cancel()
    try:
        await serve
    except (asyncio.CancelledError, Exception):
        pass

    assert not daemon.socket_path.exists()
