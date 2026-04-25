"""Graphited — single-writer daemon for the Graphite knowledge graph.

One long-lived process owns the ``PyKnowledgeGraph`` and the ``.graphite/``
directory on disk. Everything else (the MCP server spawned per Claude
session, the ``graphite`` CLI, ad-hoc clients) connects over a Unix
domain socket at ``~/.graphite/daemon.sock`` and issues RPCs.

Why: the old architecture had every process construct its own
``PyKnowledgeGraph`` and race to persist it. Single-writer kills that
class of bug and gives us one place to put background work (async
extraction, scheduled syncs, observation generation).

Phase 1 scope: the daemon answers a small set of low-level methods —
just enough for the MCP server and CLI to be refactored onto it in PR 6.
Higher-level tool semantics (``recall``, ``remember``, etc.) will graduate
to the daemon in later phases.

Usage:
    graphited                      # run in foreground (launchd will too)
    graphited --graph-root /tmp/x  # alternate storage (tests)
    graphited --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import os
import signal
import stat
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Optional

from graphite.protocol import (
    ErrorCode,
    ProtocolError,
    Request,
    Response,
    make_error,
)

log = logging.getLogger("graphite.daemon")

DEFAULT_SOCKET_PATH = Path.home() / ".graphite" / "daemon.sock"
DEFAULT_GRAPH_ROOT = Path.home()
DEFAULT_ARCHIVE_DIR = Path.home() / ".graphite" / "archive" / "sessions"
DEFAULT_OVERFLOW_DIR = Path.home() / ".graphite" / "spool_overflow"
SAVE_DEBOUNCE_SECONDS = 30.0
MAX_LINE_BYTES = 16 * 1024 * 1024  # 16 MB hard ceiling per request/response
RECENT_JOBS_MAX = 50                 # Ring buffer for job observability
SPOOL_CLEANUP_INTERVAL_SECONDS = 86400  # Once per day


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed. Used for archive idempotency
    so we never re-run the LLM pipeline on a session whose content hash
    already matches what the graph has stored."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _peek_session_header(jsonl_path: Path) -> tuple[Optional[str], str]:
    """Read just enough of a Claude Code session JSONL to derive
    ``(session_id, project_name)``. Returns ``(None, "unknown")`` if the
    file is unparseable or has no recoverable session metadata.

    Mirrors the logic in ``ConversationParser._extract_session_info`` but
    avoids loading the full transcript or instantiating the parser — this
    runs at daemon startup over the entire archive and must stay cheap.
    """
    session_id: Optional[str] = None
    project_name = "unknown"
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if session_id is None and obj.get("sessionId"):
                    session_id = obj["sessionId"]
                if project_name == "unknown" and obj.get("cwd"):
                    name = Path(obj["cwd"]).name
                    if name:
                        project_name = name
                if session_id and project_name != "unknown":
                    break
    except OSError:
        return None, "unknown"
    return session_id, project_name


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
# Handler signature: async def handler(daemon: Daemon, params: dict) -> Any
Handler = Callable[["Daemon", dict], Awaitable[Any]]
_HANDLERS: dict[str, Handler] = {}


def method(name: str) -> Callable[[Handler], Handler]:
    """Decorator: register ``func`` as the handler for ``name``."""

    def _wrap(func: Handler) -> Handler:
        if name in _HANDLERS:
            raise RuntimeError(f"Duplicate daemon method: {name}")
        _HANDLERS[name] = func
        return func

    return _wrap


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------
class Daemon:
    """Owns the graph, the socket, and the scheduler."""

    def __init__(self, graph_root: Path, socket_path: Path):
        self.graph_root = graph_root.expanduser().resolve()
        self.socket_path = socket_path.expanduser().resolve()
        self.started_at: float = 0.0

        # Lazy — constructed on first load to avoid import-time pyo3 cost.
        self._kg = None
        self._dirty = False
        self._last_save: float = 0.0
        self._save_task: Optional[asyncio.Task] = None
        self._server: Optional[asyncio.base_events.Server] = None
        # Serialize writes; reads of the PyKnowledgeGraph are also serialized
        # here for Phase 1 simplicity. Concurrent readers get added in PR 6.
        self._kg_lock = asyncio.Lock()
        self._stopping = False

        # Config + LLM + ingestion pipeline — lazy. The daemon can run read-only
        # without ever constructing any of these.
        self._config = None              # graphite.config.GraphiteConfig
        self._llm_client = None          # graphite.llm.<Provider>Client
        self._llm_init_attempted: bool = False  # Don't retry on every enqueue
        self._pipeline = None            # graphite.ingestion.pipeline.IngestionPipeline

        # Spool for fast remember() capture — lazy, opened on first use.
        self._spool = None               # graphite.spool.Spool
        self._spool_cleanup_task: Optional[asyncio.Task] = None

        # Ingest queue + worker. The worker drains the queue one job at a time
        # — local LLMs don't tolerate concurrent requests well, and bundling
        # session-scoped chunks into one pipeline run gives better canon.
        self._ingest_queue: asyncio.Queue = asyncio.Queue()
        self._ingest_worker_task: Optional[asyncio.Task] = None
        self._current_job: Optional[dict] = None
        self._recent_jobs: Deque[dict] = deque(maxlen=RECENT_JOBS_MAX)

    # -- config / LLM / pipeline -------------------------------------------

    def _ensure_config(self) -> None:
        if self._config is not None:
            return
        from graphite.config import GraphiteConfig

        toml_path = self.graph_root / ".graphite.toml"
        if toml_path.exists():
            try:
                self._config = GraphiteConfig.from_toml(toml_path)
                log.info("Loaded config from %s", toml_path)
            except Exception as e:
                log.warning("Failed to load %s: %s — using defaults", toml_path, e)
                self._config = GraphiteConfig()
        else:
            self._config = GraphiteConfig()

    def _ensure_llm_client(self) -> None:
        """Lazily construct an LLM client for the pipeline. Non-fatal on
        failure — we just flag LLM as unavailable; ingest jobs will fail
        loudly when they actually try to run."""
        if self._llm_client is not None or self._llm_init_attempted:
            return
        self._llm_init_attempted = True
        self._ensure_config()

        provider = self._config.llm_provider
        model = self._config.llm_model
        try:
            if provider == "ollama":
                from graphite.llm import OllamaClient
                self._llm_client = OllamaClient(model=model)
            elif provider == "mlx":
                from graphite.llm import MLXClient
                self._llm_client = MLXClient(model=model)
            elif provider == "openai":
                from graphite.llm import OpenAIClient
                self._llm_client = OpenAIClient(model=model)
            elif provider == "anthropic":
                from graphite.llm import AnthropicClient
                self._llm_client = AnthropicClient(model=model)
            else:
                log.warning("Unknown LLM provider in config: %s", provider)
                return
            log.info("LLM client ready: %s / %s", provider, model)
        except Exception as e:
            log.warning("Failed to construct %s LLM client: %s", provider, e)
            self._llm_client = None

    def _ensure_spool(self) -> None:
        """Open the SQLite spool at ``<graph_root>/.graphite/spool.db``.
        Resets any rows stuck in ``extracting`` from a prior crash."""
        if self._spool is not None:
            return
        from graphite.spool import Spool
        spool_path = self.graph_root / ".graphite" / "spool.db"
        self._spool = Spool(spool_path)
        bounced = self._spool.reset_stale_extracting()
        if bounced > 0:
            log.info("Spool: bounced %d stale 'extracting' fragments back to pending", bounced)

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        self._ensure_graph()
        self._ensure_config()
        self._ensure_llm_client()
        if self._llm_client is None:
            raise RuntimeError(
                f"No LLM client available (provider={self._config.llm_provider}, "
                f"model={self._config.llm_model}). Configure ~/.graphite.toml "
                f"or ensure the provider is reachable."
            )

        from graphite.ingestion.pipeline import IngestionPipeline
        self._pipeline = IngestionPipeline(
            knowledge_graph=self._kg,
            llm_client=self._llm_client,
            config=self._config,
        )

    # -- graph lifecycle ---------------------------------------------------

    def _ensure_graph(self) -> None:
        if self._kg is not None:
            return
        from graphite.semantic_engine import PyKnowledgeGraph

        graph_dir = self.graph_root / ".graphite"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_file = graph_dir / "graph.msgpack"
        if graph_file.exists():
            log.info("Loading graph from %s", graph_file)
            self._kg = PyKnowledgeGraph.from_path(str(self.graph_root))
        else:
            log.info("No persisted graph found — starting empty at %s", graph_file)
            self._kg = PyKnowledgeGraph(str(self.graph_root))

    def mark_dirty(self) -> None:
        self._dirty = True

    def _save_now(self) -> None:
        if not self._dirty or self._kg is None:
            return
        try:
            self._kg.save(str(self.graph_root))
            self._dirty = False
            self._last_save = time.time()
            log.info("Graph saved")
        except Exception as e:
            log.error("Save failed: %s", e)

    async def _schedule_save(self) -> None:
        """Debounced async save — fires once the graph has been quiet for the
        configured interval. Rescheduled whenever a write happens."""
        if self._save_task is not None and not self._save_task.done():
            self._save_task.cancel()

        async def _delayed() -> None:
            try:
                await asyncio.sleep(SAVE_DEBOUNCE_SECONDS)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_now)
            except asyncio.CancelledError:
                pass

        self._save_task = asyncio.create_task(_delayed())

    # -- ingest queue + worker ---------------------------------------------

    def enqueue_ingest_job(self, kind: str, payload: dict) -> dict:
        """Append a job to the ingest queue. Returns queue metadata."""
        job_id = uuid.uuid4().hex[:12]
        job: dict = {
            "id": job_id,
            "kind": kind,          # "session" | "document" (future: "source")
            "payload": payload,
            "status": "queued",
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        self._ingest_queue.put_nowait(job)
        return {"job_id": job_id, "queue_position": self._ingest_queue.qsize()}

    async def _ingest_worker_loop(self) -> None:
        """Drain the ingest queue serially. Runs as a long-lived task on the
        daemon's event loop; the blocking pipeline work is offloaded to an
        executor so the socket server stays responsive."""
        log.info("Ingest worker started")
        loop = asyncio.get_event_loop()
        while not self._stopping:
            try:
                job = await self._ingest_queue.get()
            except asyncio.CancelledError:
                log.info("Ingest worker cancelled")
                return

            self._current_job = job
            job["status"] = "running"
            job["started_at"] = time.time()
            log.info("Ingest job %s starting (kind=%s)", job["id"], job["kind"])

            try:
                result = await loop.run_in_executor(None, self._run_ingest_job, job)
                job["status"] = "complete"
                job["result"] = result
                self.mark_dirty()
                await self._schedule_save()
                log.info(
                    "Ingest job %s complete: %d entities, %d edges (%.1fs)",
                    job["id"],
                    result.get("entities_created", 0) + result.get("entities_linked", 0),
                    result.get("edges_created", 0),
                    result.get("duration_seconds", 0.0),
                )
            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
                log.exception("Ingest job %s failed", job["id"])
            finally:
                job["finished_at"] = time.time()
                self._current_job = None
                self._recent_jobs.append(job)
                self._ingest_queue.task_done()

    def _run_ingest_job(self, job: dict) -> dict:
        """Synchronous body of one ingest job — runs on an executor thread."""
        kind = job["kind"]
        payload = job["payload"]
        if kind == "session":
            return self._run_session_ingest(payload)
        if kind == "spool_batch":
            return self._run_spool_batch(payload)
        raise RuntimeError(f"Unknown ingest job kind: {kind}")

    def _run_spool_batch(self, payload: dict) -> dict:
        """Drain the spool through the ingestion pipeline."""
        from graphite.spool_extractor import BatchExtractor

        self._ensure_spool()
        # Lazy pipeline creation — defer LLM client construction so a
        # spool_batch job with nothing pending doesn't pay the cost.
        def _factory():
            self._ensure_pipeline()
            return self._pipeline

        extractor = BatchExtractor(self._spool, _factory)
        return extractor.extract_batch(
            batch_size_limit=int(payload.get("limit", 50)),
            source_filter=payload.get("source_filter"),
        )

    def _run_session_ingest(self, payload: dict) -> dict:
        """Run IngestionPipeline on one JSONL session file."""
        path_str = payload.get("path")
        if not path_str:
            raise RuntimeError("session ingest job missing 'path'")
        path = Path(path_str).expanduser()
        if not path.exists():
            raise RuntimeError(f"session file not found: {path}")

        self._ensure_pipeline()
        result = self._pipeline.ingest_session(path)
        # TODO(phase3): when multi-source ingestion lands, tag all new
        # entities/chunks with payload["project"] so cross-project queries
        # work. For now the project tag is ignored.
        return {
            "source_document": result.source_document,
            "status": result.status,
            "chunks_total": result.chunks_total,
            "chunks_tagged": result.chunks_tagged,
            "chunks_filler": getattr(result, "chunks_filler", 0),
            "entities_created": result.entities_created,
            "entities_linked": result.entities_linked,
            "edges_created": result.edges_created,
            "duration_seconds": result.duration_seconds,
            "errors": list(result.errors),
        }

    async def _spool_cleanup_loop(self) -> None:
        """Daily-ish cleanup task. Runs ``Spool.cleanup_old`` then sleeps
        ``SPOOL_CLEANUP_INTERVAL_SECONDS``. The first cleanup runs after
        the first interval, not at boot — boot already has plenty to do."""
        while not self._stopping:
            try:
                await asyncio.sleep(SPOOL_CLEANUP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                return
            try:
                self._ensure_spool()
                retain = getattr(self._config, "spool_retain_days", 30)
                removed = self._spool.cleanup_old(retain_days=retain)
                if removed > 0:
                    log.info(
                        "Spool cleanup: removed %d extracted fragments older than %d days",
                        removed, retain,
                    )
            except Exception:
                log.exception("Spool cleanup raised")

    # -- backlog reconciliation --------------------------------------------

    async def _reconcile_async(self) -> None:
        """Async wrapper that runs ``reconcile_archive`` on a worker thread
        so a long scan never blocks the event loop. Errors are logged but
        never raised — reconciliation is best-effort and the live hook
        path will continue catching new sessions regardless."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.reconcile_archive, None)
        except Exception:
            log.exception("Reconcile failed")

    async def _reconcile_overflow_async(self) -> None:
        """Drain the overflow directory written by external capture
        agents that couldn't reach the live socket. Same fire-and-forget
        contract as the session reconciler."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.reconcile_overflow, None)
        except Exception:
            log.exception("Overflow reconcile failed")

    def reconcile_overflow(self, overflow_dir: Optional[Path] = None) -> dict:
        """Replay all overflow JSON files via the spool. Idempotent — files
        whose content hash matches an already-ingested document are
        skipped without touching the spool."""
        from graphite.overflow_reconciler import reconcile_overflow

        target = overflow_dir if overflow_dir is not None else DEFAULT_OVERFLOW_DIR
        self._ensure_spool()
        self._ensure_graph()
        return reconcile_overflow(target, spool=self._spool, kg=self._kg)

    def reconcile_archive(self, archive_dir: Optional[Path] = None) -> dict:
        """Scan the session archive and enqueue any sessions missing from the
        graph. Idempotent — sessions whose content hash matches the stored
        graph hash are skipped without re-running the LLM pipeline.

        Synchronous; safe to call from any context. Returns a summary dict
        suitable for the ``reconcile_archive`` RPC and for daemon startup
        logging.
        """
        archive = archive_dir if archive_dir is not None else DEFAULT_ARCHIVE_DIR
        summary = {
            "archive_dir": str(archive),
            "scanned": 0,
            "enqueued": 0,
            "already_indexed": 0,
            "skipped_unparseable": 0,
            "skipped_no_llm": 0,
        }

        if not archive.is_dir():
            return summary

        # If the LLM isn't ready, queueing thousands of jobs that will all
        # fail is worse than waiting for the user to fix their config.
        self._ensure_llm_client()
        no_llm = self._llm_client is None

        self._ensure_graph()

        for jsonl_path in sorted(archive.glob("*.jsonl")):
            summary["scanned"] += 1

            session_id, project_name = _peek_session_header(jsonl_path)
            if session_id is None:
                log.warning("Reconciler: cannot derive session metadata from %s — skipping", jsonl_path)
                summary["skipped_unparseable"] += 1
                continue

            source_id = f"claude-session://{project_name}/{session_id}"

            try:
                existing_hash = self._kg.get_document_hash(source_id)
            except Exception:
                existing_hash = None

            current_hash = _file_sha256(jsonl_path)
            if existing_hash and existing_hash == current_hash:
                summary["already_indexed"] += 1
                continue

            if no_llm:
                summary["skipped_no_llm"] += 1
                continue

            self.enqueue_ingest_job(
                "session",
                {"path": str(jsonl_path), "project": project_name, "via": "reconcile"},
            )
            summary["enqueued"] += 1

        log.info(
            "Reconcile: scanned=%d enqueued=%d already=%d unparseable=%d no_llm=%d",
            summary["scanned"], summary["enqueued"], summary["already_indexed"],
            summary["skipped_unparseable"], summary["skipped_no_llm"],
        )
        return summary

    # -- socket server -----------------------------------------------------

    async def start(self) -> None:
        """Bind the socket, start accepting connections, and block forever."""
        self.started_at = time.time()
        self._ensure_graph()
        self._ensure_config()
        # Eagerly open the spool so its stale-extracting reset runs once
        # at boot rather than waiting for the first remember() call.
        self._ensure_spool()

        # Kick off the ingest worker before binding the socket so enqueues
        # from the very first connection have somewhere to go.
        self._ingest_worker_task = asyncio.create_task(self._ingest_worker_loop())

        # Backlog reconciler — replays any archived session not present in
        # the graph. Fire-and-forget so socket bind isn't gated on archive
        # I/O, which can be slow on a multi-month backlog.
        if getattr(self._config, "reconcile_on_startup", True):
            asyncio.create_task(self._reconcile_async())
            # Same trigger covers the overflow files written by external
            # capture agents (e.g. the OpenClaw plugin) when the daemon
            # was unreachable at the moment of capture.
            asyncio.create_task(self._reconcile_overflow_async())

        # Periodic spool cleanup — purges extracted-and-aged fragments so
        # the audit trail doesn't grow unbounded.
        self._spool_cleanup_task = asyncio.create_task(self._spool_cleanup_loop())

        # Ensure parent dir exists with 0700; the socket will be 0600.
        parent = self.socket_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(parent, stat.S_IRWXU)  # 0700
        except OSError:
            pass  # non-fatal; socket perms are the real boundary

        # Stale socket from a crashed daemon blocks bind; remove it first.
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as e:
                raise RuntimeError(f"Failed to remove stale socket {self.socket_path}: {e}")

        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self.socket_path),
        )
        # Restrict socket to the owning user.
        os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        log.info("Listening on %s (graph root: %s)", self.socket_path, self.graph_root)

        async with self._server:
            try:
                await self._server.serve_forever()
            except asyncio.CancelledError:
                pass

    async def stop(self) -> None:
        """Graceful shutdown — flush saves, close connections, remove socket.

        In-flight ingest jobs are cancelled; queued-but-not-started jobs
        are dropped. Anything lost on shutdown is safely recoverable
        because session JSONLs are archived on disk before enqueue — the
        backlog reconciler (PR D) replays missing sessions on next start.
        """
        if self._stopping:
            return
        self._stopping = True
        log.info("Stopping daemon...")

        if self._save_task is not None and not self._save_task.done():
            self._save_task.cancel()

        if self._ingest_worker_task is not None and not self._ingest_worker_task.done():
            self._ingest_worker_task.cancel()
            try:
                await self._ingest_worker_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._spool_cleanup_task is not None and not self._spool_cleanup_task.done():
            self._spool_cleanup_task.cancel()
            try:
                await self._spool_cleanup_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._dirty and self._kg is not None:
            self._save_now()

        if self._spool is not None:
            try:
                self._spool.close()
            except Exception:
                pass
            self._spool = None

        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass

        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        log.info("Daemon stopped.")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "<unknown>"
        log.debug("Client connected: %s", peer)
        try:
            while True:
                line = await reader.readuntil(b"\n")
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    resp = Response(
                        id=0,
                        error=make_error(
                            ErrorCode.INVALID_REQUEST,
                            f"Request exceeds {MAX_LINE_BYTES} bytes",
                        ),
                    )
                    writer.write(resp.to_line())
                    await writer.drain()
                    continue
                response = await self._dispatch(line)
                writer.write(response.to_line())
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass  # client closed cleanly
        except ConnectionResetError:
            pass
        except Exception as e:
            log.warning("Connection error from %s: %s", peer, e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            log.debug("Client disconnected: %s", peer)

    async def _dispatch(self, raw: bytes) -> Response:
        """Parse one request line, route to a handler, return a Response."""
        try:
            req = Request.from_line(raw)
        except ProtocolError as pe:
            return Response(id=0, error=make_error(pe.code, pe.message, pe.details))

        handler = _HANDLERS.get(req.method)
        if handler is None:
            return Response(
                id=req.id,
                error=make_error(ErrorCode.METHOD_NOT_FOUND, f"Unknown method: {req.method}"),
            )

        try:
            result = await handler(self, req.params)
            return Response(id=req.id, result=result)
        except ProtocolError as pe:
            return Response(id=req.id, error=make_error(pe.code, pe.message, pe.details))
        except Exception as e:
            log.exception("Handler %s raised", req.method)
            return Response(
                id=req.id,
                error=make_error(ErrorCode.INTERNAL_ERROR, str(e)),
            )


# ---------------------------------------------------------------------------
# Handlers — Phase 1 minimum set
# ---------------------------------------------------------------------------
# Higher-level tools (recall, remember, reflect, ingest_source, register_source)
# graduate to the daemon in later PRs. Phase 1 exposes just enough for the MCP
# server and CLI to move onto the daemon in PR 6.


@method("ping")
async def _ping(daemon: Daemon, params: dict) -> dict:
    return {"ok": True, "uptime_s": time.time() - daemon.started_at}


@method("status")
async def _status(daemon: Daemon, params: dict) -> dict:
    async with daemon._kg_lock:
        daemon._ensure_graph()
        stats = json.loads(daemon._kg.get_statistics())
    graph_file = daemon.graph_root / ".graphite" / "graph.msgpack"
    return {
        "graph_root": str(daemon.graph_root),
        "socket_path": str(daemon.socket_path),
        "pid": os.getpid(),
        "uptime_s": time.time() - daemon.started_at,
        "graph_persisted": graph_file.exists(),
        "last_save_s_ago": (time.time() - daemon._last_save) if daemon._last_save > 0 else None,
        "dirty": daemon._dirty,
        "stats": stats,
    }


@method("get_statistics")
async def _get_statistics(daemon: Daemon, params: dict) -> Any:
    async with daemon._kg_lock:
        daemon._ensure_graph()
        return json.loads(daemon._kg.get_statistics())


@method("search_entities")
async def _search_entities(daemon: Daemon, params: dict) -> Any:
    query = params.get("query", "")
    limit = int(params.get("limit", 10))
    async with daemon._kg_lock:
        daemon._ensure_graph()
        return json.loads(daemon._kg.search_entities(query, limit))


@method("get_entity")
async def _get_entity(daemon: Daemon, params: dict) -> Any:
    entity_id = params.get("entity_id")
    if not isinstance(entity_id, str):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "entity_id (str) required")
    async with daemon._kg_lock:
        daemon._ensure_graph()
        raw = daemon._kg.get_entity(entity_id)
    return json.loads(raw) if raw else None


@method("query_neighborhood")
async def _query_neighborhood(daemon: Daemon, params: dict) -> Any:
    entity_id = params.get("entity_id")
    hops = int(params.get("hops", 2))
    time_start = params.get("time_start")
    time_end = params.get("time_end")
    if not isinstance(entity_id, str):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "entity_id (str) required")
    async with daemon._kg_lock:
        daemon._ensure_graph()
        raw = daemon._kg.query_neighborhood(entity_id, hops, time_start, time_end)
    return json.loads(raw)


@method("force_save")
async def _force_save(daemon: Daemon, params: dict) -> dict:
    async with daemon._kg_lock:
        daemon._save_now()
    return {"saved": not daemon._dirty, "last_save_s_ago": time.time() - daemon._last_save}


@method("enqueue_session_ingest")
async def _enqueue_session_ingest(daemon: Daemon, params: dict) -> dict:
    """Queue a Claude Code session JSONL for ingestion. Returns job_id
    immediately; the worker runs extraction in the background."""
    path = params.get("path")
    if not isinstance(path, str) or not path:
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "path (str) required")
    project = params.get("project")
    if project is not None and not isinstance(project, str):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "project must be a string if set")

    # Fail fast if the LLM isn't configured so callers can decide whether
    # to bother. We still enqueue if the user explicitly opts in via
    # ``force=True`` (useful for reconciler runs where partial retry is OK).
    daemon._ensure_llm_client()
    if daemon._llm_client is None and not params.get("force"):
        raise ProtocolError(
            ErrorCode.NOT_READY,
            f"No LLM client configured (provider={daemon._config.llm_provider if daemon._config else 'unknown'}). "
            f"Ingest will fail; set force=true to enqueue anyway.",
        )

    return daemon.enqueue_ingest_job("session", {"path": path, "project": project})


@method("remember")
async def _remember(daemon: Daemon, params: dict) -> dict:
    """Durable fast capture. Writes one row into the SQLite spool and
    returns immediately; the batch extractor (PR F) handles the LLM-heavy
    extraction asynchronously.

    Replaces the synchronous ``remember`` stub from PR 5.
    """
    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "text (non-empty str) required")

    source_id = params.get("source_id")
    if source_id is not None and not isinstance(source_id, str):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "source_id must be a string if set")
    if not source_id:
        # Synthesize a stable-ish default — PID + uuid keeps us out of
        # collision territory if the caller forgot to supply one. Caller
        # is encouraged to pass a real URI for dedup later.
        source_id = f"remember://{os.getpid()}/{uuid.uuid4().hex[:12]}"

    category = params.get("category", "Episodic")
    if category not in ("Episodic", "Semantic", "Procedural"):
        raise ProtocolError(
            ErrorCode.INVALID_PARAMS,
            f"category must be Episodic|Semantic|Procedural, got {category!r}",
        )

    project = params.get("project")
    if project is not None and not isinstance(project, str):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "project must be a string if set")

    entity_hints = params.get("entity_hints")
    if entity_hints is not None:
        if not isinstance(entity_hints, list) or not all(isinstance(h, str) for h in entity_hints):
            raise ProtocolError(
                ErrorCode.INVALID_PARAMS,
                "entity_hints must be an array of strings",
            )

    daemon._ensure_spool()
    fragment_id = daemon._spool.add(
        text=text,
        source_id=source_id,
        category=category,
        project=project,
        entity_hints=entity_hints,
    )
    pending = daemon._spool.pending_count()

    # Auto-trigger: when the pending count crosses the configured threshold,
    # enqueue a batch extraction job. The worker drains it serially with
    # other ingest jobs, so we don't overload a local LLM.
    threshold = getattr(daemon._config, "spool_size_threshold", 50)
    if threshold > 0 and pending >= threshold:
        try:
            daemon.enqueue_ingest_job(
                "spool_batch",
                {"limit": max(threshold * 2, 100)},
            )
        except Exception as e:
            log.warning("Auto-trigger of spool_batch failed (continuing): %s", e)

    return {
        "fragment_id": fragment_id,
        "source_id": source_id,
        "pending_count": pending,
    }


@method("flush_spool")
async def _flush_spool(daemon: Daemon, params: dict) -> dict:
    """On-demand drain of pending fragments. Used by the SessionEnd hook
    (PR G) to flush a session's `remember()` calls right after capture,
    and by `graphite spool flush` for manual triggering."""
    daemon._ensure_spool()
    source_filter = params.get("source_filter")
    if source_filter is not None and not isinstance(source_filter, str):
        raise ProtocolError(
            ErrorCode.INVALID_PARAMS,
            "source_filter must be a string if set",
        )

    # Drain everything pending in one batch by default. Caller can pass a
    # smaller limit if they want to bound the work per job.
    limit = int(params.get("limit", 1000))
    if limit <= 0:
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "limit must be positive")

    payload: dict = {"limit": limit}
    if source_filter is not None:
        payload["source_filter"] = source_filter
    return daemon.enqueue_ingest_job("spool_batch", payload)


@method("spool_status")
async def _spool_status(daemon: Daemon, params: dict) -> dict:
    """Histogram of fragments by state + recent batches. Cheap; safe to
    poll from status pages."""
    daemon._ensure_spool()
    counts = daemon._spool.status_counts()
    return {
        "counts": counts,
        "recent_batches": daemon._spool.recent_batches(limit=10),
        "failed_sample": daemon._spool.get_failed(limit=5),
    }


@method("spool_retry_failed")
async def _spool_retry_failed(daemon: Daemon, params: dict) -> dict:
    """Move all failed fragments back to ``pending`` so they get re-tried
    on the next batch. Triage tool exposed via ``graphite spool retry-failed``."""
    daemon._ensure_spool()
    count = daemon._spool.retry_failed()
    return {"reset": count}


@method("spool_cleanup")
async def _spool_cleanup(daemon: Daemon, params: dict) -> dict:
    """On-demand purge of old extracted fragments."""
    daemon._ensure_spool()
    retain = int(params.get("retain_days", 30))
    if retain <= 0:
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "retain_days must be positive")
    removed = daemon._spool.cleanup_old(retain_days=retain)
    return {"removed": removed, "retain_days": retain}


@method("reconcile_archive")
async def _reconcile_archive_handler(daemon: Daemon, params: dict) -> dict:
    """Trigger a backlog scan on demand. Same as the startup reconcile, but
    callable from the CLI / MCP server. Optional ``archive_dir`` param for
    pointing at a non-default location (used by tests)."""
    archive_dir_str = params.get("archive_dir")
    archive_dir = Path(archive_dir_str).expanduser() if archive_dir_str else None
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, daemon.reconcile_archive, archive_dir)


@method("reconcile_overflow")
async def _reconcile_overflow_handler(daemon: Daemon, params: dict) -> dict:
    """Replay overflow JSON files written by external capture agents
    when the live socket was unreachable. Idempotent."""
    overflow_dir_str = params.get("overflow_dir")
    overflow_dir = Path(overflow_dir_str).expanduser() if overflow_dir_str else None
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, daemon.reconcile_overflow, overflow_dir)


@method("ingest_queue_status")
async def _ingest_queue_status(daemon: Daemon, params: dict) -> dict:
    """Report queue depth, current job, and a ring buffer of recent jobs."""
    daemon._ensure_config()
    return {
        "depth": daemon._ingest_queue.qsize(),
        "current": daemon._current_job,
        "recent": list(daemon._recent_jobs),
        "llm_configured": daemon._llm_client is not None,
        "llm_provider": daemon._config.llm_provider if daemon._config else None,
        "llm_model": daemon._config.llm_model if daemon._config else None,
    }


# Methods on PyKnowledgeGraph that mutate state. Any kg_call invocation of
# these sets the dirty flag and schedules a debounced save.
_MUTATING_METHODS: frozenset[str] = frozenset({
    "add_entity",
    "add_cooccurrence",
    "store_chunk",
    "merge_entities",
    "remove_entity",
    "remove_document",
    "set_document_hash",
    "remove_document_hash",
    "compute_pagerank",          # updates the `rank` field on entities
    "decay_scores",
    "recalculate_edge_weights",
    "deduplicate_edges",
    "prune_edges_below_weight",
})

# Whitelist of PyKnowledgeGraph methods the daemon will proxy. Anything not
# listed here is rejected at the protocol layer to contain blast radius.
_KG_METHODS: frozenset[str] = frozenset({
    # Reads
    "get_entity",
    "get_chunk",
    "get_cooccurrences",
    "get_chunks_for_entities",
    "get_temporal_chain",
    "get_chunks_by_document",
    "get_chunks_by_time_window",
    "get_entities_by_project",
    "get_document_hash",
    "tracked_documents",
    "query_neighborhood",
    "search_entities",
    "get_statistics",
    "all_entity_ids",
    "find_orphan_entities",
    "get_top_entities",
    "export_subgraph",
    # Writes
    *_MUTATING_METHODS,
})


@method("kg_call")
async def _kg_call(daemon: Daemon, params: dict) -> Any:
    """Generic passthrough: invoke a whitelisted method on the underlying
    PyKnowledgeGraph. ``params = {"method": str, "args": [...], "kwargs": {...}}``.

    Every return value goes back verbatim — PyKnowledgeGraph returns JSON
    strings for structured results, which the client can ``json.loads`` as
    needed. We do not eagerly deserialize on the daemon side so the protocol
    stays dumb.
    """
    m = params.get("method")
    if not isinstance(m, str):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "kg_call requires method (str)")
    if m not in _KG_METHODS:
        raise ProtocolError(
            ErrorCode.INVALID_PARAMS,
            f"Method '{m}' is not exposed via kg_call",
        )

    args = params.get("args", []) or []
    kwargs = params.get("kwargs", {}) or {}
    if not isinstance(args, list):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "args must be a list")
    if not isinstance(kwargs, dict):
        raise ProtocolError(ErrorCode.INVALID_PARAMS, "kwargs must be an object")

    async with daemon._kg_lock:
        daemon._ensure_graph()
        fn = getattr(daemon._kg, m, None)
        if fn is None:
            raise ProtocolError(
                ErrorCode.INTERNAL_ERROR,
                f"Method '{m}' not found on PyKnowledgeGraph — .so out of date?",
            )
        try:
            result = fn(*args, **kwargs)
        except TypeError as e:
            raise ProtocolError(ErrorCode.INVALID_PARAMS, f"Bad args for {m}: {e}")

        if m in _MUTATING_METHODS:
            daemon.mark_dirty()

    if m in _MUTATING_METHODS:
        await daemon._schedule_save()

    return result


# Stubs — declared here so METHOD_NOT_FOUND doesn't fire for the Phase 3/4
# surface. They return NOT_IMPLEMENTED until the real handlers land.


@method("recall")
async def _recall_stub(daemon: Daemon, params: dict) -> Any:
    raise ProtocolError(ErrorCode.NOT_IMPLEMENTED, "recall ships in PR 7")


@method("ingest_source")
async def _ingest_source_stub(daemon: Daemon, params: dict) -> Any:
    raise ProtocolError(ErrorCode.NOT_IMPLEMENTED, "ingest_source ships in Phase 3")


@method("register_source")
async def _register_source_stub(daemon: Daemon, params: dict) -> Any:
    raise ProtocolError(ErrorCode.NOT_IMPLEMENTED, "register_source ships in Phase 3")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _run(daemon: Daemon) -> int:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows; not a supported platform anyway

    server_task = asyncio.create_task(daemon.start())
    try:
        await stop_event.wait()
    finally:
        await daemon.stop()
        server_task.cancel()
        try:
            await server_task
        except (asyncio.CancelledError, Exception):
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="graphited",
        description="Graphite daemon — single-writer owner of the knowledge graph.",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=DEFAULT_GRAPH_ROOT,
        help="Directory holding .graphite/graph.msgpack (default: ~)",
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        default=DEFAULT_SOCKET_PATH,
        help="Unix socket path (default: ~/.graphite/daemon.sock)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    daemon = Daemon(graph_root=args.graph_root, socket_path=args.socket_path)

    # atexit guard for the ungraceful-shutdown case (e.g., uncaught exception)
    atexit.register(lambda: daemon._save_now())

    try:
        return asyncio.run(_run(daemon))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
