"""Client for the graphited daemon socket.

Short, synchronous request/response. Every call is one line in, one line
out. A single ``GraphiteClient`` instance is NOT thread-safe — construct
one per thread/async task.

If ``graphited`` is not running, every call raises ``DaemonUnavailable``
with a clear message pointing the user at ``graphite daemon status``.
The client does not attempt to auto-start the daemon; that is launchd's
job (see PR 8).
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Optional

from graphite.protocol import ErrorCode, Request, Response

DEFAULT_SOCKET_PATH = Path.home() / ".graphite" / "daemon.sock"
DEFAULT_TIMEOUT_S = 60.0


class DaemonUnavailable(RuntimeError):
    """Raised when the daemon socket cannot be reached."""


class DaemonError(RuntimeError):
    """Raised when the daemon returns an error response."""

    def __init__(self, code: int, message: str, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class GraphiteClient:
    """Thin synchronous client over the Unix socket."""

    def __init__(
        self,
        socket_path: Path = DEFAULT_SOCKET_PATH,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.socket_path = Path(socket_path).expanduser()
        self.timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None
        self._rbuf = b""
        self._next_id = 1

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self._sock is not None:
            return
        if not self.socket_path.exists():
            raise DaemonUnavailable(
                f"Graphite daemon socket not found at {self.socket_path}. "
                f"Is `graphited` running? Try `graphite daemon status`."
            )
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout_s)
        try:
            s.connect(str(self.socket_path))
        except (ConnectionRefusedError, FileNotFoundError, PermissionError) as e:
            s.close()
            raise DaemonUnavailable(
                f"Cannot connect to graphite daemon at {self.socket_path}: {e}"
            )
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._rbuf = b""

    def __enter__(self) -> "GraphiteClient":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- low-level call ----------------------------------------------------

    def call(self, method: str, params: Optional[dict] = None) -> Any:
        """Invoke ``method`` with ``params``, return the decoded result.

        Raises ``DaemonUnavailable`` if the socket is gone, ``DaemonError`` if
        the daemon returns an error response.
        """
        self.connect()
        assert self._sock is not None

        req = Request(id=self._next_id, method=method, params=params or {})
        self._next_id += 1

        try:
            self._sock.sendall(req.to_line())
            line = self._read_line()
        except (BrokenPipeError, ConnectionResetError) as e:
            self.close()
            raise DaemonUnavailable(f"Daemon closed the connection: {e}")
        except socket.timeout:
            raise DaemonUnavailable(f"Daemon did not respond within {self.timeout_s}s")

        resp = Response.from_line(line)
        if resp.id != req.id:
            raise DaemonError(
                int(ErrorCode.INTERNAL_ERROR),
                f"Response id {resp.id} does not match request id {req.id}",
            )
        if resp.error is not None:
            raise DaemonError(
                code=int(resp.error.get("code", ErrorCode.INTERNAL_ERROR)),
                message=resp.error.get("message", "unknown error"),
                details=resp.error.get("details"),
            )
        return resp.result

    def _read_line(self) -> bytes:
        """Read bytes from the socket until a newline delimiter."""
        assert self._sock is not None
        while b"\n" not in self._rbuf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise BrokenPipeError("daemon closed the connection mid-response")
            self._rbuf += chunk
        line, _, rest = self._rbuf.partition(b"\n")
        self._rbuf = rest
        return line + b"\n"

    # -- high-level helpers (Phase 1 subset) -------------------------------

    def ping(self) -> dict:
        return self.call("ping")

    def status(self) -> dict:
        return self.call("status")

    def get_statistics(self) -> dict:
        return self.call("get_statistics")

    def search_entities(self, query: str, limit: int = 10) -> list:
        return self.call("search_entities", {"query": query, "limit": limit})

    def get_entity(self, entity_id: str) -> Optional[dict]:
        return self.call("get_entity", {"entity_id": entity_id})

    def query_neighborhood(
        self,
        entity_id: str,
        hops: int = 2,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
    ) -> dict:
        params: dict = {"entity_id": entity_id, "hops": hops}
        if time_start is not None:
            params["time_start"] = time_start
        if time_end is not None:
            params["time_end"] = time_end
        return self.call("query_neighborhood", params)

    def force_save(self) -> dict:
        return self.call("force_save")

    def enqueue_session_ingest(
        self,
        path: str,
        project: Optional[str] = None,
        force: bool = False,
    ) -> dict:
        """Queue a Claude Code session JSONL for background ingestion."""
        params: dict = {"path": path}
        if project is not None:
            params["project"] = project
        if force:
            params["force"] = True
        return self.call("enqueue_session_ingest", params)

    def ingest_queue_status(self) -> dict:
        """Return queue depth, in-flight job, and recent completions."""
        return self.call("ingest_queue_status")

    def reconcile_archive(self, archive_dir: Optional[str] = None) -> dict:
        """Replay any archived sessions that aren't yet in the graph.
        Returns a summary of what was scanned and enqueued."""
        params: dict = {}
        if archive_dir is not None:
            params["archive_dir"] = archive_dir
        return self.call("reconcile_archive", params)

    def reconcile_overflow(self, overflow_dir: Optional[str] = None) -> dict:
        """Replay overflow JSON files written by external capture agents
        (e.g. the OpenClaw plugin) when the daemon was unreachable."""
        params: dict = {}
        if overflow_dir is not None:
            params["overflow_dir"] = overflow_dir
        return self.call("reconcile_overflow", params)

    def remember(
        self,
        text: str,
        source_id: Optional[str] = None,
        category: str = "Episodic",
        project: Optional[str] = None,
        entity_hints: Optional[list] = None,
    ) -> dict:
        """Durable fast capture — writes a fragment to the spool and returns
        immediately. Extraction happens asynchronously in the daemon's
        batch worker."""
        params: dict = {"text": text, "category": category}
        if source_id is not None:
            params["source_id"] = source_id
        if project is not None:
            params["project"] = project
        if entity_hints is not None:
            params["entity_hints"] = list(entity_hints)
        return self.call("remember", params)

    def spool_status(self) -> dict:
        """Histogram of spooled fragments by state, plus recent batches."""
        return self.call("spool_status")

    def flush_spool(
        self,
        source_filter: Optional[str] = None,
        limit: int = 1000,
    ) -> dict:
        """Enqueue an immediate batch extraction over the spool. Returns
        the job metadata; the actual draining happens on the daemon's
        ingest worker."""
        params: dict = {"limit": limit}
        if source_filter is not None:
            params["source_filter"] = source_filter
        return self.call("flush_spool", params)

    def spool_retry_failed(self) -> dict:
        """Bounce all ``failed`` fragments back to ``pending``."""
        return self.call("spool_retry_failed")

    def spool_cleanup(self, retain_days: int = 30) -> dict:
        """Purge extracted fragments older than ``retain_days``."""
        return self.call("spool_cleanup", {"retain_days": retain_days})

    def kg_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a whitelisted PyKnowledgeGraph method on the daemon's copy.

        Return values come back verbatim — PyKnowledgeGraph methods that
        return JSON strings still return JSON strings here. The caller can
        ``json.loads`` on them as needed.
        """
        return self.call(
            "kg_call",
            {"method": method, "args": list(args), "kwargs": dict(kwargs)},
        )


class DaemonBackedGraph:
    """Drop-in replacement for ``PyKnowledgeGraph`` that routes every call
    through a ``GraphiteClient``.

    Exists so the MCP server, CLI, and agent can adopt the single-writer
    daemon without rewriting every tool handler. Attribute access returns
    a callable that forwards to the daemon via ``kg_call``. Unknown method
    names are rejected at the daemon protocol layer, so typos surface as
    ``DaemonError(INVALID_PARAMS)``.

    Phase 1 scope: intentionally NOT a full compatibility layer. It handles
    the subset of ``PyKnowledgeGraph`` methods listed in the daemon's
    ``_KG_METHODS`` whitelist. ``save`` is remapped to ``force_save`` on the
    daemon since the daemon owns persistence.
    """

    def __init__(self, client: GraphiteClient):
        self._client = client

    def __getattr__(self, name: str) -> Any:
        # save/load are special: persistence is the daemon's job.
        if name == "save":
            def _save_proxy(_path: Optional[str] = None) -> None:
                self._client.force_save()
            return _save_proxy
        if name in ("load", "from_path"):
            raise AttributeError(
                "DaemonBackedGraph does not support load/from_path — "
                "the daemon owns the graph file."
            )

        def _proxy(*args: Any, **kwargs: Any) -> Any:
            return self._client.kg_call(name, *args, **kwargs)

        return _proxy
