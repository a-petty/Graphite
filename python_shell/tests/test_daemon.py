"""End-to-end tests for graphited + GraphiteClient over a real Unix socket.

These tests spawn a fresh daemon in an asyncio task (not a subprocess —
subprocess+maturin startup is too slow for the test loop). The important
invariants are the same either way:

  * Socket binds at the requested path with 0600 permissions.
  * Parent directory is created with 0700.
  * A stale socket from a prior crash is cleaned up before binding.
  * Round-trip calls parse, dispatch, and respond correctly.
  * SIGTERM-style shutdown flushes a dirty graph and removes the socket.
  * Phase-3/4 stubs return NOT_IMPLEMENTED rather than METHOD_NOT_FOUND.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import stat
import tempfile
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from graphite.client import DaemonError, DaemonUnavailable, GraphiteClient
from graphite.daemon import Daemon
from graphite.protocol import ErrorCode


def _short_socket_path() -> Path:
    """Return a short Unix socket path under /tmp.

    macOS caps ``sun_path`` at ~104 characters; pytest's ``tmp_path`` under
    ``/private/var/folders/...`` blows past that limit. We keep the graph
    data in ``tmp_path`` for cleanup, but park the socket under /tmp with a
    random short suffix.
    """
    return Path(tempfile.gettempdir()) / f"gph-{secrets.token_hex(4)}.sock"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def daemon_pair(tmp_path: Path) -> AsyncIterator[tuple[Daemon, Path]]:
    """Spin up a Daemon in the current event loop; yield (daemon, socket_path)."""
    graph_root = tmp_path / "graph_home"
    socket_path = _short_socket_path()
    daemon = Daemon(graph_root=graph_root, socket_path=socket_path)

    serve_task = asyncio.create_task(daemon.start())

    # Poll for the socket to appear before letting tests proceed.
    for _ in range(100):
        if socket_path.exists():
            break
        if serve_task.done():
            exc = serve_task.exception()
            raise RuntimeError(f"daemon task exited early: {exc!r}")
        await asyncio.sleep(0.01)
    else:
        serve_task.cancel()
        raise RuntimeError(
            f"daemon did not bind socket in time; task done={serve_task.done()}"
        )

    try:
        yield daemon, socket_path
    finally:
        await daemon.stop()
        serve_task.cancel()
        try:
            await serve_task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Socket hygiene
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_socket_has_0600_perms_and_parent_exists(daemon_pair):
    _daemon, socket_path = daemon_pair
    assert socket_path.exists()

    mode = socket_path.stat().st_mode & 0o777
    assert mode == 0o600, f"socket should be 0600, got {oct(mode)}"

    parent_mode = socket_path.parent.stat().st_mode & 0o777
    # 0700 is the intent; be permissive about umask-induced variation but
    # confirm group/other bits are not readable.
    assert parent_mode & stat.S_IRWXG == 0
    assert parent_mode & stat.S_IRWXO == 0


@pytest.mark.asyncio
async def test_stale_socket_is_cleaned_up(tmp_path: Path):
    # Plant a stale file at the socket path as if a prior daemon had crashed.
    socket_path = _short_socket_path()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_bytes(b"stale")
    assert socket_path.exists()

    daemon = Daemon(graph_root=tmp_path / "g", socket_path=socket_path)
    serve = asyncio.create_task(daemon.start())
    for _ in range(100):
        if socket_path.exists() and socket_path.is_socket():
            break
        await asyncio.sleep(0.01)
    try:
        assert socket_path.is_socket(), "daemon should replace the stale file with a real socket"
    finally:
        await daemon.stop()
        serve.cancel()
        try:
            await serve
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
def _client_call(socket_path: Path, method: str, params=None):
    """Helper: run a blocking client call on a worker thread from asyncio."""
    with GraphiteClient(socket_path=socket_path) as c:
        return c.call(method, params or {})


@pytest.mark.asyncio
async def test_ping(daemon_pair):
    _, socket_path = daemon_pair
    result = await asyncio.to_thread(_client_call, socket_path, "ping")
    assert result["ok"] is True
    assert "uptime_s" in result


@pytest.mark.asyncio
async def test_status_returns_graph_stats(daemon_pair):
    _, socket_path = daemon_pair
    result = await asyncio.to_thread(_client_call, socket_path, "status")
    assert "graph_root" in result
    assert "stats" in result
    assert isinstance(result["stats"], dict)
    assert result["graph_persisted"] is False  # fresh daemon, nothing saved yet


@pytest.mark.asyncio
async def test_get_statistics(daemon_pair):
    _, socket_path = daemon_pair
    stats = await asyncio.to_thread(_client_call, socket_path, "get_statistics")
    assert "entity_count" in stats
    assert stats["entity_count"] == 0


@pytest.mark.asyncio
async def test_search_entities_returns_list(daemon_pair):
    _, socket_path = daemon_pair
    result = await asyncio.to_thread(
        _client_call, socket_path, "search_entities", {"query": "nothing", "limit": 5},
    )
    assert isinstance(result, list)
    assert result == []


@pytest.mark.asyncio
async def test_get_entity_missing_returns_none(daemon_pair):
    _, socket_path = daemon_pair
    result = await asyncio.to_thread(
        _client_call, socket_path, "get_entity", {"entity_id": "does-not-exist"},
    )
    assert result is None


# ---------------------------------------------------------------------------
# Error pathways
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_method_returns_method_not_found(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(_client_call, socket_path, "no_such_method")
    assert ei.value.code == int(ErrorCode.METHOD_NOT_FOUND)


@pytest.mark.asyncio
async def test_phase3_stubs_return_not_implemented(daemon_pair):
    """Methods still stubbed for later phases must surface NOT_IMPLEMENTED.

    ``remember`` graduated to a real handler in PR E (Phase 2b). ``recall``
    still ships in Phase 4. ``ingest_source`` / ``register_source`` ship
    with Phase 3's multi-source ingesters.
    """
    _, socket_path = daemon_pair
    for stub_method in ("recall", "ingest_source", "register_source"):
        with pytest.raises(DaemonError) as ei:
            await asyncio.to_thread(_client_call, socket_path, stub_method)
        assert ei.value.code == int(ErrorCode.NOT_IMPLEMENTED), f"{stub_method} should be NOT_IMPLEMENTED"


@pytest.mark.asyncio
async def test_invalid_params_returns_error(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        # get_entity requires an entity_id string
        await asyncio.to_thread(_client_call, socket_path, "get_entity", {})
    assert ei.value.code == int(ErrorCode.INVALID_PARAMS)


def test_client_raises_when_daemon_unavailable(tmp_path: Path):
    # No daemon running at this path; connect() should fail immediately.
    missing = tmp_path / "nope.sock"
    client = GraphiteClient(socket_path=missing)
    with pytest.raises(DaemonUnavailable):
        client.ping()


# ---------------------------------------------------------------------------
# Shutdown & persistence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shutdown_removes_socket(tmp_path: Path):
    socket_path = _short_socket_path()
    daemon = Daemon(graph_root=tmp_path / "g", socket_path=socket_path)
    serve = asyncio.create_task(daemon.start())
    for _ in range(100):
        if socket_path.exists():
            break
        await asyncio.sleep(0.01)

    await daemon.stop()
    serve.cancel()
    try:
        await serve
    except (asyncio.CancelledError, Exception):
        pass

    assert not socket_path.exists(), "socket must be removed on graceful shutdown"


@pytest.mark.asyncio
async def test_kg_call_rejects_unlisted_method(daemon_pair):
    _, socket_path = daemon_pair
    with pytest.raises(DaemonError) as ei:
        await asyncio.to_thread(
            _client_call,
            socket_path,
            "kg_call",
            {"method": "set_graph", "args": [], "kwargs": {}},
        )
    # set_graph is a real PyKnowledgeGraph method but not on the whitelist.
    assert ei.value.code == int(ErrorCode.INVALID_PARAMS)
    assert "not exposed via kg_call" in ei.value.message.lower()


@pytest.mark.asyncio
async def test_kg_call_mutating_method_marks_dirty(daemon_pair):
    daemon, socket_path = daemon_pair

    def _add_entity_via_client():
        with GraphiteClient(socket_path=socket_path) as c:
            return c.kg_call(
                "add_entity",
                json.dumps({"canonical_name": "Alice", "entity_type": "Person"}),
            )

    before_dirty = daemon._dirty
    new_id = await asyncio.to_thread(_add_entity_via_client)
    assert isinstance(new_id, str) and len(new_id) > 0
    assert daemon._dirty is True
    assert before_dirty is False


@pytest.mark.asyncio
async def test_daemon_backed_graph_proxies_through_client(daemon_pair):
    """DaemonBackedGraph should transparently forward attribute access."""
    from graphite.client import DaemonBackedGraph

    _, socket_path = daemon_pair

    def _run():
        with GraphiteClient(socket_path=socket_path) as c:
            kg = DaemonBackedGraph(c)
            # Round-trip an entity via the proxy.
            new_id = kg.add_entity(
                json.dumps({"canonical_name": "Bob", "entity_type": "Person"})
            )
            # Read back via the proxy.
            entity_json = kg.get_entity(new_id)
            return new_id, entity_json

    new_id, entity_json = await asyncio.to_thread(_run)
    assert isinstance(new_id, str)
    entity = json.loads(entity_json)
    assert entity["canonical_name"] == "Bob"


@pytest.mark.asyncio
async def test_daemon_backed_graph_save_routes_to_force_save(daemon_pair):
    from graphite.client import DaemonBackedGraph

    daemon, socket_path = daemon_pair

    def _run():
        with GraphiteClient(socket_path=socket_path) as c:
            kg = DaemonBackedGraph(c)
            kg.add_entity(
                json.dumps({"canonical_name": "Carol", "entity_type": "Person"})
            )
            # save() on the proxy is remapped to force_save() on the daemon.
            kg.save("unused path arg")

    await asyncio.to_thread(_run)
    graph_file = daemon.graph_root / ".graphite" / "graph.msgpack"
    assert graph_file.exists()


@pytest.mark.asyncio
async def test_force_save_persists_dirty_graph(daemon_pair):
    daemon, socket_path = daemon_pair
    # Force the graph into a dirty-but-non-empty state so save_with_backup
    # actually writes something. We do this by adding one entity directly.
    async with daemon._kg_lock:
        daemon._ensure_graph()
        daemon._kg.add_entity(json.dumps({
            "canonical_name": "DaemonTestEntity",
            "entity_type": "Concept",
        }))
        daemon.mark_dirty()

    result = await asyncio.to_thread(_client_call, socket_path, "force_save")
    assert result["saved"] is True

    graph_file = daemon.graph_root / ".graphite" / "graph.msgpack"
    assert graph_file.exists()
    assert graph_file.stat().st_size > 10
