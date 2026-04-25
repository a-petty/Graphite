"""Tests for the capture hook handler (PR B of Phase 2a).

The hook runs as a subprocess of Claude Code with three hard contracts:

  1. Never exit non-zero, no matter what fails. A crashing hook would
     block session shutdown or compaction — unacceptable UX for a
     best-effort side-effect.
  2. Always archive the transcript before attempting anything daemon-side.
     The archive is the durability layer; if the daemon is down, the
     reconciler can replay from archive later.
  3. Don't block the caller for more than a second or two waiting on
     the daemon. The client enforces this via a short timeout.

Tests exercise both entry points (``handle_session_end`` and
``handle_pre_compact``) directly in-process, and the full ``main()``
subprocess for the non-zero-exit contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from graphite.capture import hook_handler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_transcript(tmp_path: Path) -> Path:
    """A minimal Claude-Code-style JSONL on disk for the handler to archive."""
    project_dir = tmp_path / ".claude" / "projects" / "-tmp-TestProject"
    project_dir.mkdir(parents=True)
    transcript = project_dir / "abc123.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "content": "hello"}) + "\n"
        + json.dumps({"type": "assistant", "content": "hi"}) + "\n"
    )
    return transcript


@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    return tmp_path / "archive"


@pytest.fixture
def session_hook_payload(fake_transcript: Path) -> dict:
    return {
        "transcript_path": str(fake_transcript),
        "session_id": "abc123",
        "cwd": "/Users/apetty/Dev/TestProject",
    }


# ---------------------------------------------------------------------------
# Archive behavior
# ---------------------------------------------------------------------------
class TestArchive:
    def test_session_end_archives_transcript(self, session_hook_payload, archive_dir):
        with mock.patch.object(hook_handler, "_try_enqueue_ingest", return_value=True) as enq:
            hook_handler.handle_session_end(session_hook_payload, archive_dir)
            archived = archive_dir / "abc123.jsonl"
            assert archived.exists()
            assert "hello" in archived.read_text()
            assert enq.called

    def test_pre_compact_archives_but_does_not_enqueue(
        self, session_hook_payload, archive_dir,
    ):
        """A compact isn't end-of-session — ingesting now would shred a
        still-live conversation. Archive only."""
        with mock.patch.object(hook_handler, "_try_enqueue_ingest") as enq:
            hook_handler.handle_pre_compact(session_hook_payload, archive_dir)
            assert (archive_dir / "abc123.jsonl").exists()
            assert not enq.called

    def test_missing_transcript_is_silently_skipped(self, archive_dir):
        payload = {"session_id": "nowhere", "cwd": "/nonexistent"}
        with mock.patch.object(hook_handler, "_try_enqueue_ingest") as enq:
            hook_handler.handle_session_end(payload, archive_dir)
            assert not archive_dir.exists() or not any(archive_dir.iterdir())
            assert not enq.called

    def test_repeated_session_end_overwrites_same_archive(
        self, session_hook_payload, archive_dir, fake_transcript,
    ):
        """Dedup by session_id: a second SessionEnd for the same session
        just overwrites. Safe because ingestion is content-hash-idempotent."""
        with mock.patch.object(hook_handler, "_try_enqueue_ingest"):
            hook_handler.handle_session_end(session_hook_payload, archive_dir)
            # Mutate the source transcript and fire again.
            fake_transcript.write_text(
                json.dumps({"type": "user", "content": "second pass"}) + "\n"
            )
            hook_handler.handle_session_end(session_hook_payload, archive_dir)

            archived = archive_dir / "abc123.jsonl"
            assert "second pass" in archived.read_text()


# ---------------------------------------------------------------------------
# Daemon enqueue path — never surfaces exceptions
# ---------------------------------------------------------------------------
class TestEnqueueBestEffort:
    def test_enqueue_success_path(self, archive_dir, tmp_path: Path):
        archive_path = tmp_path / "sess.jsonl"
        archive_path.write_text("dummy")

        fake_client = mock.MagicMock()
        fake_client.enqueue_session_ingest.return_value = {
            "job_id": "abc", "queue_position": 1,
        }
        fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
        fake_client.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("graphite.client.GraphiteClient", return_value=fake_client):
            ok = hook_handler._try_enqueue_ingest(archive_path, project="test")
        assert ok is True
        fake_client.enqueue_session_ingest.assert_called_once_with(
            path=str(archive_path), project="test",
        )

    def test_daemon_unavailable_returns_false_never_raises(self, tmp_path: Path):
        from graphite.client import DaemonUnavailable

        archive_path = tmp_path / "sess.jsonl"
        archive_path.write_text("dummy")

        raising_client = mock.MagicMock()
        raising_client.__enter__ = mock.MagicMock(
            side_effect=DaemonUnavailable("socket not found")
        )
        raising_client.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("graphite.client.GraphiteClient", return_value=raising_client):
            ok = hook_handler._try_enqueue_ingest(archive_path, project=None)
        assert ok is False  # logged, not raised

    def test_daemon_error_returns_false_never_raises(self, tmp_path: Path):
        from graphite.client import DaemonError

        archive_path = tmp_path / "sess.jsonl"
        archive_path.write_text("dummy")

        fake_client = mock.MagicMock()
        fake_client.enqueue_session_ingest.side_effect = DaemonError(
            code=-32001, message="no LLM configured",
        )
        fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
        fake_client.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("graphite.client.GraphiteClient", return_value=fake_client):
            ok = hook_handler._try_enqueue_ingest(archive_path, project=None)
        assert ok is False

    def test_unexpected_exception_returns_false(self, tmp_path: Path):
        archive_path = tmp_path / "sess.jsonl"
        archive_path.write_text("dummy")

        fake_client = mock.MagicMock()
        fake_client.__enter__ = mock.MagicMock(
            side_effect=RuntimeError("something weird")
        )
        fake_client.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("graphite.client.GraphiteClient", return_value=fake_client):
            ok = hook_handler._try_enqueue_ingest(archive_path, project=None)
        assert ok is False


# ---------------------------------------------------------------------------
# Project extraction
# ---------------------------------------------------------------------------
class TestProjectExtraction:
    def test_cwd_basename(self):
        assert hook_handler._extract_project_name({
            "cwd": "/Users/apetty/Dev/Graphite"
        }) == "Graphite"

    def test_transcript_path_fallback(self):
        # No cwd, but the encoded project dir is parseable.
        payload = {
            "transcript_path":
                "/Users/apetty/.claude/projects/-Users-apetty-Dev-Graphite/xyz.jsonl"
        }
        name = hook_handler._extract_project_name(payload)
        assert name == "Graphite"

    def test_returns_none_when_nothing_available(self):
        assert hook_handler._extract_project_name({}) is None

    def test_cwd_with_trailing_slash(self):
        # Path(...).name on a path ending in "/" returns "" — make sure
        # we don't return an empty string as a project name.
        name = hook_handler._extract_project_name({"cwd": "/Users/apetty/"})
        # Path("/Users/apetty/").name is "apetty" — fine.
        assert name == "apetty"


# ---------------------------------------------------------------------------
# main() exit-code contract
# ---------------------------------------------------------------------------
class TestNeverExitsNonZero:
    """The hook runs as a subprocess; a non-zero exit would block Claude Code.
    These tests invoke ``main()`` under every failure mode we can induce and
    assert the exit code stays at 0.
    """

    def _run_main(
        self, event: str, stdin_json: dict,
        archive_dir: Path, env: dict | None = None,
    ) -> int:
        """Invoke ``python -m graphite.capture.hook_handler`` as a real
        subprocess so we're testing the actual command Claude Code fires.
        """
        proc = subprocess.run(
            [sys.executable, "-m", "graphite.capture.hook_handler",
             "--event", event, "--archive-dir", str(archive_dir)],
            input=json.dumps(stdin_json).encode("utf-8"),
            capture_output=True,
            timeout=15,
            env={**os.environ, **(env or {})},
        )
        return proc.returncode

    def test_zero_exit_with_empty_stdin(self, tmp_path: Path):
        code = subprocess.run(
            [sys.executable, "-m", "graphite.capture.hook_handler",
             "--event", "session-end", "--archive-dir", str(tmp_path / "arch")],
            input=b"",
            capture_output=True,
            timeout=10,
        ).returncode
        assert code == 0

    def test_zero_exit_with_malformed_stdin(self, tmp_path: Path):
        code = subprocess.run(
            [sys.executable, "-m", "graphite.capture.hook_handler",
             "--event", "session-end", "--archive-dir", str(tmp_path / "arch")],
            input=b"{this is not json",
            capture_output=True,
            timeout=10,
        ).returncode
        assert code == 0

    def test_zero_exit_when_transcript_missing(self, tmp_path: Path):
        payload = {"session_id": "does-not-exist", "cwd": "/nonexistent"}
        code = self._run_main("session-end", payload, tmp_path / "arch")
        assert code == 0

    def test_zero_exit_when_daemon_unreachable(
        self, tmp_path: Path, session_hook_payload, fake_transcript,
    ):
        # No daemon running; socket path doesn't exist. Hook should still exit 0.
        code = self._run_main("session-end", session_hook_payload, tmp_path / "arch")
        assert code == 0
        # And the archive side-effect still happened.
        assert (tmp_path / "arch" / "abc123.jsonl").exists()

    def test_zero_exit_for_pre_compact(
        self, tmp_path: Path, session_hook_payload,
    ):
        code = self._run_main("pre-compact", session_hook_payload, tmp_path / "arch")
        assert code == 0
        assert (tmp_path / "arch" / "abc123.jsonl").exists()


# ---------------------------------------------------------------------------
# End-to-end against a real daemon (if one can be started in the fixture)
# ---------------------------------------------------------------------------
# Covered in test_daemon_ingest.py for the daemon side. The hook → daemon
# wire-up is validated above via subprocess + unreachable-daemon.
