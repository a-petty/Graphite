"""Tests for the PR G CLI commands: ``graphite remember`` and ``graphite spool``.

These exercise argparse wiring + handler dispatch against a mocked
``GraphiteClient`` so we don't need a live daemon. End-to-end correctness
of the underlying RPCs is covered in ``test_daemon_ingest.py``.
"""

from __future__ import annotations

from unittest import mock

import pytest

from graphite import cli


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------
class TestArgparse:
    def test_remember_command_parses(self):
        parser = cli.create_parser()
        args = parser.parse_args([
            "remember", "I prefer pnpm",
            "--source", "remember://test",
            "--category", "Semantic",
            "--project", "Graphite",
            "--entity-hint", "pnpm",
            "--entity-hint", "package-manager",
        ])
        assert args.command == "remember"
        assert args.text == "I prefer pnpm"
        assert args.source == "remember://test"
        assert args.category == "Semantic"
        assert args.project == "Graphite"
        assert args.entity_hints == ["pnpm", "package-manager"]

    def test_remember_command_has_minimal_form(self):
        parser = cli.create_parser()
        args = parser.parse_args(["remember", "just text"])
        assert args.text == "just text"
        assert args.category == "Episodic"
        assert args.source is None
        assert args.entity_hints is None

    def test_remember_command_rejects_bad_category(self):
        parser = cli.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["remember", "x", "--category", "Bogus"])

    def test_spool_actions_parse(self):
        parser = cli.create_parser()
        for action in ("status", "flush", "retry-failed", "cleanup"):
            args = parser.parse_args(["spool", action])
            assert args.command == "spool"
            assert args.spool_action == action

    def test_spool_flush_supports_filters(self):
        parser = cli.create_parser()
        args = parser.parse_args([
            "spool", "flush",
            "--source-filter", "session://abc",
            "--limit", "200",
        ])
        assert args.source_filter == "session://abc"
        assert args.limit == 200

    def test_spool_cleanup_supports_retain_days(self):
        parser = cli.create_parser()
        args = parser.parse_args(["spool", "cleanup", "--retain-days", "60"])
        assert args.retain_days == 60


# ---------------------------------------------------------------------------
# handle_remember_command
# ---------------------------------------------------------------------------
class TestHandleRemember:
    def _args(self, **kw):
        defaults = dict(
            command="remember",
            text="hello",
            source=None,
            category="Episodic",
            project=None,
            entity_hints=None,
        )
        defaults.update(kw)
        return mock.MagicMock(**defaults)

    def test_calls_client_remember_with_args(self):
        fake_client = mock.MagicMock()
        fake_client.__enter__ = mock.MagicMock(return_value=fake_client)
        fake_client.__exit__ = mock.MagicMock(return_value=False)
        fake_client.remember.return_value = {
            "fragment_id": 42, "source_id": "remember://x", "pending_count": 1,
        }

        with mock.patch("graphite.client.GraphiteClient", return_value=fake_client):
            cli.handle_remember_command(self._args(
                text="I prefer pnpm",
                source="remember://test",
                category="Semantic",
                project="Graphite",
                entity_hints=["pnpm"],
            ))

        fake_client.remember.assert_called_once_with(
            text="I prefer pnpm",
            source_id="remember://test",
            category="Semantic",
            project="Graphite",
            entity_hints=["pnpm"],
        )

    def test_strips_text_and_rejects_empty(self):
        with pytest.raises(SystemExit):
            cli.handle_remember_command(self._args(text="   "))

    def test_daemon_unavailable_exits_nonzero_with_clear_message(self, capsys):
        from graphite.client import DaemonUnavailable

        raising = mock.MagicMock()
        raising.__enter__ = mock.MagicMock(side_effect=DaemonUnavailable("no socket"))
        raising.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("graphite.client.GraphiteClient", return_value=raising):
            with pytest.raises(SystemExit) as ei:
                cli.handle_remember_command(self._args(text="x"))
        assert ei.value.code == 1


# ---------------------------------------------------------------------------
# handle_spool_command
# ---------------------------------------------------------------------------
class TestHandleSpool:
    def _client_returning(self, methods: dict):
        c = mock.MagicMock()
        c.connect = mock.MagicMock()
        c.close = mock.MagicMock()
        for name, value in methods.items():
            getattr(c, name).return_value = value
        return c

    def _args(self, action, **kw):
        return mock.MagicMock(command="spool", spool_action=action, **kw)

    def test_status_calls_spool_status(self):
        client = self._client_returning({"spool_status": {
            "counts": {"pending": 1, "extracting": 0, "extracted": 2, "failed": 0, "total": 3},
            "recent_batches": [],
            "failed_sample": [],
        }})
        with mock.patch("graphite.client.GraphiteClient", return_value=client):
            cli.handle_spool_command(self._args("status"))
        client.spool_status.assert_called_once()

    def test_flush_passes_through_filters(self):
        client = self._client_returning({"flush_spool": {"job_id": "j1", "queue_position": 1}})
        with mock.patch("graphite.client.GraphiteClient", return_value=client):
            cli.handle_spool_command(self._args(
                "flush", source_filter="session://x", limit=200,
            ))
        client.flush_spool.assert_called_once_with(
            source_filter="session://x", limit=200,
        )

    def test_retry_failed_calls_rpc(self):
        client = self._client_returning({"spool_retry_failed": {"reset": 5}})
        with mock.patch("graphite.client.GraphiteClient", return_value=client):
            cli.handle_spool_command(self._args("retry-failed"))
        client.spool_retry_failed.assert_called_once()

    def test_cleanup_passes_retain_days(self):
        client = self._client_returning(
            {"spool_cleanup": {"removed": 3, "retain_days": 60}}
        )
        with mock.patch("graphite.client.GraphiteClient", return_value=client):
            cli.handle_spool_command(self._args("cleanup", retain_days=60))
        client.spool_cleanup.assert_called_once_with(retain_days=60)

    def test_no_action_exits_nonzero(self):
        with pytest.raises(SystemExit):
            cli.handle_spool_command(self._args(None))

    def test_daemon_unavailable_exits_nonzero(self):
        from graphite.client import DaemonUnavailable

        client = mock.MagicMock()
        client.connect = mock.MagicMock(side_effect=DaemonUnavailable("nope"))
        with mock.patch("graphite.client.GraphiteClient", return_value=client):
            with pytest.raises(SystemExit) as ei:
                cli.handle_spool_command(self._args("status"))
        assert ei.value.code == 1
