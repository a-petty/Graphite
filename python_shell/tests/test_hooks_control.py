"""Tests for the ``graphite hooks`` settings.json merger (PR C of Phase 2a).

Two contracts to lock down:

  1. **Don't clobber the user.** If they already have hooks in
     ``~/.claude/settings.json``, our install must not delete or modify
     them. Uninstall must remove only our entries.
  2. **Idempotent install.** Running ``graphite hooks install`` twice does
     not produce duplicate entries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from graphite import hooks_control


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ours_in(arr: list) -> int:
    """Count matcher groups in ``arr`` whose command identifies as ours."""
    return sum(1 for g in arr if hooks_control._is_ours(g))


def _user_hook(label: str) -> dict:
    """A matcher group masquerading as someone else's hook — used to verify
    we don't disturb other tools' entries."""
    return {
        "matcher": {},
        "hooks": [
            {"type": "command", "command": f"/usr/bin/true # {label}"}
        ],
    }


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
class TestInstallFromScratch:
    def test_creates_settings_when_missing(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        assert not settings_path.exists()

        msg = hooks_control.install(settings_path)

        assert settings_path.exists()
        data = json.loads(settings_path.read_text())
        assert "hooks" in data
        for event in hooks_control.EVENTS:
            assert event in data["hooks"]
            assert _ours_in(data["hooks"][event]) == 1
        assert "Created" in msg

    def test_inserts_correct_command_for_each_event(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        hooks_control.install(settings_path)

        data = json.loads(settings_path.read_text())
        for event in hooks_control.EVENTS:
            entries = [g for g in data["hooks"][event] if hooks_control._is_ours(g)]
            assert len(entries) == 1
            cmd = entries[0]["hooks"][0]["command"]
            event_arg = "session-end" if event == "SessionEnd" else "pre-compact"
            assert sys.executable in cmd
            assert f"--event {event_arg}" in cmd
            assert "graphite.capture.hook_handler" in cmd


class TestInstallPreservesUserHooks:
    def test_existing_unrelated_hooks_are_kept(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        # User already has a SessionEnd hook for some other tool.
        prior = {
            "hooks": {
                "SessionEnd": [_user_hook("user-hook-A")],
                "PreCompact": [_user_hook("user-hook-B")],
                "Stop": [_user_hook("user-hook-C")],
            },
            "theme": "dark",  # an unrelated top-level key
        }
        settings_path.write_text(json.dumps(prior))

        hooks_control.install(settings_path)
        data = json.loads(settings_path.read_text())

        # Unrelated top-level key preserved.
        assert data["theme"] == "dark"

        # Our entries appended without disturbing the user's.
        for event in hooks_control.EVENTS:
            arr = data["hooks"][event]
            assert _ours_in(arr) == 1
            assert any(not hooks_control._is_ours(g) for g in arr), (
                f"user's {event} hook was clobbered"
            )

        # Stop hook (we don't manage it) is intact.
        assert data["hooks"]["Stop"] == [_user_hook("user-hook-C")]

    def test_idempotent_no_duplicates_on_reinstall(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        hooks_control.install(settings_path)
        hooks_control.install(settings_path)
        hooks_control.install(settings_path)

        data = json.loads(settings_path.read_text())
        for event in hooks_control.EVENTS:
            assert _ours_in(data["hooks"][event]) == 1, (
                f"reinstall produced duplicates for {event}"
            )

    def test_reinstall_replaces_stale_command_path(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        # Plant a stale entry of ours pointing at a different python.
        prior = {
            "hooks": {
                "SessionEnd": [{
                    "matcher": {},
                    "hooks": [{
                        "type": "command",
                        "command": "/old/venv/bin/python -m graphite.capture.hook_handler --event session-end",
                    }],
                }],
                "PreCompact": [],
            }
        }
        settings_path.write_text(json.dumps(prior))

        hooks_control.install(settings_path)
        data = json.loads(settings_path.read_text())
        ours = [g for g in data["hooks"]["SessionEnd"] if hooks_control._is_ours(g)]
        assert len(ours) == 1
        # New entry should point at the current sys.executable, not the old one.
        cmd = ours[0]["hooks"][0]["command"]
        assert "/old/venv/bin/python" not in cmd
        assert sys.executable in cmd


class TestInstallRefusesMalformedSettings:
    def test_invalid_json_raises_clear_error(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        original = '{"hooks": broken-not-json'
        settings_path.write_text(original)

        with pytest.raises(ValueError, match="Cannot parse"):
            hooks_control.install(settings_path)

        # The file is left untouched.
        assert settings_path.read_text() == original

    def test_top_level_array_rejected(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("[]")
        with pytest.raises(ValueError, match="top-level"):
            hooks_control.install(settings_path)

    def test_hooks_field_with_wrong_type_rejected(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text('{"hooks": "not-a-dict"}')
        with pytest.raises(ValueError, match="hooks must be a JSON object"):
            hooks_control.install(settings_path)


class TestInstallBacksUpAndWritesAtomically:
    def test_creates_bak_file_on_overwrite(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        original = {"theme": "dark"}
        settings_path.write_text(json.dumps(original))

        hooks_control.install(settings_path)
        bak = settings_path.with_suffix(settings_path.suffix + ".bak")
        assert bak.exists()
        assert json.loads(bak.read_text()) == original


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
class TestUninstall:
    def test_removes_only_ours(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        # User hooks + ours.
        prior = {
            "hooks": {
                "SessionEnd": [_user_hook("user-A")],
                "PreCompact": [_user_hook("user-B")],
            }
        }
        settings_path.write_text(json.dumps(prior))
        hooks_control.install(settings_path)  # adds ours alongside

        msg = hooks_control.uninstall(settings_path)
        assert "Removed" in msg

        data = json.loads(settings_path.read_text())
        # User hooks intact, ours gone.
        for event in hooks_control.EVENTS:
            arr = data.get("hooks", {}).get(event, [])
            assert _ours_in(arr) == 0
        assert _user_hook("user-A") in data["hooks"]["SessionEnd"]
        assert _user_hook("user-B") in data["hooks"]["PreCompact"]

    def test_drops_empty_event_arrays(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        hooks_control.install(settings_path)  # ours only, no user hooks
        hooks_control.uninstall(settings_path)

        data = json.loads(settings_path.read_text())
        # Both events should be gone now that the only entries were ours.
        assert "hooks" not in data or all(
            event not in data.get("hooks", {}) for event in hooks_control.EVENTS
        )

    def test_no_op_when_settings_missing(self, tmp_path: Path):
        settings_path = tmp_path / "nope.json"
        msg = hooks_control.uninstall(settings_path)
        assert "does not exist" in msg
        assert not settings_path.exists()

    def test_no_op_when_nothing_of_ours(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps({
            "hooks": {"SessionEnd": [_user_hook("user-only")]}
        }))
        msg = hooks_control.uninstall(settings_path)
        assert "No Graphite hook entries" in msg
        # User's hook still there.
        data = json.loads(settings_path.read_text())
        assert _user_hook("user-only") in data["hooks"]["SessionEnd"]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
class TestStatus:
    def test_reports_not_installed(self, tmp_path: Path, monkeypatch):
        # Point the archive count at a clean tmp dir to keep it deterministic.
        monkeypatch.setenv("HOME", str(tmp_path))
        s = hooks_control.status(tmp_path / "settings.json")
        assert s.installed_events == []
        assert "not installed" in s.message.lower()

    def test_reports_installed_events(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        settings_path = tmp_path / "settings.json"
        hooks_control.install(settings_path)
        s = hooks_control.status(settings_path)
        assert sorted(s.installed_events) == sorted(hooks_control.EVENTS)

    def test_reports_partial_install(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        settings_path = tmp_path / "settings.json"
        # Manually wire up just one event.
        settings_path.write_text(json.dumps({
            "hooks": {"SessionEnd": [hooks_control._our_matcher_group("SessionEnd")]}
        }))
        s = hooks_control.status(settings_path)
        assert s.installed_events == ["SessionEnd"]
        assert "Partial install" in s.message

    def test_archive_count_reflects_disk(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        archive = tmp_path / ".graphite" / "archive" / "sessions"
        archive.mkdir(parents=True)
        (archive / "a.jsonl").write_text("{}")
        (archive / "b.jsonl").write_text("{}")
        (archive / "noise.txt").write_text("ignore me")

        s = hooks_control.status(tmp_path / "settings.json")
        assert s.archived_sessions == 2  # the .txt is ignored
