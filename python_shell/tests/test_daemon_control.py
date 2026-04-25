"""Unit tests for the launchd control plane.

Tests the pieces that are mechanical — plist generation, argument shaping,
status composition. The actual ``launchctl bootstrap/bootout`` round-trip
can only be meaningfully validated on a real macOS box by running
``graphite daemon install && sleep && graphite daemon status`` by hand;
that is the manual smoke test tracked in the PR 8 completion criteria.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from graphite import daemon_control


class TestProgramArgs:
    def test_uses_graphited_script_when_on_path(self, monkeypatch):
        monkeypatch.setattr(
            daemon_control.shutil, "which",
            lambda _name: "/Users/test/.venv/bin/graphited",
        )
        args = daemon_control._program_args()
        assert args == ["/Users/test/.venv/bin/graphited"]

    def test_falls_back_to_python_m_graphite_daemon(self, monkeypatch):
        monkeypatch.setattr(daemon_control.shutil, "which", lambda _name: None)
        monkeypatch.setattr(daemon_control.sys, "executable", "/fake/python")
        args = daemon_control._program_args()
        assert args == ["/fake/python", "-m", "graphite.daemon"]


class TestPlistDict:
    def test_contains_required_fields(self):
        plist = daemon_control._build_plist_dict()
        assert plist["Label"] == daemon_control.LABEL
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True
        assert plist["ProcessType"] == "Background"
        assert "ProgramArguments" in plist
        assert plist["StandardOutPath"].endswith("daemon.out")
        assert plist["StandardErrorPath"].endswith("daemon.err")
        assert "EnvironmentVariables" in plist
        assert "PATH" in plist["EnvironmentVariables"]

    def test_plist_is_valid_plist(self, tmp_path: Path):
        plist = daemon_control._build_plist_dict()
        out = tmp_path / "test.plist"
        with open(out, "wb") as f:
            plistlib.dump(plist, f)
        # Round-trip — if plistlib can't re-parse its own output, we're broken.
        with open(out, "rb") as f:
            loaded = plistlib.load(f)
        assert loaded["Label"] == daemon_control.LABEL


class TestServiceTargets:
    def test_gui_domain_uses_current_uid(self, monkeypatch):
        monkeypatch.setattr(daemon_control.os, "getuid", lambda: 501)
        assert daemon_control._gui_domain() == "gui/501"

    def test_service_target_combines_domain_and_label(self, monkeypatch):
        monkeypatch.setattr(daemon_control.os, "getuid", lambda: 501)
        assert daemon_control._service_target() == f"gui/501/{daemon_control.LABEL}"


class TestStatus:
    def test_reports_not_installed_when_plist_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(daemon_control, "PLIST_PATH", tmp_path / "nope.plist")
        monkeypatch.setattr(daemon_control, "DEFAULT_SOCKET", tmp_path / "absent.sock")
        s = daemon_control.status()
        assert s.installed is False
        assert s.running is False
        assert "install" in s.message.lower()

    def test_reports_installed_but_not_running(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "present.plist"
        plist_path.write_text("<plist/>")
        monkeypatch.setattr(daemon_control, "PLIST_PATH", plist_path)
        monkeypatch.setattr(daemon_control, "DEFAULT_SOCKET", tmp_path / "absent.sock")
        # Force _parse_pid_from_print to return None (launchctl print returns error).
        monkeypatch.setattr(
            daemon_control, "_launchctl",
            lambda *args: subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr=""),
        )

        s = daemon_control.status()
        assert s.installed is True
        assert s.running is False
        assert s.reachable is False

    def test_pid_parsed_from_launchctl_print(self, monkeypatch):
        fake_output = (
            "com.graphite.daemon = {\n"
            "  active count = 1\n"
            "  pid = 12345\n"
            "  state = running\n"
            "}\n"
        )
        monkeypatch.setattr(
            daemon_control, "_launchctl",
            lambda *args: subprocess.CompletedProcess(
                args=args, returncode=0, stdout=fake_output, stderr=""
            ),
        )
        pid = daemon_control._parse_pid_from_print()
        assert pid == 12345


class TestInstallWritesPlistBeforeBootstrap:
    def test_plist_file_written_with_correct_label(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "install.plist"
        log_dir = tmp_path / "logs"
        monkeypatch.setattr(daemon_control, "PLIST_PATH", plist_path)
        monkeypatch.setattr(daemon_control, "LOG_DIR", log_dir)
        # Pretend launchctl always succeeds so install() proceeds through.
        monkeypatch.setattr(
            daemon_control, "_launchctl",
            lambda *args: subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=""),
        )

        msg = daemon_control.install()

        assert plist_path.exists()
        with open(plist_path, "rb") as f:
            loaded = plistlib.load(f)
        assert loaded["Label"] == daemon_control.LABEL
        assert "Installed and started" in msg
