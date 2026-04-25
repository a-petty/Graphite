"""Tests for ``graphite mcp install --target openclaw`` (PR α of OpenClaw bring-up).

Two contracts to lock down — same shape as the Claude Code hooks merger:

  1. **Don't clobber the user.** Existing entries under ``mcp.servers``
     and any other top-level config keys must survive both install and
     uninstall.
  2. **Idempotent install.** Reinstalling rewrites only our entry; never
     duplicates and always reflects the current ``sys.executable`` (the
     stale-path replacement check).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from graphite import mcp_install


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _user_server_entry(label: str) -> dict:
    """A masquerading MCP server entry so we can verify install never
    touches anything other than ours."""
    return {"command": f"/usr/bin/{label}", "args": ["--demo"]}


# ---------------------------------------------------------------------------
# Install — fresh
# ---------------------------------------------------------------------------
class TestInstallFromScratch:
    def test_creates_config_when_missing(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        assert not cfg.exists()

        msg = mcp_install._install_openclaw(config_path=cfg)

        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert "mcp" in data
        assert mcp_install.SERVER_NAME in data["mcp"]["servers"]
        assert "Created" in msg

    def test_inserts_entry_with_command_and_args(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg)

        data = json.loads(cfg.read_text())
        entry = data["mcp"]["servers"][mcp_install.SERVER_NAME]
        assert "command" in entry
        # We pin to sys.executable in the python -m fallback path; either
        # the script wrapper or the interpreter is acceptable.
        assert entry["command"].endswith("graphite-mcp") or sys.executable in entry["command"]


# ---------------------------------------------------------------------------
# Install preserves user config
# ---------------------------------------------------------------------------
class TestInstallPreservesUserConfig:
    def test_existing_unrelated_servers_kept(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        prior = {
            "mcp": {
                "servers": {
                    "context7": _user_server_entry("context7"),
                    "docs":     {"url": "https://mcp.example.com"},
                }
            },
            "theme": "dark",  # unrelated top-level key
            "plugins": {"entries": {"composio": {"enabled": True}}},
        }
        cfg.write_text(json.dumps(prior))

        mcp_install._install_openclaw(config_path=cfg)
        data = json.loads(cfg.read_text())

        # Other servers untouched.
        assert data["mcp"]["servers"]["context7"] == _user_server_entry("context7")
        assert data["mcp"]["servers"]["docs"]["url"] == "https://mcp.example.com"
        # Our entry added.
        assert mcp_install.SERVER_NAME in data["mcp"]["servers"]
        # Unrelated keys preserved.
        assert data["theme"] == "dark"
        assert data["plugins"]["entries"]["composio"]["enabled"] is True

    def test_idempotent_no_drift_on_reinstall(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg)
        first = json.loads(cfg.read_text())
        mcp_install._install_openclaw(config_path=cfg)
        second = json.loads(cfg.read_text())
        assert first == second

    def test_replaces_stale_command_path(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        # Plant a stale entry pointing at a different python.
        cfg.write_text(json.dumps({
            "mcp": {
                "servers": {
                    mcp_install.SERVER_NAME: {
                        "command": "/old/venv/bin/python",
                        "args": ["-m", "graphite.mcp_server"],
                    }
                }
            }
        }))

        mcp_install._install_openclaw(config_path=cfg)
        data = json.loads(cfg.read_text())
        cmd = data["mcp"]["servers"][mcp_install.SERVER_NAME]["command"]
        assert "/old/venv/bin/python" not in cmd


# ---------------------------------------------------------------------------
# Install refuses malformed config
# ---------------------------------------------------------------------------
class TestInstallRefusesMalformed:
    def test_invalid_json_raises_clear_error(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        original = '{"mcp": broken-not-json'
        cfg.write_text(original)

        with pytest.raises(ValueError, match="Cannot parse"):
            mcp_install._install_openclaw(config_path=cfg)
        # File untouched.
        assert cfg.read_text() == original

    def test_top_level_array_rejected(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text("[]")
        with pytest.raises(ValueError, match="top-level"):
            mcp_install._install_openclaw(config_path=cfg)

    def test_mcp_field_with_wrong_type_rejected(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text('{"mcp": "not-a-dict"}')
        with pytest.raises(ValueError, match="`mcp` must be a JSON object"):
            mcp_install._install_openclaw(config_path=cfg)

    def test_mcp_servers_with_wrong_type_rejected(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text('{"mcp": {"servers": []}}')
        with pytest.raises(ValueError, match="`mcp.servers` must be a JSON object"):
            mcp_install._install_openclaw(config_path=cfg)

    def test_json5_with_comments_gives_helpful_hint(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text('{\n  // a comment\n  "mcp": {}\n}')
        with pytest.raises(ValueError) as ei:
            mcp_install._install_openclaw(config_path=cfg)
        assert "comments" in str(ei.value)


# ---------------------------------------------------------------------------
# Backups + atomic write
# ---------------------------------------------------------------------------
class TestBackupAndAtomicity:
    def test_creates_bak_file_on_overwrite(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        original = {"theme": "dark"}
        cfg.write_text(json.dumps(original))

        mcp_install._install_openclaw(config_path=cfg)
        bak = cfg.with_suffix(cfg.suffix + ".bak")
        assert bak.exists()
        assert json.loads(bak.read_text()) == original


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
class TestUninstall:
    def test_removes_only_ours(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text(json.dumps({
            "mcp": {
                "servers": {
                    "context7": _user_server_entry("context7"),
                }
            }
        }))
        mcp_install._install_openclaw(config_path=cfg)

        msg = mcp_install._uninstall_openclaw(config_path=cfg)
        assert "Removed" in msg

        data = json.loads(cfg.read_text())
        # Our entry gone.
        assert mcp_install.SERVER_NAME not in data.get("mcp", {}).get("servers", {})
        # User entry still there.
        assert data["mcp"]["servers"]["context7"] == _user_server_entry("context7")

    def test_drops_empty_containers(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg)
        mcp_install._uninstall_openclaw(config_path=cfg)

        data = json.loads(cfg.read_text())
        # Both `mcp.servers` and `mcp` should be cleaned up since we were
        # the only entry.
        assert "mcp" not in data

    def test_no_op_when_config_missing(self, tmp_path: Path):
        cfg = tmp_path / "absent.json"
        msg = mcp_install._uninstall_openclaw(config_path=cfg)
        assert "does not exist" in msg
        assert not cfg.exists()

    def test_no_op_when_we_are_not_registered(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text(json.dumps({
            "mcp": {"servers": {"context7": _user_server_entry("context7")}}
        }))
        msg = mcp_install._uninstall_openclaw(config_path=cfg)
        assert "Nothing to remove" in msg
        # Untouched.
        data = json.loads(cfg.read_text())
        assert data["mcp"]["servers"]["context7"] == _user_server_entry("context7")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
class TestStatus:
    def test_reports_missing_config(self, tmp_path: Path):
        s = mcp_install._status_openclaw(config_path=tmp_path / "absent.json")
        assert s.config_exists is False
        assert s.server_installed is False
        assert s.plugin_installed is False
        assert "does not exist" in s.message

    def test_reports_unregistered(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text(json.dumps({"mcp": {"servers": {"foo": _user_server_entry("foo")}}}))
        s = mcp_install._status_openclaw(config_path=cfg)
        assert s.config_exists is True
        assert s.server_installed is False
        assert s.plugin_installed is False

    def test_reports_installed_without_plugin(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg)
        s = mcp_install._status_openclaw(config_path=cfg)
        assert s.server_installed is True
        assert s.plugin_installed is False
        assert s.command is not None
        assert "capture plugin not enabled" in s.message

    def test_reports_fully_wired_when_plugin_enabled(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg, with_plugin=True)
        s = mcp_install._status_openclaw(config_path=cfg)
        assert s.server_installed is True
        assert s.plugin_installed is True
        assert "fully wired" in s.message

    def test_handles_malformed_config_gracefully(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text("totally not json")
        # Status should not raise even on a malformed config.
        s = mcp_install._status_openclaw(config_path=cfg)
        assert s.config_exists is True
        assert s.server_installed is False
        assert s.plugin_installed is False


# ---------------------------------------------------------------------------
# PR β — capture plugin entry
# ---------------------------------------------------------------------------
class TestPluginEntryInstall:
    def test_with_plugin_writes_plugins_entries_block(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg, with_plugin=True)
        data = json.loads(cfg.read_text())
        entry = data["plugins"]["entries"][mcp_install.PLUGIN_ID]
        assert entry["enabled"] is True
        assert entry["hooks"]["allowConversationAccess"] is True
        # Default daemon socket path is ~/.graphite/daemon.sock; the JS
        # plugin expands that.
        assert entry["config"]["daemonSocket"].endswith("daemon.sock")

    def test_without_plugin_does_not_touch_plugins_entries(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text(json.dumps({
            "plugins": {"entries": {"composio": {"enabled": True}}}
        }))
        mcp_install._install_openclaw(config_path=cfg)
        data = json.loads(cfg.read_text())
        # Composio entry untouched.
        assert data["plugins"]["entries"]["composio"]["enabled"] is True
        # Our plugin entry NOT created.
        assert mcp_install.PLUGIN_ID not in data["plugins"]["entries"]

    def test_with_plugin_preserves_unrelated_plugins(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text(json.dumps({
            "plugins": {"entries": {"composio": {"enabled": True}}}
        }))
        mcp_install._install_openclaw(config_path=cfg, with_plugin=True)
        data = json.loads(cfg.read_text())
        assert data["plugins"]["entries"]["composio"]["enabled"] is True
        assert data["plugins"]["entries"][mcp_install.PLUGIN_ID]["enabled"] is True

    def test_idempotent_with_plugin(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg, with_plugin=True)
        first = json.loads(cfg.read_text())
        mcp_install._install_openclaw(config_path=cfg, with_plugin=True)
        second = json.loads(cfg.read_text())
        assert first == second

    def test_uninstall_removes_both_server_and_plugin(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        mcp_install._install_openclaw(config_path=cfg, with_plugin=True)
        msg = mcp_install._uninstall_openclaw(config_path=cfg)
        assert "mcp.servers." in msg
        assert "plugins.entries." in msg

        data = json.loads(cfg.read_text())
        assert "mcp" not in data
        assert "plugins" not in data

    def test_uninstall_preserves_unrelated_plugins(self, tmp_path: Path):
        cfg = tmp_path / "openclaw.json"
        cfg.write_text(json.dumps({
            "plugins": {"entries": {"composio": {"enabled": True}}}
        }))
        mcp_install._install_openclaw(config_path=cfg, with_plugin=True)
        mcp_install._uninstall_openclaw(config_path=cfg)
        data = json.loads(cfg.read_text())
        # Composio survived; our plugin gone; container retained.
        assert data["plugins"]["entries"]["composio"]["enabled"] is True
        assert mcp_install.PLUGIN_ID not in data["plugins"]["entries"]

    def test_default_plugin_source_points_at_repo_dir(self):
        # The default path resolves to <repo>/openclaw_plugin/. The
        # directory exists in this repo with the manifest + JS entry.
        p = mcp_install.DEFAULT_PLUGIN_SOURCE
        assert (p / "openclaw.plugin.json").exists()
        assert (p / "index.js").exists()


# ---------------------------------------------------------------------------
# auto-link guard (subprocess plumbing — no real subprocess call)
# ---------------------------------------------------------------------------
class TestAutoLink:
    def test_auto_link_returns_path_message_when_cli_missing(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(mcp_install.shutil, "which", lambda _: None)
        msg = mcp_install._try_openclaw_plugin_link(tmp_path / "plugin")
        assert "openclaw" in msg
        assert "PATH" in msg

    def test_auto_link_invokes_subprocess_when_cli_present(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(mcp_install.shutil, "which", lambda _: "/usr/local/bin/openclaw")
        captured = {}
        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""
            return _R()
        monkeypatch.setattr(mcp_install.subprocess, "run", fake_run)
        msg = mcp_install._try_openclaw_plugin_link(tmp_path / "plugin")
        assert captured["cmd"][:3] == ["openclaw", "plugins", "install"]
        assert "Linked" in msg


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
class TestDispatch:
    def test_unknown_target_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unsupported target"):
            mcp_install.install(target="cursor")
        with pytest.raises(ValueError, match="Unsupported target"):
            mcp_install.uninstall(target="cursor")
        with pytest.raises(ValueError, match="Unsupported target"):
            mcp_install.status(target="cursor")
