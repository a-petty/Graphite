"""Manage Graphite's entry in another MCP client's config file.

Phase 2c (the OpenClaw bring-up). Same shape as ``hooks_control`` but
targets a different config schema: OpenClaw stores MCP servers as a
flat map at ``mcp.servers.<name>`` rather than the matcher-group array
Claude Code uses for hooks.

We deliberately keep this module client-agnostic on the surface:
``install(target)`` and ``uninstall(target)`` dispatch to per-target
helpers, so adding (say) Cursor or OpenInterpreter later is a matter of
writing one more helper, not rewriting the CLI.

Config-write contract — same as elsewhere in Graphite:
  * Atomic write via tmpfile + fsync + rename.
  * ``.bak`` of the prior file on every install/uninstall.
  * Refuse to clobber a config we can't parse — surface a clear error.
  * Idempotent: reinstall replaces our entry; never duplicates.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
TARGET_OPENCLAW = "openclaw"
SUPPORTED_TARGETS = (TARGET_OPENCLAW,)

# OpenClaw stores its config under ~/.openclaw/openclaw.json. The Gateway
# watches the file and reloads automatically, so no restart needed for
# most settings. (Plugin enables can require a `gateway restart` — see
# the plugin install path.)
OPENCLAW_CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"

# The exact name we register under mcp.servers.<name>. Anything else here
# is fair game for the user; we own this slot.
SERVER_NAME = "graphite"

# Plugin id (must match openclaw_plugin/openclaw.plugin.json's "id" field).
# We own this slot under plugins.entries.<id> in the user's OpenClaw config.
PLUGIN_ID = "graphite-capture"

# Default location of this repo's bundled plugin source. Used when the
# user runs `graphite mcp install --with-plugin` without an explicit path.
DEFAULT_PLUGIN_SOURCE = (
    Path(__file__).resolve().parent.parent.parent / "openclaw_plugin"
)


@dataclass
class McpInstallStatus:
    target: str
    config_path: Path
    config_exists: bool
    server_installed: bool
    plugin_installed: bool
    command: Optional[str]
    message: str


# ---------------------------------------------------------------------------
# Generic helpers (config I/O)
# ---------------------------------------------------------------------------
def _read_config(path: Path) -> tuple[dict, bool]:
    """Return (parsed_dict, file_existed). Raises ValueError on malformed
    JSON. OpenClaw nominally accepts JSON5 (comments, trailing commas);
    we only handle plain JSON. If the user's config has comments, the
    error message tells them to convert it manually rather than silently
    clobbering."""
    if not path.exists():
        return {}, False
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}, True
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        hint = ""
        if "//" in raw or "/*" in raw:
            hint = (
                " (the file appears to contain comments — Graphite reads strict "
                "JSON; either remove the comments or wire the entry by hand)"
            )
        raise ValueError(
            f"Cannot parse {path}: {e}{hint}. Refusing to overwrite."
        )
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} top-level must be a JSON object, got {type(data).__name__}"
        )
    return data, True


def _write_config(path: Path, data: dict) -> None:
    """Atomic write with .bak. Same contract as hooks_control._write_settings."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        except OSError:
            pass  # best-effort

    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# OpenClaw target
# ---------------------------------------------------------------------------
def _graphite_mcp_command() -> list[str]:
    """The ProgramArguments-equivalent for OpenClaw's stdio MCP transport.
    Matches the strategy used in daemon_control._program_args(): prefer
    the installed ``graphite-mcp`` script; fall back to ``python -m`` if
    the wrapper isn't on PATH (e.g. running from source without an
    editable install). We pin to ``sys.executable`` either way so OpenClaw
    spawns the right venv.
    """
    found = shutil.which("graphite-mcp")
    if found:
        return [found]
    return [sys.executable, "-m", "graphite.mcp_server"]


def _openclaw_server_entry() -> dict:
    """The exact value we want to live at ``mcp.servers.graphite``."""
    args = _graphite_mcp_command()
    head, *rest = args
    entry: dict = {"command": head}
    if rest:
        entry["args"] = rest
    return entry


def _ensure_mcp_servers(config: dict) -> dict:
    """Return the ``mcp.servers`` map, creating it if missing. Raises
    ValueError on shape mismatches."""
    mcp = config.setdefault("mcp", {})
    if not isinstance(mcp, dict):
        raise ValueError(
            f"`mcp` must be a JSON object, got {type(mcp).__name__}"
        )
    servers = mcp.setdefault("servers", {})
    if not isinstance(servers, dict):
        raise ValueError(
            f"`mcp.servers` must be a JSON object, got {type(servers).__name__}"
        )
    return servers


def _plugin_entry(daemon_socket: Optional[str] = None) -> dict:
    """The shape we want at ``plugins.entries.graphite-capture``.

    Mirrors the keys the JS plugin reads from ``api.config`` plus the
    ``hooks.allowConversationAccess`` flag the OpenClaw docs require for
    non-bundled conversation hooks (``llm_input``, ``llm_output``,
    ``agent_end``).
    """
    return {
        "enabled": True,
        "hooks": {"allowConversationAccess": True},
        "config": {
            "daemonSocket": daemon_socket or "~/.graphite/daemon.sock",
            "timeoutMs": 2000,
        },
    }


def _ensure_plugins_entries(config: dict) -> dict:
    plugins = config.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        raise ValueError(
            f"`plugins` must be a JSON object, got {type(plugins).__name__}"
        )
    entries = plugins.setdefault("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(
            f"`plugins.entries` must be a JSON object, got {type(entries).__name__}"
        )
    return entries


def _install_openclaw(
    config_path: Path = OPENCLAW_CONFIG_PATH,
    *,
    with_plugin: bool = False,
    plugin_source: Optional[Path] = None,
    auto_link: bool = False,
) -> str:
    config, existed = _read_config(config_path)
    servers = _ensure_mcp_servers(config)

    # Idempotent: overwrite our slot. Other servers untouched.
    new_entry = _openclaw_server_entry()
    previous = servers.get(SERVER_NAME)
    servers[SERVER_NAME] = new_entry

    plugin_msgs: list[str] = []
    if with_plugin:
        entries = _ensure_plugins_entries(config)
        prev_plugin = entries.get(PLUGIN_ID)
        new_plugin = _plugin_entry()
        entries[PLUGIN_ID] = new_plugin

        plugin_path = (plugin_source or DEFAULT_PLUGIN_SOURCE).resolve()
        if not (plugin_path / "HOOK.md").exists():
            plugin_msgs.append(
                f"  ! plugin source not found at {plugin_path} (HOOK.md missing); "
                f"entry written but `openclaw plugins install -l` will fail "
                f"until the path exists."
            )
        if prev_plugin != new_plugin:
            plugin_msgs.append(
                f"  + wrote plugins.entries.{PLUGIN_ID} (allowConversationAccess=true)."
            )
        else:
            plugin_msgs.append(
                f"  · plugins.entries.{PLUGIN_ID} unchanged."
            )

        if auto_link:
            link_msg = _try_openclaw_plugin_link(plugin_path)
            plugin_msgs.append(f"  · {link_msg}")
        else:
            plugin_msgs.append(
                f"  → next: `openclaw plugins install -l {plugin_path}` "
                f"to register the plugin source."
            )

    _write_config(config_path, config)

    if not existed:
        verb = "Created"
    elif previous == new_entry:
        verb = "Verified (no change)"
    elif previous is None:
        verb = "Added to"
    else:
        verb = "Updated in"

    head = (
        f"{verb} {config_path}: mcp.servers.{SERVER_NAME} -> "
        f"{json.dumps(new_entry)}"
    )
    tail = "\nOpenClaw watches this file and reloads automatically; no restart needed."
    plugin_block = ("\n" + "\n".join(plugin_msgs)) if plugin_msgs else ""
    return head + plugin_block + tail


def _try_openclaw_plugin_link(plugin_path: Path) -> str:
    """Run ``openclaw plugins install -l <path>`` if the CLI is on PATH.
    Best-effort — returns a status string but never raises into the
    install flow."""
    if not shutil.which("openclaw"):
        return f"`openclaw` CLI not on PATH; run manually: openclaw plugins install -l {plugin_path}"
    try:
        result = subprocess.run(
            ["openclaw", "plugins", "install", "-l", str(plugin_path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"openclaw plugins install -l failed: {e}"
    if result.returncode != 0:
        return f"openclaw plugins install -l exited {result.returncode}: {result.stderr.strip()}"
    return "Linked plugin via `openclaw plugins install -l`."


def _uninstall_openclaw(config_path: Path = OPENCLAW_CONFIG_PATH) -> str:
    if not config_path.exists():
        return f"{config_path} does not exist — nothing to uninstall."

    config, _ = _read_config(config_path)
    removed: list[str] = []

    mcp = config.get("mcp")
    if isinstance(mcp, dict):
        servers = mcp.get("servers")
        if isinstance(servers, dict) and SERVER_NAME in servers:
            del servers[SERVER_NAME]
            removed.append(f"mcp.servers.{SERVER_NAME}")
            if not servers:
                del mcp["servers"]
            if not mcp:
                del config["mcp"]

    plugins = config.get("plugins")
    if isinstance(plugins, dict):
        entries = plugins.get("entries")
        if isinstance(entries, dict) and PLUGIN_ID in entries:
            del entries[PLUGIN_ID]
            removed.append(f"plugins.entries.{PLUGIN_ID}")
            if not entries:
                del plugins["entries"]
            if not plugins:
                del config["plugins"]

    if not removed:
        return (
            f"Nothing to remove — neither mcp.servers.{SERVER_NAME} nor "
            f"plugins.entries.{PLUGIN_ID} present in {config_path}."
        )

    _write_config(config_path, config)
    return f"Removed {', '.join(removed)} from {config_path}."


def _status_openclaw(config_path: Path = OPENCLAW_CONFIG_PATH) -> McpInstallStatus:
    exists = config_path.exists()
    server_installed = False
    plugin_installed = False
    command: Optional[str] = None

    if exists:
        try:
            config, _ = _read_config(config_path)

            mcp = config.get("mcp") if isinstance(config, dict) else None
            servers = mcp.get("servers") if isinstance(mcp, dict) else None
            entry = servers.get(SERVER_NAME) if isinstance(servers, dict) else None
            if isinstance(entry, dict):
                server_installed = True
                cmd = entry.get("command")
                args = entry.get("args") or []
                if isinstance(cmd, str):
                    command = " ".join([cmd, *(a for a in args if isinstance(a, str))])

            plugins = config.get("plugins") if isinstance(config, dict) else None
            entries = plugins.get("entries") if isinstance(plugins, dict) else None
            plugin_entry = entries.get(PLUGIN_ID) if isinstance(entries, dict) else None
            if isinstance(plugin_entry, dict) and plugin_entry.get("enabled") is True:
                plugin_installed = True
        except ValueError:
            # Malformed config — report as not-installed plus a hint.
            pass

    if not exists:
        msg = (
            f"OpenClaw config at {config_path} does not exist yet. "
            f"Run `graphite mcp install --target openclaw` to create it."
        )
    elif not server_installed:
        msg = (
            f"Graphite MCP server not registered in {config_path}. "
            f"Run `graphite mcp install --target openclaw`."
        )
    elif not plugin_installed:
        msg = (
            f"MCP server wired (read works), but capture plugin not enabled. "
            f"Run `graphite mcp install --target openclaw --with-plugin` for write capture."
        )
    else:
        msg = (
            f"Graphite fully wired into OpenClaw: mcp.servers.{SERVER_NAME} + "
            f"plugins.entries.{PLUGIN_ID}. Command: {command}"
        )

    return McpInstallStatus(
        target=TARGET_OPENCLAW,
        config_path=config_path,
        config_exists=exists,
        server_installed=server_installed,
        plugin_installed=plugin_installed,
        command=command,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Public dispatchers (target-agnostic surface for the CLI)
# ---------------------------------------------------------------------------
def install(target: str = TARGET_OPENCLAW, **kwargs) -> str:
    if target == TARGET_OPENCLAW:
        return _install_openclaw(**kwargs)
    raise ValueError(f"Unsupported target: {target!r}. Supported: {SUPPORTED_TARGETS}")


def uninstall(target: str = TARGET_OPENCLAW, **kwargs) -> str:
    if target == TARGET_OPENCLAW:
        return _uninstall_openclaw(**kwargs)
    raise ValueError(f"Unsupported target: {target!r}. Supported: {SUPPORTED_TARGETS}")


def status(target: str = TARGET_OPENCLAW, **kwargs) -> McpInstallStatus:
    if target == TARGET_OPENCLAW:
        return _status_openclaw(**kwargs)
    raise ValueError(f"Unsupported target: {target!r}. Supported: {SUPPORTED_TARGETS}")
