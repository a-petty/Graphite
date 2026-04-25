"""Manage the entries Graphite owns in ``~/.claude/settings.json``.

Claude Code's hook surface is a top-level ``hooks`` map with one array per
event (``PreCompact``, ``SessionEnd``, etc.). Each array element is a
"matcher group" with its own ``hooks`` array of commands. Multiple tools
can register hooks for the same event; we have to **merge** without
clobbering whatever else the user has configured.

Identification: we tag our entries by command substring — every Graphite
hook command contains ``graphite.capture.hook_handler``. This is simpler
and more durable than a custom ``id`` field that Claude Code might strip.

Atomic writes: settings.json is overwritten via tmpfile + rename, with a
``.bak`` saved on every install/uninstall. Refuses to touch the file if
it can't be parsed (saves the user from a silent clobber after they've
hand-edited).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# The two events we care about. PreCompact fires before context compaction;
# SessionEnd fires when the Claude Code session terminates.
EVENTS = ("PreCompact", "SessionEnd")

# Our identifier — every command we insert contains this exact substring,
# which is how we recognize our entries on uninstall + idempotent reinstall.
HOOK_COMMAND_MARKER = "graphite.capture.hook_handler"

DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


@dataclass
class HooksStatus:
    settings_path: Path
    settings_exists: bool
    installed_events: list[str]   # which of our events are currently wired
    archived_sessions: int        # files in ~/.graphite/archive/sessions/
    daemon_reachable: bool
    message: str


def _command_for(event: str) -> str:
    """The exact ``command`` string we install for a given event.

    We pin to ``sys.executable`` rather than ``python3`` so the hook runs
    against the venv where Graphite is installed, regardless of what's on
    the user's PATH at hook-fire time. The Claude Code hook runner does
    not source the user's shell rc files.
    """
    event_arg = "session-end" if event == "SessionEnd" else "pre-compact"
    return f"{sys.executable} -m graphite.capture.hook_handler --event {event_arg}"


def _is_ours(matcher_group: dict) -> bool:
    """True iff this matcher group is one of ours (by command substring)."""
    for hook in matcher_group.get("hooks", []) or []:
        cmd = hook.get("command", "")
        if isinstance(cmd, str) and HOOK_COMMAND_MARKER in cmd:
            return True
    return False


def _our_matcher_group(event: str) -> dict:
    """Build the matcher-group dict to insert for a given event."""
    return {
        "matcher": {},
        "hooks": [
            {
                "type": "command",
                "command": _command_for(event),
            }
        ],
    }


def _read_settings(path: Path) -> tuple[dict, bool]:
    """Read settings.json. Returns (parsed_dict, file_existed_before).

    Raises ValueError if the file exists but isn't valid JSON — refuse to
    silently clobber a hand-edited file.
    """
    if not path.exists():
        return {}, False
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}, True
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Cannot parse {path}: {e}. Refusing to overwrite — please fix the "
            f"JSON manually, then re-run."
        )
    if not isinstance(data, dict):
        raise ValueError(f"{path} top-level must be a JSON object, got {type(data).__name__}")
    return data, True


def _write_settings(path: Path, data: dict) -> None:
    """Atomic write: tmpfile + fsync + rename. Backs up the prior file
    contents to ``settings.json.bak`` before clobbering."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        except OSError:
            pass  # best-effort backup; don't fail the install over it

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


def _ensure_event_array(settings: dict, event: str) -> list:
    """Ensure ``settings.hooks.<event>`` exists as a list and return it."""
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"settings.hooks must be a JSON object, got {type(hooks).__name__}")
    arr = hooks.setdefault(event, [])
    if not isinstance(arr, list):
        raise ValueError(
            f"settings.hooks.{event} must be a JSON array, got {type(arr).__name__}"
        )
    return arr


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------
def install(settings_path: Path = DEFAULT_SETTINGS_PATH) -> str:
    """Insert (or refresh) Graphite's hook entries in ``settings.json``.

    Idempotent: if our entries are already present, they're replaced with a
    fresh copy (handy if ``sys.executable`` changed because the user moved
    venvs). Other tools' hooks are preserved.
    """
    settings, existed = _read_settings(settings_path)

    for event in EVENTS:
        arr = _ensure_event_array(settings, event)
        # Drop any prior entries of ours, then append a fresh one.
        arr[:] = [g for g in arr if not _is_ours(g)]
        arr.append(_our_matcher_group(event))

    _write_settings(settings_path, settings)
    verb = "Updated" if existed else "Created"
    return f"{verb} {settings_path} with hooks for: {', '.join(EVENTS)}."


def uninstall(settings_path: Path = DEFAULT_SETTINGS_PATH) -> str:
    """Remove Graphite's hook entries from ``settings.json``. No-op if not
    installed or settings file doesn't exist. Other tools' hooks are
    untouched."""
    if not settings_path.exists():
        return f"{settings_path} does not exist — nothing to uninstall."

    settings, _ = _read_settings(settings_path)
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        return f"{settings_path} has no `hooks` map — nothing to uninstall."

    removed = 0
    for event in EVENTS:
        arr = hooks.get(event)
        if not isinstance(arr, list):
            continue
        before = len(arr)
        arr[:] = [g for g in arr if not _is_ours(g)]
        removed += before - len(arr)
        # Tidy: if our removal left an empty array, drop the key.
        if not arr:
            del hooks[event]

    # If `hooks` itself ended up empty, drop it for cleanliness.
    if not hooks:
        settings.pop("hooks", None)

    if removed == 0:
        return "No Graphite hook entries found — nothing to remove."

    _write_settings(settings_path, settings)
    return f"Removed {removed} Graphite hook entr{'y' if removed == 1 else 'ies'} from {settings_path}."


def status(settings_path: Path = DEFAULT_SETTINGS_PATH) -> HooksStatus:
    """Report which of our hooks are wired + archive counts + daemon state."""
    settings_exists = settings_path.exists()
    installed_events: list[str] = []

    if settings_exists:
        try:
            settings, _ = _read_settings(settings_path)
        except ValueError:
            settings = {}
        hooks = settings.get("hooks", {}) if isinstance(settings, dict) else {}
        if isinstance(hooks, dict):
            for event in EVENTS:
                arr = hooks.get(event, [])
                if isinstance(arr, list) and any(_is_ours(g) for g in arr):
                    installed_events.append(event)

    # Archive count is cheap to compute and useful for "did anything actually
    # capture" diagnostics.
    archive_dir = Path.home() / ".graphite" / "archive" / "sessions"
    archived_sessions = 0
    if archive_dir.is_dir():
        archived_sessions = sum(1 for p in archive_dir.iterdir() if p.suffix == ".jsonl")

    # Daemon reachability is a soft check.
    daemon_reachable = False
    try:
        from graphite.client import DaemonUnavailable, GraphiteClient
        with GraphiteClient(timeout_s=1.0) as c:
            c.ping()
            daemon_reachable = True
    except Exception:
        daemon_reachable = False

    if not installed_events:
        msg = f"Hooks not installed. Run `graphite hooks install` to wire them into {settings_path}."
    elif len(installed_events) < len(EVENTS):
        missing = sorted(set(EVENTS) - set(installed_events))
        msg = (
            f"Partial install: {', '.join(installed_events)} wired; "
            f"{', '.join(missing)} missing. Re-run `graphite hooks install`."
        )
    else:
        msg = (
            f"All hooks installed. {archived_sessions} session(s) in archive. "
            f"Daemon: {'reachable' if daemon_reachable else 'unreachable — start with `graphite daemon start`'}."
        )

    return HooksStatus(
        settings_path=settings_path,
        settings_exists=settings_exists,
        installed_events=installed_events,
        archived_sessions=archived_sessions,
        daemon_reachable=daemon_reachable,
        message=msg,
    )
