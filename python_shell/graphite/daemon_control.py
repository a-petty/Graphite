"""macOS launchd management for the graphited daemon.

Generates a LaunchAgent plist at ``~/Library/LaunchAgents/com.graphite.daemon.plist``
and talks to ``launchctl`` via subprocess. Deliberately shell-free: every
launchctl invocation is a direct argv list, and we never ``shell=True``.

Separate from ``daemon.py`` so the server doesn't import subprocess/plist
machinery at startup time.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from graphite.client import DaemonUnavailable, GraphiteClient

LABEL = "com.graphite.daemon"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_DIR = Path.home() / ".graphite" / "logs"
DEFAULT_SOCKET = Path.home() / ".graphite" / "daemon.sock"


@dataclass
class DaemonStatus:
    installed: bool
    running: bool
    pid: Optional[int]
    socket_present: bool
    reachable: bool
    message: str


def _graphited_path() -> Path:
    """Locate the ``graphited`` executable. In a venv this is
    ``<venv>/bin/graphited``. Fall back to ``python -m graphite.daemon``
    via ``sys.executable`` if the script isn't on PATH.
    """
    found = shutil.which("graphited")
    if found:
        return Path(found)
    # Fall back: the current Python interpreter + module invocation.
    return Path(sys.executable)


def _program_args() -> list[str]:
    """The ``ProgramArguments`` array for the plist."""
    path = _graphited_path()
    if path.name == "graphited":
        return [str(path)]
    # Python fallback: <sys.executable> -m graphite.daemon
    return [str(path), "-m", "graphite.daemon"]


def _build_plist_dict() -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    return {
        "Label": LABEL,
        "ProgramArguments": _program_args(),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(LOG_DIR / "daemon.out"),
        "StandardErrorPath": str(LOG_DIR / "daemon.err"),
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": {
            "PATH": env_path,
        },
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    """Run ``launchctl`` with the given args; capture output."""
    return subprocess.run(
        ["launchctl", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _gui_domain() -> str:
    """Return the launchctl GUI domain for the current user."""
    return f"gui/{os.getuid()}"


def _service_target() -> str:
    return f"{_gui_domain()}/{LABEL}"


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------
def install() -> str:
    """Write the plist and bootstrap it into launchd. Returns a status string."""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    plist_dict = _build_plist_dict()
    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist_dict, f)
    os.chmod(PLIST_PATH, 0o644)

    # If already loaded, bootout first so bootstrap picks up any plist changes.
    _launchctl("bootout", _service_target())

    result = _launchctl("bootstrap", _gui_domain(), str(PLIST_PATH))
    if result.returncode != 0:
        return (
            f"launchctl bootstrap failed (code {result.returncode}):\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}\n"
            f"Plist is at {PLIST_PATH} — you can inspect or remove it manually."
        )

    # Kick it into running now rather than waiting for RunAtLoad at next login.
    _launchctl("kickstart", "-k", _service_target())
    return f"Installed and started. Plist: {PLIST_PATH}. Logs: {LOG_DIR}/"


def uninstall() -> str:
    """Bootout the service and remove the plist."""
    _launchctl("bootout", _service_target())
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
    return f"Uninstalled. Removed {PLIST_PATH}."


def start() -> str:
    """Kickstart-restart the service. If it's not installed, install first."""
    if not PLIST_PATH.exists():
        return install()
    result = _launchctl("kickstart", "-k", _service_target())
    if result.returncode != 0:
        # Not bootstrapped yet — bootstrap it.
        bs = _launchctl("bootstrap", _gui_domain(), str(PLIST_PATH))
        if bs.returncode != 0:
            return (
                f"Could not start: {result.stderr.strip() or bs.stderr.strip()}\n"
                f"Try `graphite daemon install`."
            )
        _launchctl("kickstart", "-k", _service_target())
    return "Started."


def stop() -> str:
    """Bootout — fully removes the service from launchd. ``start`` will bring
    it back. This is the right stop for KeepAlive services; ``launchctl stop``
    alone would be followed by an immediate restart."""
    result = _launchctl("bootout", _service_target())
    if result.returncode != 0:
        return (
            f"Stop may have failed (code {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return "Stopped."


def restart() -> str:
    stop_msg = stop()
    start_msg = start()
    return f"{stop_msg}\n{start_msg}"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def _parse_pid_from_print() -> Optional[int]:
    """Ask launchctl print for the service and extract the PID if running."""
    result = _launchctl("print", _service_target())
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        # launchctl print format: `pid = 12345`
        if stripped.startswith("pid = "):
            try:
                return int(stripped.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _socket_path() -> Path:
    return DEFAULT_SOCKET


def status() -> DaemonStatus:
    installed = PLIST_PATH.exists()
    pid = _parse_pid_from_print() if installed else None
    sock_path = _socket_path()
    socket_present = sock_path.exists() and sock_path.is_socket()

    reachable = False
    if socket_present:
        try:
            client = GraphiteClient(socket_path=sock_path)
            client.ping()
            reachable = True
        except (DaemonUnavailable, OSError, socket.error):
            reachable = False

    if not installed:
        msg = f"Not installed. Run `graphite daemon install` to create {PLIST_PATH}."
    elif pid is None and not reachable:
        msg = "Installed but not currently running. `graphite daemon start` to launch it."
    elif pid is not None and reachable:
        msg = f"Running (pid {pid}), socket at {sock_path} responding."
    elif pid is not None and not reachable:
        msg = f"Running (pid {pid}) but socket at {sock_path} not responding — check logs at {LOG_DIR}/."
    else:
        msg = f"Socket reachable at {sock_path} but launchctl reports no pid — something is off."

    return DaemonStatus(
        installed=installed,
        running=pid is not None,
        pid=pid,
        socket_present=socket_present,
        reachable=reachable,
        message=msg,
    )
