"""Claude Code hook handler for conversation capture.

Lightweight handler invoked by Claude Code hooks (``PreCompact``, ``SessionEnd``).

What it does, in order:

  1. **Archive.** Copies the session JSONL into
     ``~/.graphite/archive/sessions/{session_id}.jsonl``. This is the
     durability layer — everything downstream can be reconstructed from the
     archive alone.
  2. **Enqueue** (SessionEnd only). Best-effort RPC to the graphited daemon
     asking it to ingest the archived file. If the daemon isn't running or
     the LLM isn't configured, this step silently fails — the backlog
     reconciler (PR D) will catch the file later.

Hard invariant: **this script never exits non-zero**. Claude Code's hook
runner treats a non-zero exit as "abort whatever was happening," which
would be awful UX for a best-effort capture side-effect. Every error path
logs and exits 0.

Hook configuration (add to ``~/.claude/settings.json``):

    {
      "hooks": {
        "PreCompact": [{
          "matcher": {},
          "hooks": [{
            "type": "command",
            "command": "python3 -m graphite.capture.hook_handler --event pre-compact"
          }]
        }],
        "SessionEnd": [{
          "matcher": {},
          "hooks": [{
            "type": "command",
            "command": "python3 -m graphite.capture.hook_handler --event session-end"
          }]
        }]
      }
    }

Prefer ``graphite hooks install`` (PR C) over editing settings.json by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("graphite.capture.hook_handler")

# Default archive + log locations — both live under ~/.graphite so the
# daemon and the reconciler can find them without config.
DEFAULT_ARCHIVE_DIR = Path.home() / ".graphite" / "archive" / "sessions"
DEFAULT_LOG_DIR = Path.home() / ".graphite" / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "hook.log"

# Total wall-clock budget for the daemon enqueue RPC. Hooks should feel
# instant to the user; we'd rather drop a session (and let the reconciler
# pick it up) than block Claude Code shutdown for more than a second or two.
DAEMON_TIMEOUT_S = 2.0


def _configure_logging(verbose: bool) -> None:
    """Emit logs to both stderr and a persistent file. The file log is the
    one the user actually reads when a capture silently failed —
    ``graphite hooks status`` (PR C) will point at it.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger("graphite")
    root.setLevel(level)

    # Avoid piling up handlers across repeat invocations within the same
    # Python session (can happen in tests).
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stderr = logging.StreamHandler(sys.stderr)
        stderr.setFormatter(logging.Formatter("%(levelname)s hook: %(message)s"))
        root.addHandler(stderr)

    try:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename) == DEFAULT_LOG_FILE
            for h in root.handlers
        ):
            fh = logging.FileHandler(DEFAULT_LOG_FILE)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s hook[%(process)d]: %(message)s"
            ))
            root.addHandler(fh)
    except OSError:
        # If we can't create the log dir (read-only $HOME? unusual), fall
        # back to stderr-only. Don't let this fail the hook.
        pass


def _read_hook_input() -> dict:
    """Read hook input JSON from stdin. Missing/malformed input is fine —
    we simply skip the work and exit 0."""
    try:
        raw = sys.stdin.read()
        if raw.strip():
            return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _extract_session_id(hook_input: dict) -> str:
    transcript_path = hook_input.get("transcript_path", "")
    if transcript_path:
        return Path(transcript_path).stem
    return hook_input.get("session_id", "") or ""


def _find_transcript(hook_input: dict) -> Optional[Path]:
    """Locate the JSONL on disk. Hook input may give us ``transcript_path``
    directly, or just a ``session_id`` we have to hunt for under
    ``~/.claude/projects/``.
    """
    transcript_path = hook_input.get("transcript_path", "")
    if transcript_path:
        path = Path(transcript_path)
        if path.exists():
            return path

    session_id = hook_input.get("session_id", "")
    cwd = hook_input.get("cwd", "")
    if session_id and cwd:
        claude_dir = Path.home() / ".claude" / "projects"
        if claude_dir.exists():
            for project_dir in claude_dir.iterdir():
                candidate = project_dir / f"{session_id}.jsonl"
                if candidate.exists():
                    return candidate

    return None


def _extract_project_name(hook_input: dict) -> Optional[str]:
    """Derive a project tag for the session from the hook payload.

    Preference: the basename of ``cwd`` (stable, human-recognizable).
    Fallback: decode it from the Claude Code project directory encoding
    in ``transcript_path`` (e.g. ``-Users-apetty-Dev-Graphite`` →
    ``Graphite``). Returns None if neither is available.
    """
    cwd = hook_input.get("cwd", "")
    if cwd:
        name = Path(cwd).name
        if name:
            return name

    transcript_path = hook_input.get("transcript_path", "")
    if transcript_path:
        # Claude stores transcripts under ~/.claude/projects/<encoded>/<id>.jsonl
        # where <encoded> replaces path separators with '-' (e.g.
        # -Users-apetty-Dev-Graphite). Take the last segment as project name.
        parent = Path(transcript_path).parent.name
        if parent.startswith("-"):
            # Drop leading dash; take final path segment.
            tail = parent.lstrip("-").split("-")[-1]
            return tail or None
    return None


def archive_transcript(
    hook_input: dict,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
) -> Optional[Path]:
    """Archive a conversation transcript. Returns the archive path, or
    ``None`` if nothing was found to archive. Dedupes by session_id —
    re-firing the hook during a long session just overwrites, which is
    safe because ingestion is content-hash-idempotent."""
    transcript = _find_transcript(hook_input)
    if transcript is None:
        logger.debug("No transcript file found in hook input")
        return None

    session_id = _extract_session_id(hook_input) or transcript.stem
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{session_id}.jsonl"

    try:
        shutil.copy2(str(transcript), str(dest))
        logger.info("Archived transcript: %s -> %s", transcript, dest)
        return dest
    except OSError as e:
        logger.error("Failed to archive transcript %s: %s", transcript, e)
        return None


def _try_enqueue_ingest(archive_path: Path, project: Optional[str]) -> bool:
    """Best-effort: ask the daemon to ingest ``archive_path`` and to flush
    any pending spool fragments accumulated during the session.

    Never raises. Returns True if the session enqueue succeeded, False
    otherwise (daemon down, LLM missing, timeout, etc.) — failures are
    logged. The spool flush is fired regardless on a best-effort basis.
    """
    try:
        # Import here so that hook_handler.main() doesn't pay the
        # GraphiteClient / socket import cost in the PreCompact path.
        from graphite.client import DaemonError, DaemonUnavailable, GraphiteClient
    except ImportError as e:
        logger.warning("graphite.client unavailable (%s) — skipping enqueue", e)
        return False

    try:
        with GraphiteClient(timeout_s=DAEMON_TIMEOUT_S) as c:
            result = c.enqueue_session_ingest(
                path=str(archive_path),
                project=project,
            )
            # Best-effort spool flush — drain anything Claude captured via
            # remember() during the session. Failures here don't change
            # the session-enqueue outcome; the size-based auto-trigger and
            # the manual `graphite spool flush` are both fallbacks.
            try:
                c.flush_spool()
            except Exception as flush_err:
                logger.info("Spool flush at session end failed (non-fatal): %s", flush_err)
        logger.info(
            "Enqueued %s for ingest (job_id=%s, queue_position=%s)",
            archive_path.name,
            result.get("job_id"),
            result.get("queue_position"),
        )
        return True
    except DaemonUnavailable as e:
        logger.info(
            "Daemon unreachable (%s) — session archived at %s; "
            "reconciler will pick it up on next daemon start.",
            e, archive_path,
        )
        return False
    except DaemonError as e:
        logger.warning(
            "Daemon rejected enqueue (code=%s): %s — reconciler can retry later.",
            e.code, e.message,
        )
        return False
    except Exception as e:
        logger.warning("Unexpected enqueue failure: %s", e)
        return False


def handle_session_end(
    hook_input: dict,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
) -> None:
    """Archive the session JSONL, then try to enqueue it for ingestion.

    The two steps are deliberately independent: if enqueue fails, the
    archive is still on disk and the reconciler will handle it later.
    """
    archive_path = archive_transcript(hook_input, archive_dir)
    if archive_path is None:
        return
    project = _extract_project_name(hook_input)
    _try_enqueue_ingest(archive_path, project)


def handle_pre_compact(
    hook_input: dict,
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
) -> None:
    """Archive the current transcript state. Does NOT enqueue — a compact
    isn't the end of the session, so ingesting now would shred a
    still-in-progress conversation. The archive copy is safety only."""
    archive_transcript(hook_input, archive_dir)


def main() -> int:
    """Entry point for hook invocation. Always exits 0."""
    parser = argparse.ArgumentParser(
        description="Graphite conversation capture hook handler"
    )
    parser.add_argument(
        "--event",
        choices=["pre-compact", "session-end"],
        required=True,
        help="Hook event type",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Archive directory (default: {DEFAULT_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        # argparse calls sys.exit() on --help or bad args; swallow that too.
        return 0

    _configure_logging(args.verbose)

    try:
        hook_input = _read_hook_input()
        if not hook_input:
            logger.debug("No hook input received (empty stdin)")
            return 0

        logger.debug(
            "Hook event: %s, input keys: %s",
            args.event, sorted(hook_input.keys()),
        )

        if args.event == "session-end":
            handle_session_end(hook_input, args.archive_dir)
        else:
            handle_pre_compact(hook_input, args.archive_dir)
    except Exception as e:
        # The contract is "never break Claude Code". Anything that bubbled
        # out of the normal paths lands here, gets logged, and we exit 0.
        logger.error("Unhandled exception in hook handler: %s", e, exc_info=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
