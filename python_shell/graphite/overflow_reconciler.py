"""Overflow reconciler — backfills `remember` captures that missed the live path.

Phase 2c PR γ. The OpenClaw plugin (and any future MCP-side capture agent)
can drop a small JSON file at ``~/.graphite/spool_overflow/`` when the
daemon socket is unreachable. On the next daemon start — or on demand —
this module replays each file via the daemon's normal ``remember`` API,
then files them away under ``spool_overflow/processed/``.

We intentionally do NOT push these straight into the spool. Going through
``Spool.add`` keeps the auto-trigger threshold and the
``flush_spool(source_filter=...)`` handles working uniformly for both
live captures and replayed ones.

Overflow file format (v1):
    {
        "version": 1,
        "source_id": "openclaw://<agent>/<session>",
        "text": "<full text>",
        "category": "Episodic" | "Semantic" | "Procedural",
        "project": "<optional>",
        "entity_hints": ["..."]?,
        "captured_at": <unix_seconds>
    }

Idempotency: each replay is gated by ``kg.get_document_hash(source_id)``.
A re-replay of the same content is a no-op at the graph layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_OVERFLOW_DIR = Path.home() / ".graphite" / "spool_overflow"
PROCESSED_SUBDIR = "processed"
FAILED_SUBDIR = "failed"

CURRENT_FORMAT_VERSION = 1


@dataclass
class OverflowFile:
    path: Path
    source_id: str
    text: str
    category: str
    project: Optional[str]
    entity_hints: Optional[list[str]]
    captured_at: int


def _read_overflow(path: Path) -> Optional[OverflowFile]:
    """Parse one overflow JSON. Returns None on any failure — caller
    quarantines unparseable files instead of crashing the reconciler."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Overflow reconciler: cannot read %s (%s)", path, e)
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Overflow reconciler: malformed JSON in %s (%s)", path, e)
        return None
    if not isinstance(obj, dict):
        logger.warning("Overflow reconciler: %s does not contain a JSON object", path)
        return None
    if obj.get("version") != CURRENT_FORMAT_VERSION:
        logger.warning(
            "Overflow reconciler: skipping %s (unsupported version %r)",
            path, obj.get("version"),
        )
        return None

    source_id = obj.get("source_id")
    text = obj.get("text")
    if not isinstance(source_id, str) or not source_id:
        logger.warning("Overflow reconciler: %s missing source_id", path)
        return None
    if not isinstance(text, str) or not text.strip():
        logger.warning("Overflow reconciler: %s missing text", path)
        return None

    category = obj.get("category", "Episodic")
    if category not in ("Episodic", "Semantic", "Procedural"):
        category = "Episodic"

    project = obj.get("project")
    if project is not None and not isinstance(project, str):
        project = None

    hints = obj.get("entity_hints")
    if hints is not None and (
        not isinstance(hints, list) or not all(isinstance(h, str) for h in hints)
    ):
        hints = None

    captured_at = obj.get("captured_at")
    if not isinstance(captured_at, int):
        captured_at = int(time.time())

    return OverflowFile(
        path=path,
        source_id=source_id,
        text=text,
        category=category,
        project=project,
        entity_hints=hints,
        captured_at=captured_at,
    )


def _content_hash(of: OverflowFile) -> str:
    h = hashlib.sha256()
    h.update(of.text.encode("utf-8"))
    return h.hexdigest()


def reconcile_overflow(
    overflow_dir: Path,
    *,
    spool,                   # graphite.spool.Spool
    kg,                      # graphite.semantic_engine.PyKnowledgeGraph (or proxy)
) -> dict:
    """Drain the overflow directory: each file becomes a fresh spool row
    (and stays one fragment per file, regardless of internal length).
    Files we successfully replay move to ``processed/``; files that fail
    move to ``failed/`` so the user can triage. Returns a summary dict."""
    summary = {
        "overflow_dir": str(overflow_dir),
        "scanned": 0,
        "replayed": 0,
        "already_indexed": 0,
        "skipped_unparseable": 0,
        "failed": 0,
    }
    if not overflow_dir.is_dir():
        return summary

    processed_dir = overflow_dir / PROCESSED_SUBDIR
    failed_dir = overflow_dir / FAILED_SUBDIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    for jpath in sorted(overflow_dir.glob("*.json")):
        if jpath.parent != overflow_dir:
            continue  # don't recurse into processed/ or failed/
        summary["scanned"] += 1

        of = _read_overflow(jpath)
        if of is None:
            try:
                shutil.move(str(jpath), str(failed_dir / jpath.name))
            except OSError:
                pass
            summary["skipped_unparseable"] += 1
            continue

        # Idempotency: skip if the graph already has a doc hash matching
        # this file's content. We use the same SHA256-of-bytes that the
        # ingestion pipeline uses for content-hash dedup.
        try:
            existing = kg.get_document_hash(of.source_id)
        except Exception:
            existing = None
        if existing and existing == _content_hash(of):
            summary["already_indexed"] += 1
            try:
                shutil.move(str(jpath), str(processed_dir / jpath.name))
            except OSError:
                pass
            continue

        try:
            spool.add(
                text=of.text,
                source_id=of.source_id,
                category=of.category,
                project=of.project,
                entity_hints=of.entity_hints,
            )
        except Exception as e:
            logger.warning("Overflow reconciler: spool.add failed for %s: %s", jpath, e)
            try:
                shutil.move(str(jpath), str(failed_dir / jpath.name))
            except OSError:
                pass
            summary["failed"] += 1
            continue

        summary["replayed"] += 1
        try:
            shutil.move(str(jpath), str(processed_dir / jpath.name))
        except OSError as e:
            logger.warning("Overflow reconciler: replay succeeded but archive move failed for %s: %s", jpath, e)

    if summary["scanned"] > 0:
        logger.info(
            "Overflow reconcile: scanned=%d replayed=%d already=%d unparseable=%d failed=%d",
            summary["scanned"], summary["replayed"], summary["already_indexed"],
            summary["skipped_unparseable"], summary["failed"],
        )
    return summary
