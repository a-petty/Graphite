"""Tests for the overflow reconciler (PR γ of OpenClaw bring-up).

The overflow path is the durability backstop for external capture
agents (the OpenClaw JS plugin, future MCP-side agents) when they
cannot reach the daemon at the moment of capture. Two contracts:

  1. **Idempotent.** Files whose content hash matches an existing
     graph document are skipped without spool writes. Re-replay of the
     same overflow file is a no-op.
  2. **Surgical quarantine.** Unparseable files move to ``failed/``;
     successfully replayed files move to ``processed/``. Neither
     replay nor bad data ever leaves stale entries in the live overflow
     directory.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from graphite import overflow_reconciler


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeSpool:
    def __init__(self):
        self.added: list[dict] = []

    def add(self, text, source_id, *, category="Episodic", project=None, entity_hints=None):
        self.added.append({
            "text": text, "source_id": source_id, "category": category,
            "project": project, "entity_hints": entity_hints,
        })
        return len(self.added)


class FakeKG:
    def __init__(self, hashes: dict | None = None):
        self._hashes = hashes or {}

    def get_document_hash(self, source_id):
        return self._hashes.get(source_id)


def _write_overflow(
    overflow_dir: Path,
    name: str,
    *,
    source_id: str = "openclaw://main/sess-001",
    text: str = "User: hi\nAssistant: hello",
    category: str = "Episodic",
    project: str | None = "TestProject",
    version: int = 1,
    extra: dict | None = None,
) -> Path:
    overflow_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "version": version,
        "source_id": source_id,
        "text": text,
        "category": category,
        "project": project,
        "captured_at": int(time.time()),
    }
    if extra:
        body.update(extra)
    path = overflow_dir / name
    path.write_text(json.dumps(body))
    return path


def _content_hash_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------
class TestEmpty:
    def test_missing_directory_returns_zero_summary(self, tmp_path):
        spool, kg = FakeSpool(), FakeKG()
        s = overflow_reconciler.reconcile_overflow(
            tmp_path / "missing", spool=spool, kg=kg,
        )
        assert s["scanned"] == 0
        assert s["replayed"] == 0
        assert spool.added == []

    def test_empty_directory_returns_zero_summary(self, tmp_path):
        (tmp_path / "of").mkdir()
        spool, kg = FakeSpool(), FakeKG()
        s = overflow_reconciler.reconcile_overflow(tmp_path / "of", spool=spool, kg=kg)
        assert s["scanned"] == 0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestHappyPath:
    def test_single_overflow_file_replayed_into_spool(self, tmp_path):
        of_dir = tmp_path / "of"
        _write_overflow(of_dir, "1.json")
        spool, kg = FakeSpool(), FakeKG()

        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)

        assert summary["scanned"] == 1
        assert summary["replayed"] == 1
        assert summary["already_indexed"] == 0
        assert len(spool.added) == 1
        assert spool.added[0]["source_id"] == "openclaw://main/sess-001"
        assert spool.added[0]["text"].startswith("User: hi")

    def test_replayed_file_moves_to_processed(self, tmp_path):
        of_dir = tmp_path / "of"
        path = _write_overflow(of_dir, "1.json")
        spool, kg = FakeSpool(), FakeKG()

        overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)

        assert not path.exists()
        assert (of_dir / "processed" / "1.json").exists()

    def test_processed_subdir_not_re_walked(self, tmp_path):
        """A second reconcile over the same directory must not pick up
        files we already moved to processed/."""
        of_dir = tmp_path / "of"
        _write_overflow(of_dir, "1.json")
        spool, kg = FakeSpool(), FakeKG()

        first = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)
        second = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)

        assert first["replayed"] == 1
        assert second["scanned"] == 0

    def test_multiple_files_all_replayed_in_one_pass(self, tmp_path):
        of_dir = tmp_path / "of"
        for i in range(3):
            _write_overflow(of_dir, f"{i}.json", source_id=f"src://x/{i}")
        spool, kg = FakeSpool(), FakeKG()

        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)
        assert summary["scanned"] == 3
        assert summary["replayed"] == 3
        assert {a["source_id"] for a in spool.added} == {f"src://x/{i}" for i in range(3)}


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
class TestIdempotency:
    def test_skipped_when_hash_matches_existing_document(self, tmp_path):
        of_dir = tmp_path / "of"
        text = "previously captured"
        _write_overflow(of_dir, "1.json", source_id="src://x", text=text)

        kg = FakeKG(hashes={"src://x": _content_hash_of(text)})
        spool = FakeSpool()

        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)
        assert summary["already_indexed"] == 1
        assert summary["replayed"] == 0
        assert spool.added == []
        # Already-indexed files still move to processed/ so we don't
        # re-scan them on the next reconcile.
        assert (of_dir / "processed" / "1.json").exists()

    def test_replayed_when_hash_differs(self, tmp_path):
        of_dir = tmp_path / "of"
        _write_overflow(of_dir, "1.json", source_id="src://x", text="new content")

        kg = FakeKG(hashes={"src://x": "stale-hash-from-prior-run"})
        spool = FakeSpool()

        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)
        assert summary["replayed"] == 1
        assert summary["already_indexed"] == 0


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------
class TestQuarantine:
    def test_malformed_json_quarantined(self, tmp_path):
        of_dir = tmp_path / "of"
        of_dir.mkdir()
        bad = of_dir / "bad.json"
        bad.write_text("{this is not json")

        spool, kg = FakeSpool(), FakeKG()
        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)

        assert summary["scanned"] == 1
        assert summary["skipped_unparseable"] == 1
        assert summary["replayed"] == 0
        assert not bad.exists()
        assert (of_dir / "failed" / "bad.json").exists()

    def test_unsupported_version_quarantined(self, tmp_path):
        of_dir = tmp_path / "of"
        _write_overflow(of_dir, "v99.json", version=99)

        spool, kg = FakeSpool(), FakeKG()
        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)
        assert summary["skipped_unparseable"] == 1
        assert (of_dir / "failed" / "v99.json").exists()

    def test_missing_required_fields_quarantined(self, tmp_path):
        of_dir = tmp_path / "of"
        of_dir.mkdir()
        # No source_id, no text.
        (of_dir / "incomplete.json").write_text(
            json.dumps({"version": 1, "category": "Episodic"})
        )
        spool, kg = FakeSpool(), FakeKG()
        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)
        assert summary["skipped_unparseable"] == 1

    def test_invalid_category_falls_back_to_episodic(self, tmp_path):
        of_dir = tmp_path / "of"
        _write_overflow(of_dir, "1.json", category="Bogus")
        spool, kg = FakeSpool(), FakeKG()
        overflow_reconciler.reconcile_overflow(of_dir, spool=spool, kg=kg)
        assert spool.added[0]["category"] == "Episodic"

    def test_spool_add_failure_marks_file_failed(self, tmp_path):
        of_dir = tmp_path / "of"
        _write_overflow(of_dir, "1.json")

        class BrokenSpool:
            def add(self, *args, **kwargs):
                raise RuntimeError("disk full")

        kg = FakeKG()
        summary = overflow_reconciler.reconcile_overflow(of_dir, spool=BrokenSpool(), kg=kg)

        assert summary["failed"] == 1
        assert (of_dir / "failed" / "1.json").exists()
