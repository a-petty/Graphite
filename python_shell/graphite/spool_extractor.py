"""Batch extractor — drains pending spool fragments into the graph.

Phase 2b's other half. PR E gave us a durable spool; this module is what
turns spool rows into entities and edges by feeding them through the
existing three-pass ``IngestionPipeline``.

Design:

  * **Per-source synthetic documents.** Fragments sharing a ``source_id``
    are concatenated (in ``created_at`` order) into one document so the
    tagger sees coherent narrative context. Different sources get
    different documents — no cross-source bleed.
  * **One batch_id per pipeline call.** All fragments fed into the same
    ``ingest_text`` invocation share a ``batch_id`` so audit queries can
    reconstruct "which fragments produced this entity."
  * **Failure isolation.** A pipeline exception marks only that group's
    fragments as ``failed``; sibling groups in the same batch keep going.
    The next batch run picks up where this one left off.
  * **No retries inside the extractor.** A failed group stays ``failed``
    in the spool until ``retry_failed`` flips it back to ``pending``.
    Automatic retry would risk loops on a permanently malformed input.
"""

from __future__ import annotations

import logging
import uuid
from typing import Callable, Optional

from graphite.spool import Fragment, Spool

logger = logging.getLogger(__name__)


PipelineFactory = Callable[[], "object"]
"""Zero-arg callable returning an IngestionPipeline. Late-bound so the
extractor doesn't import the pipeline module at definition time."""


class BatchExtractor:
    """Pulls a batch of pending fragments and feeds each per-source group
    through the ingestion pipeline. Returns a structured summary."""

    def __init__(self, spool: Spool, pipeline_factory: PipelineFactory):
        self._spool = spool
        self._pipeline_factory = pipeline_factory

    def extract_batch(
        self,
        batch_size_limit: int = 50,
        source_filter: Optional[str] = None,
    ) -> dict:
        """Claim up to ``batch_size_limit`` pending fragments, run them
        through the pipeline grouped by source, and return a summary.

        Args:
            batch_size_limit: Cap on fragments to claim this run.
            source_filter: If set, only claim fragments matching this
                ``source_id``. Used by ``flush_spool`` for per-session
                drains on hook events.

        Returns:
            Dict with ``claimed``, ``groups``, ``succeeded``,
            ``failed``, and ``errors``.
        """
        fragments = self._spool.claim_batch(
            limit=batch_size_limit,
            source_filter=source_filter,
        )
        summary: dict = {
            "claimed": len(fragments),
            "groups": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
        }
        if not fragments:
            return summary

        groups = _group_by_source(fragments)
        summary["groups"] = len(groups)

        # Lazy-construct pipeline once for the whole batch — these are
        # heavyweight (LLM client, tokenizer); we don't want N copies.
        try:
            pipeline = self._pipeline_factory()
        except Exception as e:
            # Pipeline construction failure: bounce every claimed fragment
            # back to pending so the next batch retries with (hopefully)
            # a working pipeline.
            ids = [f.id for f in fragments]
            self._spool.mark_failed(ids, error=f"pipeline init failed: {e}")
            summary["failed"] = len(ids)
            summary["errors"].append(str(e))
            logger.exception("Batch extractor: pipeline construction failed")
            return summary

        for source_id, group in groups.items():
            batch_id = f"batch-{uuid.uuid4().hex[:12]}"
            ids = [f.id for f in group]
            text = _concatenate(group)
            category = group[0].category  # all rows in a group share category

            try:
                result = pipeline.ingest_text(
                    text=text,
                    source_id=source_id,
                    memory_category=category,
                )
            except Exception as e:
                self._spool.mark_failed(ids, error=str(e), batch_id=batch_id)
                summary["failed"] += len(ids)
                summary["errors"].append(f"{source_id}: {e}")
                logger.exception("Batch extractor: pipeline raised on %s", source_id)
                continue

            if getattr(result, "status", None) == "failed":
                err = "; ".join(getattr(result, "errors", [])) or "pipeline reported failure"
                self._spool.mark_failed(ids, error=err, batch_id=batch_id)
                summary["failed"] += len(ids)
                summary["errors"].append(f"{source_id}: {err}")
                logger.warning("Batch extractor: pipeline failed on %s — %s", source_id, err)
                continue

            # status == "complete" or "partial" — both mean we got far
            # enough to consider the fragment processed. ``partial`` means
            # some chunks failed to tag but the doc was extracted; we'd
            # rather record the partial result than retry indefinitely.
            self._spool.mark_extracted(ids, batch_id=batch_id)
            summary["succeeded"] += len(ids)
            logger.info(
                "Batch extractor: %s -> %d entities, %d edges (%d fragments)",
                source_id,
                getattr(result, "entities_created", 0) + getattr(result, "entities_linked", 0),
                getattr(result, "edges_created", 0),
                len(ids),
            )

        return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _group_by_source(fragments: list[Fragment]) -> dict[str, list[Fragment]]:
    """Group fragments by source_id while preserving insertion order
    within each group (already source-then-created_at-sorted by claim)."""
    groups: dict[str, list[Fragment]] = {}
    for f in fragments:
        groups.setdefault(f.source_id, []).append(f)
    return groups


def _concatenate(fragments: list[Fragment]) -> str:
    """Join fragment texts with a blank line separator. Blank-line
    separation matches what the structural parser uses to find paragraph
    breaks, so the parser will see each fragment as its own structural
    unit even after concatenation."""
    return "\n\n".join(f.text for f in fragments)
