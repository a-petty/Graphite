"""Tests for Phase 6: Incremental document update handling.

Covers:
- Content hash computation (deterministic, stable)
- update_document: unchanged detection, remove + re-ingest
- remove_document: cascade cleanup, shared entities
- ingest_file stores hash
- Edge cases: nonexistent file, already-gone document
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex.config import CortexConfig
from cortex.ingestion.pipeline import (
    DocumentUpdateResult,
    IngestionPipeline,
    IngestionResult,
    _compute_file_hash,
)
from tests.test_memory_context import FakeKnowledgeGraph


# ═══════════════════════════════════════════════════════════════════════════════
# Hash Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestContentHash:
    def test_content_hash_stable(self, tmp_path):
        """SHA-256 hash is deterministic — same content gives same hash."""
        f = tmp_path / "test.md"
        f.write_text("# Meeting\nAlice and Bob discussed Rust.\n")

        hash1 = _compute_file_hash(f)
        hash2 = _compute_file_hash(f)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_content_hash_changes(self, tmp_path):
        """Different content produces different hash."""
        f = tmp_path / "test.md"
        f.write_text("Version 1")
        hash1 = _compute_file_hash(f)

        f.write_text("Version 2")
        hash2 = _compute_file_hash(f)
        assert hash1 != hash2


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Incremental Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_pipeline_with_fake_kg(tmp_path) -> tuple:
    """Create an IngestionPipeline with a FakeKnowledgeGraph and mock LLM.

    Returns (pipeline, fake_kg, file_path).
    """
    kg = FakeKnowledgeGraph()
    config = CortexConfig()
    config.memory_root = tmp_path

    # Create a test document
    doc = tmp_path / "meetings" / "standup.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# Standup\n\nAlice discussed the Rust migration.\n")

    # Mock LLM client — we won't actually call it in most tests
    mock_llm = MagicMock()

    pipeline = IngestionPipeline(
        knowledge_graph=kg,
        llm_client=mock_llm,
        config=config,
    )

    return pipeline, kg, doc


class TestUpdateDocumentUnchanged:
    def test_update_document_unchanged(self, tmp_path):
        """Same hash returns action='unchanged'."""
        pipeline, kg, doc = _make_pipeline_with_fake_kg(tmp_path)

        # Pre-set the hash to match file content
        content_hash = _compute_file_hash(doc)
        kg.set_document_hash(str(doc), content_hash)

        result = pipeline.update_document(doc)
        assert result.action == "unchanged"
        assert result.chunks_removed == 0


class TestUpdateDocumentChanged:
    def test_update_document_changed(self, tmp_path):
        """Changed hash triggers remove + re-ingest."""
        pipeline, kg, doc = _make_pipeline_with_fake_kg(tmp_path)

        # Set an old hash that doesn't match current content
        kg.set_document_hash(str(doc), "old_hash_that_doesnt_match")

        # Pre-populate some fake data as if the doc was previously ingested
        kg.add_test_entity(
            "e-alice", "Alice", "Person",
            source_documents=[str(doc)],
        )
        kg.add_test_chunk(
            "c1", "Alice discussed the migration.",
            source_document=str(doc),
            tags=["e-alice"],
        )

        # Mock ingest_file to simulate re-ingestion
        mock_ingest = MagicMock(return_value=IngestionResult(
            source_document=str(doc),
            status="complete",
            chunks_tagged=2,
            entities_created=1,
            edges_created=1,
            duration_seconds=0.1,
        ))
        pipeline.ingest_file = mock_ingest

        result = pipeline.update_document(doc)
        assert result.action == "updated"
        assert result.chunks_removed == 1  # c1 removed
        assert result.entities_removed == 1  # e-alice removed (sole source)
        assert result.ingestion_result is not None
        assert result.ingestion_result.status == "complete"
        mock_ingest.assert_called_once_with(doc)


class TestUpdateDocumentNonexistent:
    def test_update_document_nonexistent(self, tmp_path):
        """Non-existent file returns action='failed'."""
        pipeline, kg, _ = _make_pipeline_with_fake_kg(tmp_path)

        result = pipeline.update_document(tmp_path / "nonexistent.md")
        assert result.action == "failed"
        assert any("does not exist" in e for e in result.errors)


class TestRemoveDocument:
    def test_remove_document(self, tmp_path):
        """Remove cleans all artifacts and hash."""
        pipeline, kg, doc = _make_pipeline_with_fake_kg(tmp_path)

        # Simulate previously ingested data
        doc_str = str(doc)
        kg.set_document_hash(doc_str, "somehash")
        kg.add_test_entity(
            "e-alice", "Alice", "Person",
            source_documents=[doc_str],
        )
        kg.add_test_chunk(
            "c1", "Alice discussed.",
            source_document=doc_str,
            tags=["e-alice"],
        )

        result = pipeline.remove_document(doc)
        assert result.action == "removed"
        assert result.chunks_removed == 1
        assert result.entities_removed == 1

        # Hash should be cleaned up
        assert kg.get_document_hash(doc_str) is None

    def test_remove_document_shared_entity(self, tmp_path):
        """Entity from two documents survives removal of one."""
        pipeline, kg, doc = _make_pipeline_with_fake_kg(tmp_path)

        doc_str = str(doc)
        other_doc = str(tmp_path / "other.md")

        kg.add_test_entity(
            "e-shared", "Shared Entity", "Concept",
            source_documents=[doc_str, other_doc],
        )
        kg.add_test_chunk(
            "c1", "Shared entity appears here.",
            source_document=doc_str,
            tags=["e-shared"],
        )
        kg.add_test_chunk(
            "c2", "Shared entity also here.",
            source_document=other_doc,
            tags=["e-shared"],
        )

        result = pipeline.remove_document(doc)
        assert result.action == "removed"
        assert result.chunks_removed == 1
        assert result.entities_removed == 0  # shared entity survives
        assert result.entities_updated == 1

        # Entity still exists
        entity_json = kg.get_entity("e-shared")
        assert entity_json is not None
        entity = json.loads(entity_json)
        assert doc_str not in entity["source_documents"]
        assert other_doc in entity["source_documents"]


class TestRemoveDocumentAlreadyGone:
    def test_remove_document_already_gone(self, tmp_path):
        """Removing a document that was never ingested is a no-op."""
        pipeline, kg, _ = _make_pipeline_with_fake_kg(tmp_path)

        result = pipeline.remove_document(tmp_path / "never_existed.md")
        assert result.action == "removed"
        assert result.chunks_removed == 0
        assert result.edges_removed == 0
        assert result.entities_removed == 0


class TestIngestStoresHash:
    def test_ingest_stores_hash(self, tmp_path):
        """ingest_file stores the content hash after successful ingestion."""
        pipeline, kg, doc = _make_pipeline_with_fake_kg(tmp_path)

        # Mock the pipeline stages to avoid real LLM calls
        pipeline.parser.parse = MagicMock(return_value=[])

        result = pipeline.ingest_file(doc)
        # Even with 0 chunks, hash should be stored
        stored_hash = kg.get_document_hash(str(doc))
        assert stored_hash is not None
        assert stored_hash == _compute_file_hash(doc)
