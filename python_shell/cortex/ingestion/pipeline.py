"""Three-pass document ingestion orchestrator.

Coordinates the tag-and-index pipeline:
1. Structural parse — deterministic chunking (no LLM)
2. Classify — LLM chunk classification (decision, discussion, etc.)
3. Tag — LLM entity tagging + disambiguation

Filler chunks are discarded between Pass 2 and Pass 3.
Co-occurrence edges are created for all entity pairs within each chunk.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

from cortex.config import CortexConfig
from cortex.extraction.classifier import ChunkClassifier
from cortex.extraction.structural_parser import StructuralParser
from cortex.extraction.tagger import EntityTagger
from cortex.ingestion.categorizer import categorize_document
from cortex.llm import LLMClient

logger = logging.getLogger(__name__)


# Map Python-side chunk_type strings to the Rust-side ChunkType enum names
_CHUNK_TYPE_MAP = {
    "decision": "Decision",
    "discussion": "Discussion",
    "action_item": "ActionItem",
    "status_update": "StatusUpdate",
    "preference": "Preference",
    "background": "Background",
}

# Map Python-side entity_type strings to the Rust-side EntityType enum names
_ENTITY_TYPE_MAP = {
    "person": "Person",
    "project": "Project",
    "technology": "Technology",
    "organization": "Organization",
    "location": "Location",
    "decision": "Decision",
    "concept": "Concept",
}


@dataclass
class IngestionResult:
    """Summary of ingesting a single document."""

    source_document: str
    status: str = "pending"  # "complete" | "partial" | "failed" | "pending"
    chunks_total: int = 0
    chunks_classified: int = 0
    chunks_filler: int = 0
    chunks_tagged: int = 0
    chunks_failed: int = 0
    entities_created: int = 0
    entities_linked: int = 0
    edges_created: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class DocumentUpdateResult:
    """Summary of an incremental document update or removal."""

    source_document: str
    action: str = "unchanged"  # "updated" | "removed" | "unchanged" | "failed"
    chunks_removed: int = 0
    edges_removed: int = 0
    entities_removed: int = 0
    entities_updated: int = 0
    ingestion_result: Optional[IngestionResult] = None
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    h.update(file_path.read_bytes())
    return h.hexdigest()


class IngestionPipeline:
    """Orchestrates the three-pass ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline(knowledge_graph, llm_client)
        result = pipeline.ingest_file(Path("memory/meetings/standup.md"))
        results = pipeline.ingest_directory(Path("memory/"))
    """

    def __init__(
        self,
        knowledge_graph,
        llm_client: Optional[LLMClient] = None,
        embedding_manager=None,
        config: Optional[CortexConfig] = None,
    ):
        self.kg = knowledge_graph
        self.config = config or CortexConfig()

        self.parser = StructuralParser(config=self.config)
        self.classifier = (
            ChunkClassifier(llm_client, config=self.config)
            if llm_client
            else None
        )
        self.tagger = (
            EntityTagger(
                llm_client,
                knowledge_graph=knowledge_graph,
                embedding_manager=embedding_manager,
                config=self.config,
            )
            if llm_client
            else None
        )

    def ingest_file(self, file_path: Path) -> IngestionResult:
        """Ingest a single document file through the three-pass pipeline.

        Args:
            file_path: Path to the document to ingest.

        Returns:
            IngestionResult with statistics and status.
        """
        start_time = time.time()
        result = IngestionResult(source_document=str(file_path))

        # Read file
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            result.status = "failed"
            result.errors.append(f"Failed to read file: {e}")
            result.duration_seconds = time.time() - start_time
            return result

        if not text.strip():
            result.status = "complete"
            self._store_hash(file_path)
            result.duration_seconds = time.time() - start_time
            return result

        # Determine memory category
        memory_category = categorize_document(file_path, self.config.memory_root)

        # ── Pass 1: Structural Parse ──
        try:
            raw_chunks = self.parser.parse(
                text, str(file_path), memory_category
            )
        except Exception as e:
            result.status = "failed"
            result.errors.append(f"Pass 1 (structural parse) failed: {e}")
            result.duration_seconds = time.time() - start_time
            return result

        result.chunks_total = len(raw_chunks)

        if not raw_chunks:
            result.status = "complete"
            self._store_hash(file_path)
            result.duration_seconds = time.time() - start_time
            return result

        # ── Pass 2: Classification ──
        if self.classifier is None:
            result.status = "failed"
            result.errors.append("No LLM client provided — cannot classify")
            result.duration_seconds = time.time() - start_time
            return result

        try:
            classified_chunks, filler_chunks = self.classifier.classify_chunks(
                raw_chunks
            )
        except Exception as e:
            result.status = "failed"
            result.errors.append(f"Pass 2 (classification) failed: {e}")
            result.duration_seconds = time.time() - start_time
            return result

        result.chunks_classified = len(classified_chunks)
        result.chunks_filler = len(filler_chunks)

        if not classified_chunks:
            result.status = "complete"
            self._store_hash(file_path)
            result.duration_seconds = time.time() - start_time
            return result

        # ── Pass 3: Tagging ──
        if self.tagger is None:
            result.status = "failed"
            result.errors.append("No LLM client provided — cannot tag")
            result.duration_seconds = time.time() - start_time
            return result

        try:
            tagged_chunks = self.tagger.tag_chunks(classified_chunks)
        except Exception as e:
            result.status = "failed"
            result.errors.append(f"Pass 3 (tagging) failed: {e}")
            result.duration_seconds = time.time() - start_time
            return result

        result.chunks_tagged = len(tagged_chunks)
        result.chunks_failed = result.chunks_classified - result.chunks_tagged

        # ── Write to Graph ──
        try:
            write_stats = self._write_to_graph(
                tagged_chunks, str(file_path), memory_category
            )
            result.entities_created = write_stats["entities_created"]
            result.entities_linked = write_stats["entities_linked"]
            result.edges_created = write_stats["edges_created"]
        except Exception as e:
            result.status = "partial"
            result.errors.append(f"Graph write failed: {e}")
            result.duration_seconds = time.time() - start_time
            return result

        result.status = "complete" if result.chunks_failed == 0 else "partial"
        self._store_hash(file_path)
        result.duration_seconds = time.time() - start_time
        return result

    def ingest_directory(self, dir_path: Path) -> List[IngestionResult]:
        """Ingest all markdown files in a directory tree.

        Args:
            dir_path: Directory to recursively scan for .md files.

        Returns:
            List of IngestionResult, one per file.
        """
        results: List[IngestionResult] = []

        if not dir_path.is_dir():
            logger.error("Not a directory: %s", dir_path)
            return results

        md_files = sorted(dir_path.rglob("*.md"))
        logger.info("Found %d markdown files in %s", len(md_files), dir_path)

        for file_path in md_files:
            logger.info("Ingesting: %s", file_path)
            result = self.ingest_file(file_path)
            results.append(result)
            logger.info(
                "  → %s: %d chunks, %d entities, %d edges (%.1fs)",
                result.status,
                result.chunks_tagged,
                result.entities_created + result.entities_linked,
                result.edges_created,
                result.duration_seconds,
            )

        return results

    def _store_hash(self, file_path: Path) -> None:
        """Store content hash for a successfully processed file."""
        try:
            content_hash = _compute_file_hash(file_path)
            self.kg.set_document_hash(str(file_path), content_hash)
        except Exception as e:
            logger.warning("Failed to store document hash: %s", e)

    def update_document(self, file_path: Path) -> DocumentUpdateResult:
        """Update a previously-ingested document (remove old data + re-ingest).

        If the file content hasn't changed (same hash), returns action="unchanged".

        Args:
            file_path: Path to the document to update.

        Returns:
            DocumentUpdateResult with removal + re-ingestion stats.
        """
        start_time = time.time()
        doc_str = str(file_path)
        result = DocumentUpdateResult(source_document=doc_str)

        if not file_path.exists():
            result.action = "failed"
            result.errors.append(f"File does not exist: {file_path}")
            result.duration_seconds = time.time() - start_time
            return result

        # Check content hash
        try:
            new_hash = _compute_file_hash(file_path)
        except Exception as e:
            result.action = "failed"
            result.errors.append(f"Failed to compute file hash: {e}")
            result.duration_seconds = time.time() - start_time
            return result

        old_hash = self.kg.get_document_hash(doc_str)
        if old_hash is not None and old_hash == new_hash:
            result.action = "unchanged"
            result.duration_seconds = time.time() - start_time
            return result

        # Remove old artifacts
        try:
            removal_json = self.kg.remove_document(doc_str)
            removal = json.loads(removal_json)
            result.chunks_removed = removal.get("chunks_removed", 0)
            result.edges_removed = removal.get("edges_removed", 0)
            result.entities_removed = removal.get("entities_removed", 0)
            result.entities_updated = removal.get("entities_updated", 0)
        except Exception as e:
            logger.warning("Failed to remove old document data: %s", e)
            result.errors.append(f"Removal failed: {e}")

        # Re-ingest
        ingestion_result = self.ingest_file(file_path)
        result.ingestion_result = ingestion_result

        if ingestion_result.status == "failed":
            result.action = "failed"
            result.errors.extend(ingestion_result.errors)
        else:
            result.action = "updated"

        result.duration_seconds = time.time() - start_time
        return result

    def remove_document(self, file_path: Path) -> DocumentUpdateResult:
        """Remove all artifacts for a document from the knowledge graph.

        Args:
            file_path: Path to the document to remove.

        Returns:
            DocumentUpdateResult with removal stats.
        """
        start_time = time.time()
        doc_str = str(file_path)
        result = DocumentUpdateResult(source_document=doc_str)

        try:
            removal_json = self.kg.remove_document(doc_str)
            removal = json.loads(removal_json)
            result.chunks_removed = removal.get("chunks_removed", 0)
            result.edges_removed = removal.get("edges_removed", 0)
            result.entities_removed = removal.get("entities_removed", 0)
            result.entities_updated = removal.get("entities_updated", 0)
            result.action = "removed"
        except Exception as e:
            result.action = "failed"
            result.errors.append(f"Removal failed: {e}")

        # Clean up hash
        self.kg.remove_document_hash(doc_str)

        result.duration_seconds = time.time() - start_time
        return result

    def _write_to_graph(
        self,
        tagged_chunks: list,
        source_document: str,
        memory_category: str,
    ) -> Dict[str, int]:
        """Write tagged chunks to the knowledge graph.

        For each tagged chunk:
        1. Create or link entities
        2. Store the chunk with entity tags
        3. Create co-occurrence edges for all entity pairs

        Returns dict with counts: entities_created, entities_linked, edges_created.
        """
        entities_created = 0
        entities_linked = 0
        edges_created = 0

        # Track entity name → graph ID mapping for this ingestion
        entity_id_map: Dict[str, str] = {}

        rust_mem_category = memory_category  # Already "Episodic"/"Semantic"/"Procedural"

        for tagged in tagged_chunks:
            chunk = tagged.classified.raw
            chunk_type_str = tagged.classified.chunk_type
            rust_chunk_type = _CHUNK_TYPE_MAP.get(chunk_type_str, "Discussion")

            # 1. Ensure all entities exist in the graph
            chunk_entity_ids: List[str] = []

            for entity in tagged.entities:
                entity_key = entity.name.lower()

                if entity_key in entity_id_map:
                    # Already created/linked in this ingestion
                    chunk_entity_ids.append(entity_id_map[entity_key])
                    continue

                if not entity.is_new and entity.existing_entity_id:
                    # Link to existing entity
                    entity_id_map[entity_key] = entity.existing_entity_id
                    chunk_entity_ids.append(entity.existing_entity_id)
                    entities_linked += 1
                else:
                    # Create new entity
                    rust_entity_type = _ENTITY_TYPE_MAP.get(
                        entity.entity_type, "Concept"
                    )
                    entity_json = json.dumps(
                        {
                            "canonical_name": entity.name,
                            "entity_type": rust_entity_type,
                            "aliases": [],
                            "source_documents": [source_document],
                        }
                    )
                    try:
                        entity_id = self.kg.add_entity(entity_json)
                        entity_id_map[entity_key] = entity_id
                        chunk_entity_ids.append(entity_id)
                        entities_created += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to create entity '%s': %s",
                            entity.name,
                            e,
                        )
                        continue

            # 2. Store the chunk with entity tags
            chunk_json = json.dumps(
                {
                    "source_document": source_document,
                    "chunk_type": rust_chunk_type,
                    "memory_category": rust_mem_category,
                    "text": chunk.text,
                    "section_name": chunk.section_name,
                    "speaker": chunk.speaker,
                    "timestamp": chunk.timestamp,
                    "tags": chunk_entity_ids,
                }
            )
            try:
                chunk_id = self.kg.store_chunk(chunk_json)
            except Exception as e:
                logger.warning("Failed to store chunk: %s", e)
                continue

            # 3. Create co-occurrence edges for all entity pairs
            for id_a, id_b in combinations(chunk_entity_ids, 2):
                edge_json = json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "chunk_type": rust_chunk_type,
                        "memory_category": rust_mem_category,
                        "source_document": source_document,
                        "timestamp": chunk.timestamp,
                    }
                )
                try:
                    self.kg.add_cooccurrence(id_a, id_b, edge_json)
                    edges_created += 1
                except Exception as e:
                    logger.debug(
                        "Failed to create edge %s↔%s: %s",
                        id_a[:8],
                        id_b[:8],
                        e,
                    )

        return {
            "entities_created": entities_created,
            "entities_linked": entities_linked,
            "edges_created": edges_created,
        }

    def save_graph(self, save_path: str):
        """Save the knowledge graph to disk.

        Args:
            save_path: Path for the graph file.
        """
        self.kg.save(save_path)
        logger.info("Graph saved to %s", save_path)
