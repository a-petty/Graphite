"""Simple vector-only RAG baseline for comparison.

Uses the same chunking (Pass 1) and embedding model as Graphite,
but no LLM classification, no entity tagging, no graph traversal.
Just embed + cosine similarity.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from graphite.config import GraphiteConfig
from graphite.extraction.structural_parser import RawChunk, StructuralParser
from graphite.ingestion.categorizer import categorize_document

logger = logging.getLogger(__name__)


class SimpleRAGBaseline:
    """Vector-only retrieval: same chunks + same embeddings, no graph."""

    def __init__(self, corpus_dir: Path, config: GraphiteConfig = None):
        self.corpus_dir = corpus_dir
        self.config = config or GraphiteConfig()
        self.parser = StructuralParser(config=self.config)
        self.chunks: List[RawChunk] = []
        self.chunk_embeddings: List[np.ndarray] = []
        self._embedding_manager = None

    @property
    def embedding_manager(self):
        """Lazy-load embedding manager."""
        if self._embedding_manager is None:
            from graphite.embeddings import EmbeddingManager
            self._embedding_manager = EmbeddingManager()
        return self._embedding_manager

    def build_index(self) -> None:
        """Parse all .md files in corpus_dir and embed all chunks."""
        md_files = sorted(self.corpus_dir.rglob("*.md"))
        if not md_files:
            logger.warning(f"No .md files found in {self.corpus_dir}")
            return

        all_chunks = []
        for md_file in md_files:
            text = md_file.read_text()
            rel_path = str(md_file.relative_to(self.corpus_dir))
            category = categorize_document(md_file, self.corpus_dir)
            chunks = self.parser.parse(text, rel_path, category)
            all_chunks.extend(chunks)

        self.chunks = all_chunks
        logger.info(f"Parsed {len(self.chunks)} chunks from {len(md_files)} files")

        if self.chunks:
            texts = [c.text for c in self.chunks]
            self.chunk_embeddings = self.embedding_manager.generate_embedding(texts)
            logger.info(f"Embedded {len(self.chunk_embeddings)} chunks")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[RawChunk, float]]:
        """Top-k chunks by cosine similarity to query embedding."""
        if not self.chunks or not self.chunk_embeddings:
            return []

        query_emb = self.embedding_manager.generate_embedding([query])[0]

        scores = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            sim = self._cosine_similarity(query_emb, chunk_emb)
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append((self.chunks[idx], score))

        return results

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
