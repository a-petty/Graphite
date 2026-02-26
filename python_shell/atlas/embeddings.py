# FastEmbed wrapper with skeleton-based chunk-level embeddings

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from fastembed import TextEmbedding
import logging

logger = logging.getLogger(__name__)

# Definition keywords that mark chunk boundaries when splitting skeleton text
_DEFINITION_PREFIXES = (
    "def ", "async def ", "class ",       # Python
    "fn ", "pub fn ", "impl ",            # Rust
    "function ", "export function ",      # JS/TS
    "export default function ",           # JS/TS
    "export class ", "export interface ",  # TS
    "func ",                              # Go
)

# Rough token-to-word ratio for estimating whether text fits in model context.
# BGE-small has a 512 token limit. Using ~400 as target to leave headroom.
_MAX_CHUNK_WORDS = 300  # ~400 tokens


@dataclass
class FileEmbedding:
    """Embeddings for a single file, split into semantic chunks."""
    chunk_embeddings: List[np.ndarray] = field(default_factory=list)


class EmbeddingManager:
    """
    Manages the generation and search of vector embeddings for code files.
    Uses the "Anchor" part of the Anchor & Expand strategy.

    When a repo_graph is provided, files are embedded using their skeleton
    representation (signatures + docstrings + imports, no function bodies)
    split into function-level chunks. This improves search precision by:
    - Fitting within the model's 512-token context window
    - Allowing individual functions to match queries independently
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", repo_graph=None):
        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        try:
            self.model = TextEmbedding(model_name=model_name)
            self.embeddings_cache: Dict[Path, FileEmbedding] = {}
            self.repo_graph = repo_graph
            logger.info("FastEmbed model initialized and ready.")
        except Exception as e:
            logger.error(f"Failed to initialize fastembed model {model_name}: {e}")
            raise

    def _get_embedding_text(self, file_path: Path) -> str:
        """Get text to embed for a file. Uses skeleton if repo_graph is available."""
        if self.repo_graph is not None:
            try:
                skeleton = self.repo_graph.get_skeleton(str(file_path))
                if skeleton and skeleton.strip():
                    return skeleton
            except Exception:
                pass
        return file_path.read_text()

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks at function/class definition boundaries.

        If the text is short enough to fit in a single model context window,
        returns it as a single chunk. Otherwise, splits on definition keywords
        so each function/class gets its own embedding.
        """
        if len(text.split()) <= _MAX_CHUNK_WORDS:
            return [text]

        lines = text.split("\n")
        chunks: List[str] = []
        current_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            is_definition = any(stripped.startswith(kw) for kw in _DEFINITION_PREFIXES)

            if is_definition and current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_lines = []

            current_lines.append(line)

        # Flush remaining lines
        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks if chunks else [text]

    def _file_similarity(self, query_embedding: np.ndarray, file_entry: FileEmbedding) -> float:
        """Compute similarity between a query and a file using max-chunk scoring.

        Returns the highest cosine similarity across all chunks, so a file
        ranks highly if *any* of its functions match the query well.
        """
        if not file_entry.chunk_embeddings:
            return 0.0
        return max(
            self._cosine_similarity(query_embedding, chunk_emb)
            for chunk_emb in file_entry.chunk_embeddings
        )

    def generate_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """Generates embeddings for a list of text inputs."""
        if not texts:
            return []
        try:
            embeddings: List[np.ndarray] = list(self.model.embed(texts))
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            raise

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def find_relevant_files(self, query: str, file_paths: List[Path], top_n: int = 5) -> List[Path]:
        """
        Finds files most relevant to a given query using cosine similarity.

        Files are embedded using their skeleton representation (if repo_graph
        is available) split into function-level chunks. Similarity is computed
        as the maximum chunk similarity, so a file ranks highly if any single
        function matches the query well.

        Args:
            query: The user's query string.
            file_paths: File paths in the repository to consider.
            top_n: Number of top relevant files to return.

        Returns:
            The top_n most relevant file paths, ordered by relevance.
        """
        if not file_paths:
            return []

        logger.debug(f"Finding relevant files for query: '{query}' among {len(file_paths)} files.")

        query_embedding = self.generate_embedding([query])[0]

        similarities: List[Tuple[Path, float]] = []
        # Collect all chunks from uncached files for batch embedding
        all_new_chunks: List[str] = []
        # Track which file each chunk belongs to: (file_path, chunk_count)
        chunk_file_map: List[Tuple[Path, int]] = []

        for file_path in file_paths:
            if file_path in self.embeddings_cache:
                similarity = self._file_similarity(query_embedding, self.embeddings_cache[file_path])
                similarities.append((file_path, similarity))
            else:
                try:
                    content = self._get_embedding_text(file_path)
                    chunks = self._split_into_chunks(content)
                    chunk_file_map.append((file_path, len(chunks)))
                    all_new_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Could not read file {file_path} for embedding: {e}")

        # Batch-embed all new chunks at once
        if all_new_chunks:
            logger.debug(f"Generating embeddings for {len(all_new_chunks)} chunks from {len(chunk_file_map)} new files.")
            new_embeddings = self.generate_embedding(all_new_chunks)

            embed_idx = 0
            for file_path, n_chunks in chunk_file_map:
                chunk_embeds = new_embeddings[embed_idx:embed_idx + n_chunks]
                embed_idx += n_chunks

                entry = FileEmbedding(chunk_embeddings=chunk_embeds)
                self.embeddings_cache[file_path] = entry

                similarity = self._file_similarity(query_embedding, entry)
                similarities.append((file_path, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        relevant_files = [path for path, _ in similarities[:top_n]]
        logger.debug(f"Found {len(relevant_files)} relevant files.")
        return relevant_files
