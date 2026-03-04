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

    # Common source directory names that should be stripped from module paths
    _SOURCE_DIR_NAMES = {"src", "lib", "source", "sources"}

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", repo_graph=None, project_root: Optional[Path] = None):
        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        try:
            self.model = TextEmbedding(model_name=model_name)
            self.embeddings_cache: Dict[Path, FileEmbedding] = {}
            self.repo_graph = repo_graph
            self.project_root = project_root
            logger.info("FastEmbed model initialized and ready.")
        except Exception as e:
            logger.error(f"Failed to initialize fastembed model {model_name}: {e}")
            raise

    def _file_path_to_module_prefix(self, file_path: Path) -> str:
        """Convert a file path to a dotted module prefix for embedding.

        Examples:
            /repo/airflow-core/src/airflow/models/pool.py → airflow.models.pool
            /repo/src/utils/helpers.py → utils.helpers
            /repo/mypackage/__init__.py → mypackage
        """
        if self.project_root is None:
            return ""
        try:
            rel = file_path.resolve().relative_to(self.project_root.resolve())
        except ValueError:
            return ""

        parts = list(rel.parts)
        if not parts:
            return ""

        # Strip common source directory names wherever they appear (e.g., src/, lib/)
        # This handles both top-level (src/airflow/...) and nested
        # monorepo layouts (airflow-core/src/airflow/...)
        parts = [p for p in parts if p.lower() not in self._SOURCE_DIR_NAMES]

        if not parts:
            return ""

        # Remove file extension from last part
        last = parts[-1]
        stem = Path(last).stem
        if stem == "__init__":
            parts = parts[:-1]
        else:
            parts[-1] = stem

        return ".".join(parts)

    def _get_embedding_text(self, file_path: Path) -> str:
        """Get text to embed for a file.

        Uses skeleton if repo_graph is available, and prepends the dotted
        module path (e.g., 'airflow.models.pool') so that file identity
        is captured in the embedding.
        """
        text = None
        if self.repo_graph is not None:
            try:
                skeleton = self.repo_graph.get_skeleton(str(file_path))
                if skeleton and skeleton.strip():
                    text = skeleton
            except Exception:
                pass
        if text is None:
            text = file_path.read_text()

        # Prepend module path prefix for semantic signal
        prefix = self._file_path_to_module_prefix(file_path)
        if prefix:
            text = f"{prefix}\n{text}"

        return text

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
        """Generates embeddings for a list of text inputs.

        Sorts inputs by length before batching to minimize ONNX padding overhead.
        ONNX pads all inputs in a batch to the longest sequence; without sorting,
        a single long text forces every item in that batch to process at max length.
        """
        if not texts:
            return []
        try:
            # Sort by length so similarly-sized texts batch together
            indexed = sorted(enumerate(texts), key=lambda x: len(x[1]))
            sorted_texts = [t for _, t in indexed]

            sorted_embeddings = list(self.model.embed(sorted_texts))

            # Restore original order
            embeddings = [None] * len(texts)
            for i, (orig_idx, _) in enumerate(indexed):
                embeddings[orig_idx] = sorted_embeddings[i]

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

    def find_relevant_files_scored(self, query: str, file_paths: List[Path], top_n: int = 5) -> List[Tuple[Path, float]]:
        """
        Finds files most relevant to a query, returning (path, similarity) tuples.

        Files are embedded using their skeleton representation (if repo_graph
        is available) split into function-level chunks. Similarity is computed
        as the maximum chunk similarity, so a file ranks highly if any single
        function matches the query well.

        Args:
            query: The user's query string.
            file_paths: File paths in the repository to consider.
            top_n: Number of top relevant files to return.

        Returns:
            The top_n most relevant (file_path, similarity_score) tuples,
            ordered by descending similarity.
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
            if len(chunk_file_map) > 100:
                logger.info(
                    f"Embedding {len(all_new_chunks)} chunks from {len(chunk_file_map)} files "
                    f"(first search — this may take 20-30s on CPU, cached after that)..."
                )
            else:
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

        logger.debug(f"Found {min(top_n, len(similarities))} relevant files.")
        return similarities[:top_n]

    def find_relevant_files(self, query: str, file_paths: List[Path], top_n: int = 5) -> List[Path]:
        """
        Finds files most relevant to a given query using cosine similarity.

        Convenience wrapper around find_relevant_files_scored() that returns
        just the file paths without scores.
        """
        return [path for path, _ in self.find_relevant_files_scored(query, file_paths, top_n)]
