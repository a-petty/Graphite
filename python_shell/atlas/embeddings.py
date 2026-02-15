# FastEmbed wrapper

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from fastembed import TextEmbedding
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages the generation and search of vector embeddings for code files.
    Uses the "Anchor" part of the Anchor & Expand strategy.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        # Initialize fastembed TextEmbedding model
        # The model will be downloaded on first use if not already present
        try:
            self.model = TextEmbedding(model_name=model_name)
            self.embeddings_cache: Dict[Path, np.ndarray] = {}
            logger.info("FastEmbed model initialized and ready.")
        except Exception as e:
            logger.error(f"Failed to initialize fastembed model {model_name}: {e}")
            raise

    def generate_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates embeddings for a list of text inputs.
        """
        if not texts:
            return []
        try:
            # fastembed returns a generator, so convert it to a list
            embeddings: List[np.ndarray] = list(self.model.embed(texts))
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            raise

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_relevant_files(self, query: str, file_paths: List[Path], top_n: int = 5) -> List[Path]:
        """
        Finds files most relevant to a given query using cosine similarity.

        Args:
            query (str): The user's query string.
            file_paths (List[Path]): A list of all file paths in the repository to consider.
            top_n (int): The number of top relevant files to return.

        Returns:
            List[Path]: A list of the top_n most relevant file paths.
        """
        if not file_paths:
            return []

        logger.debug(f"Finding relevant files for query: '{query}' among {len(file_paths)} files.")

        query_embedding = self.generate_embedding([query])[0] # Assuming single query string

        similarities: List[Tuple[Path, float]] = []
        files_to_embed_content: List[str] = []
        files_to_embed_paths: List[Path] = []

        # Prepare files that are not in cache for embedding
        for file_path in file_paths:
            if file_path not in self.embeddings_cache:
                try:
                    content = file_path.read_text()
                    files_to_embed_content.append(content)
                    files_to_embed_paths.append(file_path)
                except Exception as e:
                    logger.warning(f"Could not read file {file_path} for embedding: {e}")
            else:
                # If in cache, calculate similarity immediately
                file_embedding = self.embeddings_cache[file_path]
                similarity = self._cosine_similarity(query_embedding, file_embedding)
                similarities.append((file_path, similarity))

        # Generate embeddings for new files and add to cache
        if files_to_embed_content:
            logger.debug(f"Generating embeddings for {len(files_to_embed_content)} new files.")
            new_embeddings = self.generate_embedding(files_to_embed_content)
            for i, file_path in enumerate(files_to_embed_paths):
                self.embeddings_cache[file_path] = new_embeddings[i]
                similarity = self._cosine_similarity(query_embedding, new_embeddings[i])
                similarities.append((file_path, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top_n file paths
        relevant_files = [path for path, _ in similarities[:top_n]]
        logger.debug(f"Found {len(relevant_files)} relevant files.")
        return relevant_files

