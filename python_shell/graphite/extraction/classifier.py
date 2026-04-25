"""Pass 2 — LLM chunk classification.

Classifies each chunk into a fixed category set: decision, discussion,
action_item, status_update, preference, background, filler. Chunks
classified as "filler" are discarded before Pass 3.

Batches multiple chunks into a single LLM call for efficiency.
Falls back to individual calls if batch parsing fails.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from graphite.config import GraphiteConfig
from graphite.extraction.structural_parser import RawChunk
from graphite.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedChunk:
    """A chunk with its classification label attached."""

    raw: RawChunk
    chunk_type: str  # never "filler" — those are filtered out


class ChunkClassifier:
    """Classifies raw chunks using an LLM.

    Batches chunks into groups for efficiency. Falls back to
    individual calls if a batch response can't be parsed.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[GraphiteConfig] = None,
    ):
        self.llm = llm_client
        self.config = config or GraphiteConfig()
        self._single_prompt_template = self.config.get_prompt("classify")
        self._batch_prompt_template = self.config.get_prompt("classify_batch")
        self._batch_size = self.config.classify_batch_size

        # Conversation-specific prompts (loaded lazily)
        self._conv_single_prompt: Optional[str] = None
        self._conv_batch_prompt: Optional[str] = None

    def classify_chunks(
        self, chunks: List[RawChunk]
    ) -> Tuple[List[ClassifiedChunk], List[RawChunk]]:
        """Classify all chunks, separating filler from non-filler.

        Uses batched LLM calls for efficiency.

        Args:
            chunks: Raw chunks from Pass 1.

        Returns:
            Tuple of (non_filler_classified, filler_chunks).
        """
        non_filler: List[ClassifiedChunk] = []
        filler: List[RawChunk] = []

        # Process in batches
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]

            if len(batch) == 1:
                # Single chunk — use the simple prompt
                chunk_type = self._classify_single(batch[0])
                if chunk_type == "filler":
                    filler.append(batch[0])
                else:
                    non_filler.append(
                        ClassifiedChunk(raw=batch[0], chunk_type=chunk_type)
                    )
                continue

            # Try batch classification
            results = self._classify_batch(batch)

            if results is None:
                # Batch failed — fall back to individual calls
                logger.info(
                    "Batch classify failed for chunks %d-%d, falling back to individual calls",
                    i, i + len(batch) - 1,
                )
                for chunk in batch:
                    chunk_type = self._classify_single(chunk)
                    if chunk_type == "filler":
                        filler.append(chunk)
                    else:
                        non_filler.append(
                            ClassifiedChunk(raw=chunk, chunk_type=chunk_type)
                        )
                continue

            # Process batch results
            for chunk, chunk_type in zip(batch, results):
                if chunk_type == "filler":
                    logger.debug(
                        "Chunk %s classified as filler — discarding",
                        chunk.id[:8],
                    )
                    filler.append(chunk)
                else:
                    non_filler.append(
                        ClassifiedChunk(raw=chunk, chunk_type=chunk_type)
                    )

        logger.info(
            "Classification complete: %d non-filler, %d filler (batch size: %d)",
            len(non_filler),
            len(filler),
            self._batch_size,
        )
        return non_filler, filler

    def _is_conversation_source(self, source_document: str) -> bool:
        """Check if a source document is a conversation transcript."""
        return source_document.startswith("claude-session://")

    def _get_prompts(self, is_conversation: bool) -> tuple:
        """Get the appropriate (single, batch) prompt templates."""
        if not is_conversation:
            return self._single_prompt_template, self._batch_prompt_template

        # Lazy-load conversation prompts
        if self._conv_single_prompt is None:
            try:
                self._conv_single_prompt = self.config.get_prompt("classify_conversation")
                self._conv_batch_prompt = self.config.get_prompt("classify_conversation_batch")
            except FileNotFoundError:
                # Fall back to default prompts
                logger.debug("Conversation classify prompts not found, using defaults")
                self._conv_single_prompt = self._single_prompt_template
                self._conv_batch_prompt = self._batch_prompt_template

        return self._conv_single_prompt, self._conv_batch_prompt

    def _classify_batch(self, chunks: List[RawChunk]) -> Optional[List[str]]:
        """Classify a batch of chunks in a single LLM call.

        Returns a list of category strings (same order as input),
        or None if the batch response couldn't be parsed.
        """
        # Build numbered chunk text
        chunk_parts = []
        for idx, chunk in enumerate(chunks, 1):
            chunk_parts.append(f"---CHUNK {idx}---\n{chunk.text}")
        chunks_text = "\n\n".join(chunk_parts)

        # Select prompt based on source
        is_conv = any(
            self._is_conversation_source(c.source_document) for c in chunks
        )
        _, batch_prompt = self._get_prompts(is_conv)
        prompt = batch_prompt.replace("{chunks}", chunks_text)

        try:
            raw_output = self.llm.chat([{"role": "user", "content": prompt}])
            return self._parse_batch_response(raw_output, len(chunks))
        except Exception as e:
            logger.warning("Batch classify LLM call failed: %s", e)
            return None

    def _parse_batch_response(
        self, raw_output: str, expected_count: int
    ) -> Optional[List[str]]:
        """Parse a batch classification response.

        Expects one category per line. Returns None if the count
        doesn't match or too many lines are invalid.
        """
        lines = [
            line.strip().lower()
            for line in raw_output.strip().splitlines()
            if line.strip()
        ]

        # Strip numbering if present (e.g., "1. decision" or "1) decision")
        cleaned = []
        for line in lines:
            # Remove leading number + separator
            stripped = line.lstrip("0123456789").lstrip(".):- ").strip()
            if stripped:
                cleaned.append(stripped)

        # Extract valid labels, filtering out preamble/commentary
        results = []
        for label in cleaned:
            # Take first word, strip punctuation
            first_word = label.split()[0] if label.split() else ""
            first_word = first_word.strip(".,;:!?\"'`")

            if first_word in self.config.valid_chunk_types:
                results.append(first_word)
            # Skip lines that aren't valid labels (preamble, commentary)

        if len(results) == expected_count:
            return results

        # If we got more valid labels than expected, take the first N
        if len(results) > expected_count:
            logger.debug(
                "Batch classify: got %d valid labels for %d chunks — trimming",
                len(results),
                expected_count,
            )
            return results[:expected_count]

        logger.debug(
            "Batch classify: expected %d labels, got %d valid — rejecting batch",
            expected_count,
            len(results),
        )
        return None

    def _classify_single(self, chunk: RawChunk) -> str:
        """Classify a single chunk via LLM call (fallback)."""
        is_conv = self._is_conversation_source(chunk.source_document)
        single_prompt, _ = self._get_prompts(is_conv)
        prompt = single_prompt.replace("{chunk_text}", chunk.text)

        try:
            raw_output = self.llm.chat([{"role": "user", "content": prompt}])
            return self._validate_classification(raw_output)
        except Exception as e:
            logger.warning(
                "Classification failed for chunk %s: %s — defaulting to '%s'",
                chunk.id[:8],
                e,
                self.config.default_chunk_type,
            )
            return self.config.default_chunk_type

    def _validate_classification(self, raw_output: str) -> str:
        """Validate and normalize LLM classification output."""
        cleaned = raw_output.strip().lower()

        # Handle multi-word responses: take the first word
        if " " in cleaned:
            cleaned = cleaned.split()[0]

        # Strip any punctuation
        cleaned = cleaned.strip(".,;:!?\"'`")

        if cleaned in self.config.valid_chunk_types:
            return cleaned

        logger.debug(
            "Invalid classification '%s' — defaulting to '%s'",
            raw_output.strip()[:50],
            self.config.default_chunk_type,
        )
        return self.config.default_chunk_type
