"""Pass 2 — LLM chunk classification.

Classifies each chunk into a fixed category set: decision, discussion,
action_item, status_update, preference, background, filler. Single-word
output — reliable even with small models. Chunks classified as "filler"
are discarded before Pass 3.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from cortex.config import CortexConfig
from cortex.extraction.structural_parser import RawChunk
from cortex.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedChunk:
    """A chunk with its classification label attached."""

    raw: RawChunk
    chunk_type: str  # never "filler" — those are filtered out


class ChunkClassifier:
    """Classifies raw chunks using an LLM.

    Sends each chunk's text to the LLM with the classify prompt.
    Expects a single-word response matching a valid chunk type.
    Filler chunks are separated out so Pass 3 can skip them.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[CortexConfig] = None,
    ):
        self.llm = llm_client
        self.config = config or CortexConfig()
        self._prompt_template = self.config.get_prompt("classify")

    def classify_chunks(
        self, chunks: List[RawChunk]
    ) -> Tuple[List[ClassifiedChunk], List[RawChunk]]:
        """Classify all chunks, separating filler from non-filler.

        Args:
            chunks: Raw chunks from Pass 1.

        Returns:
            Tuple of (non_filler_classified, filler_chunks).
        """
        non_filler: List[ClassifiedChunk] = []
        filler: List[RawChunk] = []

        for chunk in chunks:
            chunk_type = self._classify_single(chunk)

            if chunk_type == "filler":
                logger.debug(
                    "Chunk %s classified as filler — discarding",
                    chunk.id[:8],
                )
                filler.append(chunk)
            else:
                non_filler.append(ClassifiedChunk(raw=chunk, chunk_type=chunk_type))

        logger.info(
            "Classification complete: %d non-filler, %d filler",
            len(non_filler),
            len(filler),
        )
        return non_filler, filler

    def _classify_single(self, chunk: RawChunk) -> str:
        """Classify a single chunk via LLM call."""
        prompt = self._prompt_template.replace("{chunk_text}", chunk.text)

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
        """Validate and normalize LLM classification output.

        Strips whitespace, lowercases, checks against valid set.
        Falls back to default_chunk_type if invalid.
        """
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
