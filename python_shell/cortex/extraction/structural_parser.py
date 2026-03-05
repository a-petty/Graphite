"""Pass 1 — Deterministic structural parsing (no LLM).

Splits documents using their own structure: markdown headers,
speaker turns, date markers, paragraph breaks. Each chunk gets
metadata (section_name, speaker, timestamp, memory_category).
Fully deterministic and repeatable.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from cortex.config import CortexConfig


@dataclass
class RawChunk:
    """A single chunk produced by structural parsing."""

    id: str
    source_document: str
    section_name: Optional[str]
    speaker: Optional[str]
    timestamp: Optional[int]  # Unix timestamp
    memory_category: str  # "Episodic" / "Semantic" / "Procedural"
    text: str


# ── Patterns ──

# Speaker turn: **Name:** or **Name**: or Speaker N: at start of line
# Handles both **Name:** (colon inside bold) and **Name**: (colon outside)
_SPEAKER_BOLD = re.compile(
    r"^\*\*([^*:]+?):\*\*\s*"  # **Name:** — colon inside bold
    r"|"
    r"^\*\*([^*]+?)\*\*:\s*",  # **Name**: — colon outside bold
    re.MULTILINE,
)
_SPEAKER_LABEL = re.compile(r"^(Speaker\s+\d+):\s*", re.MULTILINE)

# Markdown headers (## or ###)
_MD_HEADER = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

# Date markers as section delimiters (ISO dates or written dates)
_DATE_SECTION = re.compile(
    r"^(?:#+\s*)?(\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4})$",
    re.MULTILINE,
)

# Date patterns for timestamp extraction
_DATE_PATTERNS = [
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), "%Y-%m-%d"),
    (re.compile(r"\b(\w+ \d{1,2},? \d{4})\b"), None),  # flexible parse
]


class StructuralParser:
    """Deterministic document chunker for Pass 1.

    Detects document type and splits accordingly:
    - meeting_transcript: speaker turns (**Name:** or Speaker N:)
    - markdown: ## headers + paragraph breaks
    - date_structured: date markers as section delimiters
    - plain_text: paragraph breaks with token-limit fallback
    """

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()

    def parse(
        self,
        text: str,
        source_path: str,
        memory_category: str,
    ) -> List[RawChunk]:
        """Parse a document into structural chunks.

        Args:
            text: Full document text.
            source_path: Path/identifier for the source document.
            memory_category: "Episodic", "Semantic", or "Procedural".

        Returns:
            List of RawChunk objects.
        """
        if not text or not text.strip():
            return []

        doc_type = self._detect_document_type(text)
        timestamp = self._extract_timestamp(text)

        if doc_type == "meeting_transcript":
            chunks = self._parse_meeting_transcript(text, source_path, memory_category, timestamp)
        elif doc_type == "markdown":
            chunks = self._parse_markdown(text, source_path, memory_category, timestamp)
        elif doc_type == "date_structured":
            chunks = self._parse_date_structured(text, source_path, memory_category)
        else:
            chunks = self._parse_plain_text(text, source_path, memory_category, timestamp)

        return chunks

    def _detect_document_type(self, text: str) -> str:
        """Detect document format from content patterns.

        Returns one of: meeting_transcript, markdown, date_structured, plain_text
        """
        # Check for speaker turns first (most specific)
        if _SPEAKER_BOLD.search(text) or _SPEAKER_LABEL.search(text):
            return "meeting_transcript"

        # Check for date section markers (before markdown, since dates
        # may use # headers)
        date_matches = _DATE_SECTION.findall(text)
        if len(date_matches) >= 2:
            return "date_structured"

        # Check for markdown headers
        if _MD_HEADER.search(text):
            return "markdown"

        return "plain_text"

    def _parse_meeting_transcript(
        self,
        text: str,
        source_path: str,
        memory_category: str,
        doc_timestamp: Optional[int],
    ) -> List[RawChunk]:
        """Split on speaker turns: **Name:** or Speaker N: patterns."""
        chunks: List[RawChunk] = []

        # Find the current section name from any header before speaker turns
        section_name = None
        header_match = _MD_HEADER.search(text)
        if header_match:
            section_name = header_match.group(2).strip()

        # Split by speaker turns — combine both patterns
        # Find all speaker positions
        turns: List[tuple] = []  # (start, speaker_name, text_start)

        for m in _SPEAKER_BOLD.finditer(text):
            # group(1) = colon-inside-bold, group(2) = colon-outside-bold
            speaker_name = (m.group(1) or m.group(2)).strip()
            turns.append((m.start(), speaker_name, m.end()))
        for m in _SPEAKER_LABEL.finditer(text):
            turns.append((m.start(), m.group(1).strip(), m.end()))

        # Sort by position
        turns.sort(key=lambda t: t[0])

        if not turns:
            # No speaker turns found — treat as plain text
            return self._parse_plain_text(text, source_path, memory_category, doc_timestamp)

        # Extract text before first speaker turn (preamble)
        preamble = text[: turns[0][0]].strip()
        if preamble and self._estimate_tokens(preamble) > 10:
            chunks.append(
                RawChunk(
                    id=str(uuid.uuid4()),
                    source_document=source_path,
                    section_name=section_name,
                    speaker=None,
                    timestamp=doc_timestamp,
                    memory_category=memory_category,
                    text=preamble,
                )
            )

        # Extract each speaker turn
        for i, (start, speaker, text_start) in enumerate(turns):
            # End is either start of next turn or end of document
            if i + 1 < len(turns):
                end = turns[i + 1][0]
            else:
                end = len(text)

            turn_text = text[text_start:end].strip()
            if not turn_text:
                continue

            # Update section name if there's a header within this turn
            for hm in _MD_HEADER.finditer(turn_text):
                section_name = hm.group(2).strip()

            # Split oversized turns
            sub_texts = self._split_oversized_chunk(turn_text)
            for sub in sub_texts:
                chunks.append(
                    RawChunk(
                        id=str(uuid.uuid4()),
                        source_document=source_path,
                        section_name=section_name,
                        speaker=speaker,
                        timestamp=doc_timestamp,
                        memory_category=memory_category,
                        text=sub,
                    )
                )

        return chunks

    def _parse_markdown(
        self,
        text: str,
        source_path: str,
        memory_category: str,
        doc_timestamp: Optional[int],
    ) -> List[RawChunk]:
        """Split on ## headers and double-newline paragraph breaks."""
        chunks: List[RawChunk] = []

        # Split at ## or ### headers
        sections = _MD_HEADER.split(text)

        # sections alternates: [pre_header, level, title, content, level, title, content, ...]
        # First element is text before any header
        current_section = None

        if sections[0].strip():
            # Preamble text before first header
            self._add_paragraph_chunks(
                sections[0].strip(),
                source_path,
                memory_category,
                doc_timestamp,
                None,
                chunks,
            )

        # Process header-content pairs
        i = 1
        while i < len(sections) - 1:
            # sections[i] = header level (## or ###)
            # sections[i+1] = header text
            current_section = sections[i + 1].strip()
            # Content is sections[i+2] if it exists
            if i + 2 < len(sections):
                content = sections[i + 2].strip()
                if content:
                    self._add_paragraph_chunks(
                        content,
                        source_path,
                        memory_category,
                        doc_timestamp,
                        current_section,
                        chunks,
                    )
            i += 3

        return chunks

    def _add_paragraph_chunks(
        self,
        text: str,
        source_path: str,
        memory_category: str,
        timestamp: Optional[int],
        section_name: Optional[str],
        chunks: List[RawChunk],
    ):
        """Split text on double newlines into paragraph-level chunks."""
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            sub_texts = self._split_oversized_chunk(para)
            for sub in sub_texts:
                chunks.append(
                    RawChunk(
                        id=str(uuid.uuid4()),
                        source_document=source_path,
                        section_name=section_name,
                        speaker=None,
                        timestamp=timestamp,
                        memory_category=memory_category,
                        text=sub,
                    )
                )

    def _parse_date_structured(
        self,
        text: str,
        source_path: str,
        memory_category: str,
    ) -> List[RawChunk]:
        """Split on date markers as section delimiters."""
        chunks: List[RawChunk] = []

        # Find date positions
        date_positions: List[tuple] = []  # (start, end, date_str)
        for m in _DATE_SECTION.finditer(text):
            date_positions.append((m.start(), m.end(), m.group(1)))

        if not date_positions:
            return self._parse_plain_text(text, source_path, memory_category, None)

        # Preamble before first date
        preamble = text[: date_positions[0][0]].strip()
        if preamble and self._estimate_tokens(preamble) > 10:
            chunks.append(
                RawChunk(
                    id=str(uuid.uuid4()),
                    source_document=source_path,
                    section_name=None,
                    speaker=None,
                    timestamp=self._extract_timestamp(preamble),
                    memory_category=memory_category,
                    text=preamble,
                )
            )

        for i, (start, end, date_str) in enumerate(date_positions):
            # Content runs until next date marker or end of document
            if i + 1 < len(date_positions):
                content_end = date_positions[i + 1][0]
            else:
                content_end = len(text)

            content = text[end:content_end].strip()
            if not content:
                continue

            ts = self._parse_date_string(date_str)
            sub_texts = self._split_oversized_chunk(content)
            for sub in sub_texts:
                chunks.append(
                    RawChunk(
                        id=str(uuid.uuid4()),
                        source_document=source_path,
                        section_name=date_str,
                        speaker=None,
                        timestamp=ts,
                        memory_category=memory_category,
                        text=sub,
                    )
                )

        return chunks

    def _parse_plain_text(
        self,
        text: str,
        source_path: str,
        memory_category: str,
        doc_timestamp: Optional[int],
    ) -> List[RawChunk]:
        """Split on paragraph breaks, with token-limit fallback."""
        chunks: List[RawChunk] = []
        paragraphs = re.split(r"\n\s*\n", text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            sub_texts = self._split_oversized_chunk(para)
            for sub in sub_texts:
                chunks.append(
                    RawChunk(
                        id=str(uuid.uuid4()),
                        source_document=source_path,
                        section_name=None,
                        speaker=None,
                        timestamp=doc_timestamp,
                        memory_category=memory_category,
                        text=sub,
                    )
                )

        return chunks

    def _split_oversized_chunk(self, text: str) -> List[str]:
        """Split text exceeding max_chunk_tokens with overlap.

        If the text fits within the token limit, returns it as-is.
        Otherwise splits on sentence boundaries with overlap.
        """
        if self._estimate_tokens(text) <= self.config.max_chunk_tokens:
            return [text]

        # Split into sentences for better boundary handling
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            # Can't split on sentences — split on words
            return self._split_by_words(text)

        result: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if current_tokens + sent_tokens > self.config.max_chunk_tokens and current_sentences:
                # Emit current chunk
                result.append(" ".join(current_sentences))

                # Calculate overlap: keep trailing sentences up to overlap_tokens
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens > self.config.chunk_overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            result.append(" ".join(current_sentences))

        return result

    def _split_by_words(self, text: str) -> List[str]:
        """Fallback: split by words when sentence splitting isn't possible."""
        words = text.split()
        max_words = max(int(self.config.max_chunk_tokens / 1.3), 1)
        overlap_words = min(
            int(self.config.chunk_overlap_tokens / 1.3),
            max_words - 1,  # overlap must be strictly less than max_words
        )

        result: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            result.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - overlap_words

        return result

    def _extract_timestamp(self, text: str) -> Optional[int]:
        """Try to extract a date from text and return as Unix timestamp."""
        # Try ISO date first
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if iso_match:
            return self._parse_date_string(iso_match.group(1))

        # Try written dates like "November 18, 2024"
        written_match = re.search(
            r"\b(\w+ \d{1,2},?\s+\d{4})\b", text
        )
        if written_match:
            return self._parse_date_string(written_match.group(1))

        return None

    def _parse_date_string(self, date_str: str) -> Optional[int]:
        """Parse a date string into a UTC Unix timestamp."""
        from datetime import timezone
        for fmt in ["%Y-%m-%d", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"]:
            try:
                dt = datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except ValueError:
                continue
        return None

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate: words * 1.3."""
        return int(len(text.split()) * 1.3)
