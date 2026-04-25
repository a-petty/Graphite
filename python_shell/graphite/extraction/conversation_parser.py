"""Conversation transcript parser for Claude Code JSONL sessions.

Parses JSONL session transcripts into RawChunk objects compatible with
the existing classify -> tag -> write pipeline.

Strategy — exchange-based chunking:
1. Read JSONL line by line
2. Group into "exchanges" — a user message + all assistant responses
   until the next user message
3. Extract user text, assistant text blocks, summarize tool usage
4. Skip thinking blocks, file-history-snapshot, progress, system types
5. Split oversized exchanges on sentence boundaries

Each chunk gets:
  - source_document: "claude-session://<project_name>/<sessionId>"
  - section_name: "Exchange N"
  - memory_category: "Episodic"
"""

import json
import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from graphite.extraction.structural_parser import RawChunk

logger = logging.getLogger(__name__)

# Types to skip entirely — no useful content for entity extraction
_SKIP_TYPES = {"file-history-snapshot", "progress", "system"}


@dataclass
class SessionMetadata:
    """Metadata about a parsed session."""

    session_id: str
    project_path: str
    project_name: str
    git_branch: Optional[str]
    start_time: Optional[str]  # ISO timestamp string
    end_time: Optional[str]
    exchange_count: int
    tool_usage_summary: Dict[str, int] = field(default_factory=dict)


@dataclass
class _Exchange:
    """Internal: groups a user message with its assistant responses."""

    user_text: str = ""
    assistant_texts: List[str] = field(default_factory=list)
    tool_uses: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None  # ISO from first message in exchange


def _parse_iso_timestamp(iso_str: Optional[str]) -> Optional[int]:
    """Convert ISO 8601 timestamp to Unix seconds, or None."""
    if not iso_str:
        return None
    try:
        from datetime import datetime, timezone

        # Handle Z suffix and +00:00
        cleaned = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return int(dt.timestamp())
    except (ValueError, OSError):
        return None


def _extract_tool_name(tool_use: dict) -> Optional[str]:
    """Extract a human-readable tool summary from a tool_use block."""
    name = tool_use.get("name", "")
    if not name:
        return None

    tool_input = tool_use.get("input", {})
    if isinstance(tool_input, dict):
        # For file-related tools, shorten to just the filename
        file_path = tool_input.get("file_path") or tool_input.get("path")
        if file_path and isinstance(file_path, str):
            if "/" in file_path:
                file_path = file_path.rsplit("/", 1)[-1]
            return f"{name}: {file_path}"

        # For Bash, include truncated command
        command = tool_input.get("command", "")
        if command:
            return f"{name}: {command[:60]}"

    return name


class ConversationParser:
    """Parses Claude Code JSONL session transcripts into RawChunk objects.

    Usage:
        parser = ConversationParser(max_chunk_tokens=1200)
        chunks, metadata = parser.parse_session(Path("~/.claude/projects/.../session.jsonl"))
    """

    def __init__(
        self,
        max_chunk_tokens: int = 1200,
        include_tool_summaries: bool = True,
        skip_tool_output: bool = True,
        chunk_overlap_tokens: int = 100,
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.include_tool_summaries = include_tool_summaries
        self.skip_tool_output = skip_tool_output
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def parse_session(
        self, session_path: Path
    ) -> tuple[List[RawChunk], SessionMetadata]:
        """Parse a JSONL session file into RawChunk objects.

        Args:
            session_path: Path to a .jsonl session transcript.

        Returns:
            Tuple of (chunks, metadata).
        """
        lines = self._read_jsonl(session_path)
        if not lines:
            meta = SessionMetadata(
                session_id=session_path.stem,
                project_path="",
                project_name="unknown",
                git_branch=None,
                start_time=None,
                end_time=None,
                exchange_count=0,
            )
            return [], meta

        # Extract session-level metadata from first relevant message
        session_id, project_path, project_name, git_branch = (
            self._extract_session_info(lines)
        )

        # Build exchanges
        exchanges = self._build_exchanges(lines)

        # Track timestamps for metadata
        timestamps = [e.timestamp for e in exchanges if e.timestamp]
        start_time = timestamps[0] if timestamps else None
        end_time = timestamps[-1] if timestamps else None

        # Aggregate tool usage
        all_tools: Counter = Counter()
        for ex in exchanges:
            for tool_str in ex.tool_uses:
                # Extract just the tool name (before colon)
                tool_name = tool_str.split(":")[0].strip()
                all_tools[tool_name] += 1

        # Build source document URI
        source_document = f"claude-session://{project_name}/{session_id}"

        # Convert exchanges to chunks
        chunks: List[RawChunk] = []
        for idx, exchange in enumerate(exchanges, 1):
            exchange_chunks = self._exchange_to_chunks(
                exchange, idx, source_document
            )
            chunks.extend(exchange_chunks)

        metadata = SessionMetadata(
            session_id=session_id,
            project_path=project_path,
            project_name=project_name,
            git_branch=git_branch,
            start_time=start_time,
            end_time=end_time,
            exchange_count=len(exchanges),
            tool_usage_summary=dict(all_tools),
        )

        logger.info(
            "Parsed session %s: %d exchanges -> %d chunks",
            session_id[:12],
            len(exchanges),
            len(chunks),
        )

        return chunks, metadata

    def _read_jsonl(self, path: Path) -> List[dict]:
        """Read and parse a JSONL file, skipping malformed lines."""
        lines = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping malformed JSON at %s:%d", path, line_num
                        )
        except Exception as e:
            logger.error("Failed to read session file %s: %s", path, e)
        return lines

    def _extract_session_info(
        self, lines: List[dict]
    ) -> tuple[str, str, str, Optional[str]]:
        """Extract session ID, project path, project name, and git branch."""
        session_id = ""
        project_path = ""
        project_name = "unknown"
        git_branch = None

        for line in lines:
            if line.get("type") in _SKIP_TYPES:
                continue

            if not session_id and line.get("sessionId"):
                session_id = line["sessionId"]
            if not project_path and line.get("cwd"):
                project_path = line["cwd"]
                # Derive project name from the last path component
                project_name = Path(project_path).name
            if git_branch is None and line.get("gitBranch"):
                git_branch = line["gitBranch"]

            if session_id and project_path:
                break

        # Fallback session_id from filename if not found in content
        if not session_id:
            session_id = "unknown"

        return session_id, project_path, project_name, git_branch

    def _build_exchanges(self, lines: List[dict]) -> List[_Exchange]:
        """Group JSONL lines into user-assistant exchanges."""
        exchanges: List[_Exchange] = []
        current: Optional[_Exchange] = None

        for line in lines:
            msg_type = line.get("type", "")

            # Skip non-message types
            if msg_type in _SKIP_TYPES:
                continue

            if msg_type == "user":
                # Start a new exchange
                if current is not None and (
                    current.user_text or current.assistant_texts
                    or current.tool_uses
                ):
                    exchanges.append(current)
                current = _Exchange(timestamp=line.get("timestamp"))
                user_text = self._extract_user_text(line)
                if user_text:
                    current.user_text = user_text

            elif msg_type == "assistant":
                if current is None:
                    current = _Exchange(timestamp=line.get("timestamp"))

                # Extract text and tool_use blocks
                message = line.get("message", {})
                content = message.get("content", [])

                if isinstance(content, str):
                    if content.strip():
                        current.assistant_texts.append(content.strip())
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue

                        block_type = block.get("type", "")

                        if block_type == "text":
                            text = block.get("text", "").strip()
                            if text:
                                current.assistant_texts.append(text)

                        elif block_type == "tool_use":
                            tool_summary = _extract_tool_name(block)
                            if tool_summary:
                                current.tool_uses.append(tool_summary)

                        # Skip "thinking" blocks entirely
                        # Skip "tool_result" blocks (tool output)

        # Don't forget the last exchange
        if current is not None and (
            current.user_text or current.assistant_texts
        ):
            exchanges.append(current)

        return exchanges

    def _extract_user_text(self, line: dict) -> str:
        """Extract text content from a user message line."""
        message = line.get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block.strip())
            return "\n".join(parts)

        return ""

    def _exchange_to_chunks(
        self,
        exchange: _Exchange,
        exchange_num: int,
        source_document: str,
    ) -> List[RawChunk]:
        """Convert an exchange into one or more RawChunk objects."""
        # Build the exchange text
        parts: List[str] = []

        if exchange.user_text:
            parts.append(f"User: {exchange.user_text}")

        if exchange.assistant_texts:
            combined_assistant = "\n".join(exchange.assistant_texts)
            parts.append(f"Assistant: {combined_assistant}")

        if self.include_tool_summaries and exchange.tool_uses:
            # Summarize tool usage as a compact line
            tool_counter: Counter = Counter()
            for tool_str in exchange.tool_uses:
                tool_name = tool_str.split(":")[0].strip()
                tool_counter[tool_name] += 1
            tool_summary_parts = [
                f"{name} x{count}" if count > 1 else name
                for name, count in tool_counter.most_common()
            ]
            parts.append(f"[Tools: {', '.join(tool_summary_parts)}]")

        if not parts:
            return []

        full_text = "\n\n".join(parts)
        timestamp = _parse_iso_timestamp(exchange.timestamp)
        section_name = f"Exchange {exchange_num}"

        # Check if text fits in one chunk
        if self._estimate_tokens(full_text) <= self.max_chunk_tokens:
            return [
                RawChunk(
                    id=str(uuid.uuid4()),
                    source_document=source_document,
                    section_name=section_name,
                    speaker=None,
                    timestamp=timestamp,
                    memory_category="Episodic",
                    text=full_text,
                )
            ]

        # Split oversized exchange
        sub_texts = self._split_oversized(full_text)
        chunks = []
        for i, sub in enumerate(sub_texts):
            suffix = f" (part {i + 1})" if len(sub_texts) > 1 else ""
            chunks.append(
                RawChunk(
                    id=str(uuid.uuid4()),
                    source_document=source_document,
                    section_name=f"{section_name}{suffix}",
                    speaker=None,
                    timestamp=timestamp,
                    memory_category="Episodic",
                    text=sub,
                )
            )
        return chunks

    def _split_oversized(self, text: str) -> List[str]:
        """Split text exceeding max_chunk_tokens on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return self._split_by_words(text)

        result: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if (
                current_tokens + sent_tokens > self.max_chunk_tokens
                and current_sentences
            ):
                result.append(" ".join(current_sentences))

                # Overlap
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens > self.chunk_overlap_tokens:
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
        """Fallback word-level splitting."""
        words = text.split()
        max_words = max(int(self.max_chunk_tokens / 1.3), 1)
        overlap_words = min(
            int(self.chunk_overlap_tokens / 1.3),
            max_words - 1,
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

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count: words * 1.3."""
        return int(len(text.split()) * 1.3)
