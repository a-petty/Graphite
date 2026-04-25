"""Pass 3 — LLM entity tagging.

For each non-filler chunk, identifies all entity mentions (person,
project, technology, organization, location, decision, concept).
Returns a flat JSON array of {name, type} pairs. Includes post-
extraction validation (verify names appear in source text) and
inline entity disambiguation against the existing tag index.

Batches multiple chunks into a single LLM call for efficiency.
Falls back to individual calls if batch parsing fails.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from graphite.config import GraphiteConfig
from graphite.extraction.classifier import ClassifiedChunk
from graphite.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from a chunk, with disambiguation info."""

    name: str
    entity_type: str
    is_new: bool = True
    existing_entity_id: Optional[str] = None
    needs_review: bool = False


@dataclass
class TaggedChunk:
    """A classified chunk with its extracted entities."""

    classified: ClassifiedChunk
    entities: List[ExtractedEntity] = field(default_factory=list)


@dataclass
class ReviewItem:
    """An entity match that needs human review (similarity in grey zone)."""

    new_entity_name: str
    candidate_entity_id: str
    candidate_entity_name: str
    similarity_score: float
    chunk_text: str
    source_document: str


class EntityTagger:
    """Extracts entities from classified chunks via LLM.

    Handles:
    - Batched LLM calls with fallback to individual calls
    - LLM output parsing (JSON with error recovery)
    - Anti-hallucination validation (entity names must appear in chunk text)
    - Entity disambiguation against existing graph
    - Circuit breaker (abort document if >50% chunks fail)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_graph=None,
        embedding_manager=None,
        config: Optional[GraphiteConfig] = None,
    ):
        self.llm = llm_client
        self.kg = knowledge_graph
        self.embedding_manager = embedding_manager
        self.config = config or GraphiteConfig()
        self._single_prompt_template = self.config.get_prompt("tag")
        self._batch_prompt_template = self.config.get_prompt("tag_batch")
        self._batch_size = self.config.tag_batch_size
        self.review_queue: List[ReviewItem] = []

        # Conversation-specific prompts (loaded lazily)
        self._conv_single_prompt: Optional[str] = None
        self._conv_batch_prompt: Optional[str] = None

    def tag_chunks(self, chunks: List[ClassifiedChunk]) -> List[TaggedChunk]:
        """Tag all chunks with entities. Uses batching with fallback.

        Implements circuit breaker: if more than circuit_breaker_failure_rate
        of chunks fail tagging, aborts early.

        Args:
            chunks: Classified chunks from Pass 2.

        Returns:
            List of TaggedChunk with extracted entities.
        """
        results: List[TaggedChunk] = []
        failures = 0
        processed = 0

        if not chunks:
            return results

        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]

            if len(batch) == 1:
                # Single chunk — use simple prompt
                tagged = self._tag_single(batch[0])
                processed += 1
                if tagged is None:
                    failures += 1
                else:
                    results.append(tagged)
            else:
                # Try batch tagging
                batch_results = self._tag_batch(batch)

                if batch_results is None:
                    # Batch failed — fall back to individual calls
                    logger.info(
                        "Batch tag failed for chunks %d-%d, "
                        "falling back to individual calls",
                        i, i + len(batch) - 1,
                    )
                    for chunk in batch:
                        tagged = self._tag_single(chunk)
                        processed += 1
                        if tagged is None:
                            failures += 1
                        else:
                            results.append(tagged)
                else:
                    # Process batch results
                    for chunk, entities_raw in zip(batch, batch_results):
                        processed += 1
                        if entities_raw is None:
                            failures += 1
                            continue

                        entities = self._validate_and_disambiguate(
                            entities_raw, chunk
                        )
                        results.append(
                            TaggedChunk(classified=chunk, entities=entities)
                        )

            # Circuit breaker check
            if (
                processed > 2
                and failures / processed > self.config.circuit_breaker_failure_rate
            ):
                logger.error(
                    "Circuit breaker tripped: %d/%d chunks failed (%.0f%%). "
                    "Aborting tagging for this document.",
                    failures,
                    processed,
                    (failures / processed) * 100,
                )
                break

        logger.info(
            "Tagging complete: %d/%d chunks tagged, %d failures (batch size: %d)",
            len(results),
            len(chunks),
            failures,
            self._batch_size,
        )
        return results

    # ── Prompt Selection ──

    def _is_conversation_source(self, source_document: str) -> bool:
        """Check if a source document is a conversation transcript."""
        return source_document.startswith("claude-session://")

    def _get_prompts(self, is_conversation: bool) -> tuple:
        """Get the appropriate (single, batch) prompt templates."""
        if not is_conversation:
            return self._single_prompt_template, self._batch_prompt_template

        if self._conv_single_prompt is None:
            try:
                self._conv_single_prompt = self.config.get_prompt("tag_conversation")
                self._conv_batch_prompt = self.config.get_prompt("tag_conversation_batch")
            except FileNotFoundError:
                logger.debug("Conversation tag prompts not found, using defaults")
                self._conv_single_prompt = self._single_prompt_template
                self._conv_batch_prompt = self._batch_prompt_template

        return self._conv_single_prompt, self._conv_batch_prompt

    # ── Batch Tagging ──

    def _tag_batch(
        self, chunks: List[ClassifiedChunk]
    ) -> Optional[List[Optional[List[Dict]]]]:
        """Tag a batch of chunks in a single LLM call.

        Returns a list of entity lists (same order as input chunks).
        Individual entries can be None if that chunk's entities couldn't
        be parsed. Returns None entirely if the response is unparseable.
        """
        # Build numbered chunk text
        chunk_parts = []
        for idx, chunk in enumerate(chunks, 1):
            chunk_parts.append(f"---CHUNK {idx}---\n{chunk.raw.text}")
        chunks_text = "\n\n".join(chunk_parts)

        is_conv = any(
            self._is_conversation_source(c.raw.source_document) for c in chunks
        )
        _, batch_prompt = self._get_prompts(is_conv)
        prompt = batch_prompt.replace("{chunks}", chunks_text)

        try:
            raw_output = self.llm.chat([{"role": "user", "content": prompt}])
            return self._parse_batch_tag_response(raw_output, len(chunks))
        except Exception as e:
            logger.warning("Batch tag LLM call failed: %s", e)
            return None

    def _parse_batch_tag_response(
        self, raw_output: str, expected_count: int
    ) -> Optional[List[Optional[List[Dict]]]]:
        """Parse a batch tag response.

        Expects a JSON object: {"1": [...], "2": [...], ...}
        Returns list of entity lists, or None if unparseable.
        """
        text = raw_output.strip()

        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")

        # Strip XML-like wrapper tags
        text = re.sub(
            r"</?(?:answer|response|result|output|json)>",
            "", text, flags=re.IGNORECASE,
        )

        # Replace smart quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")

        # Remove JS-style comments
        text = re.sub(r'//[^\n"]*$', "", text, flags=re.MULTILINE)

        text = text.strip()

        # Try to parse as a JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")

        if brace_start == -1 or brace_end <= brace_start:
            logger.debug("Batch tag: no JSON object found in response")
            return None

        json_str = text[brace_start : brace_end + 1]

        parsed = self._try_parse_object(json_str)
        if parsed is None:
            # Try repair
            repaired = self._repair_json_str(json_str)
            if repaired:
                parsed = self._try_parse_object(repaired)

        if parsed is None:
            logger.debug("Batch tag: failed to parse JSON object")
            return None

        # Extract results in order
        results: List[Optional[List[Dict]]] = []
        for idx in range(1, expected_count + 1):
            key = str(idx)
            if key in parsed and isinstance(parsed[key], list):
                results.append(self._normalize_keys(parsed[key]))
            else:
                # Try without quotes (some models return int keys)
                results.append(None)

        # If we got nothing for any chunk, that's OK — individual entries
        # can be None. But if ALL are None, reject the batch.
        if all(r is None for r in results):
            logger.debug("Batch tag: all chunk results were None")
            return None

        return results

    def _try_parse_object(self, text: str) -> Optional[Dict]:
        """Try to parse text as a JSON object."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    # ── Single-Chunk Tagging (Fallback) ──

    def _tag_single(self, chunk: ClassifiedChunk) -> Optional[TaggedChunk]:
        """Tag a single chunk via LLM call with retry."""
        is_conv = self._is_conversation_source(chunk.raw.source_document)
        single_prompt, _ = self._get_prompts(is_conv)
        prompt = single_prompt.replace(
            "{chunk_text}", chunk.raw.text
        )

        for attempt in range(1 + self.config.max_tag_retries):
            try:
                raw_output = self.llm.chat(
                    [{"role": "user", "content": prompt}]
                )
                entities_raw = self._parse_entities_json(raw_output)

                if entities_raw is None:
                    if attempt < self.config.max_tag_retries:
                        logger.debug(
                            "JSON parse failed for chunk %s, retrying (%d/%d)",
                            chunk.raw.id[:8],
                            attempt + 1,
                            self.config.max_tag_retries,
                        )
                        continue
                    logger.warning(
                        "JSON parse failed for chunk %s after %d attempts",
                        chunk.raw.id[:8],
                        attempt + 1,
                    )
                    return None

                entities = self._validate_and_disambiguate(
                    entities_raw, chunk
                )
                return TaggedChunk(classified=chunk, entities=entities)

            except Exception as e:
                if attempt < self.config.max_tag_retries:
                    logger.debug(
                        "Tagging error for chunk %s: %s, retrying",
                        chunk.raw.id[:8],
                        e,
                    )
                    continue
                logger.warning(
                    "Tagging failed for chunk %s: %s",
                    chunk.raw.id[:8],
                    e,
                )
                return None

        return None

    # ── Validation & Disambiguation ──

    def _validate_and_disambiguate(
        self,
        entities_raw: List[Dict],
        chunk: ClassifiedChunk,
    ) -> List[ExtractedEntity]:
        """Validate entity names appear in chunk text, then disambiguate."""
        entities: List[ExtractedEntity] = []
        for ent_dict in entities_raw:
            if not self._validate_entity(ent_dict, chunk.raw.text):
                logger.debug(
                    "Hallucinated entity removed: '%s' not in chunk text",
                    ent_dict.get("name", "?"),
                )
                continue

            entity = self._disambiguate_entity(
                ent_dict["name"],
                ent_dict.get("type", "concept"),
                chunk.raw.text,
                chunk.raw.source_document,
            )
            entities.append(entity)
        return entities

    def _validate_entity(self, entity: Dict, chunk_text: str) -> bool:
        """Validate that an entity name actually appears in the chunk text.

        Case-insensitive substring match. This is the anti-hallucination check.
        """
        name = entity.get("name", "")
        if not name or not isinstance(name, str):
            return False

        etype = entity.get("type", "")
        if not etype or not isinstance(etype, str):
            return False

        # Normalize entity type
        etype_lower = etype.lower()
        if etype_lower not in self.config.valid_entity_types:
            return False

        # Case-insensitive substring match
        return name.lower() in chunk_text.lower()

    def _disambiguate_entity(
        self,
        name: str,
        entity_type: str,
        chunk_text: str,
        source_document: str,
    ) -> ExtractedEntity:
        """Disambiguate an entity against the existing knowledge graph.

        Strategy:
        1. Exact name match via kg.search_entities()
        2. Embedding similarity (if embedding_manager available)
           - >auto_merge_threshold -> auto-link
           - >review_threshold -> add to review queue
        3. No match -> new entity
        """
        if self.kg is None:
            return ExtractedEntity(
                name=name,
                entity_type=entity_type.lower(),
                is_new=True,
            )

        # 1. Exact match via tag index search
        try:
            results_json = self.kg.search_entities(name, 5)
            results = json.loads(results_json)
        except Exception:
            results = []

        for existing in results:
            existing_name = existing.get("canonical_name", "")
            # Exact match (case-insensitive)
            if existing_name.lower() == name.lower():
                return ExtractedEntity(
                    name=name,
                    entity_type=entity_type.lower(),
                    is_new=False,
                    existing_entity_id=existing.get("id"),
                )
            # Also check aliases
            aliases = existing.get("aliases", [])
            for alias in aliases:
                if alias.lower() == name.lower():
                    return ExtractedEntity(
                        name=name,
                        entity_type=entity_type.lower(),
                        is_new=False,
                        existing_entity_id=existing.get("id"),
                    )

        # 2. Embedding similarity check (if available)
        if self.embedding_manager is not None and results:
            try:
                query_emb = self.embedding_manager.generate_embedding([name])[0]
                best_score = 0.0
                best_match = None

                for existing in results:
                    existing_name = existing.get("canonical_name", "")
                    cand_emb = self.embedding_manager.generate_embedding(
                        [existing_name]
                    )[0]
                    score = float(
                        self.embedding_manager._cosine_similarity(
                            query_emb, cand_emb
                        )
                    )

                    if score > best_score:
                        best_score = score
                        best_match = existing

                if (
                    best_match
                    and best_score
                    >= self.config.disambiguation_auto_merge_threshold
                ):
                    return ExtractedEntity(
                        name=name,
                        entity_type=entity_type.lower(),
                        is_new=False,
                        existing_entity_id=best_match.get("id"),
                    )

                if (
                    best_match
                    and best_score
                    >= self.config.disambiguation_review_threshold
                ):
                    self.review_queue.append(
                        ReviewItem(
                            new_entity_name=name,
                            candidate_entity_id=best_match.get("id", ""),
                            candidate_entity_name=best_match.get(
                                "canonical_name", ""
                            ),
                            similarity_score=best_score,
                            chunk_text=chunk_text[:200],
                            source_document=source_document,
                        )
                    )
                    return ExtractedEntity(
                        name=name,
                        entity_type=entity_type.lower(),
                        is_new=True,
                        needs_review=True,
                    )

            except Exception as e:
                logger.debug("Embedding disambiguation failed: %s", e)

        # 3. No match — new entity
        return ExtractedEntity(
            name=name,
            entity_type=entity_type.lower(),
            is_new=True,
        )

    # ── JSON Parsing Utilities ──

    def _parse_entities_json(self, raw_output: str) -> Optional[List[Dict]]:
        """Parse LLM output into a list of entity dicts.

        Handles common LLM output quirks:
        - Markdown code fences
        - XML-like tags
        - Text before/after the JSON array
        - Wrapper object ({"entities": [...]})
        - Trailing commas, single quotes, smart quotes
        - JS-style comments
        - Numbered lists without array wrapper
        - Inconsistent key names
        """
        text = raw_output.strip()

        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")

        # Strip XML-like wrapper tags
        text = re.sub(
            r"</?(?:answer|response|result|output|json)>",
            "", text, flags=re.IGNORECASE,
        )

        # Replace smart/curly quotes with straight quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")

        # Remove JS-style line comments
        text = re.sub(r'//[^\n"]*$', "", text, flags=re.MULTILINE)

        text = text.strip()

        # Try to find a JSON array
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")

        if bracket_start != -1 and bracket_end > bracket_start:
            json_str = text[bracket_start : bracket_end + 1]
            result = self._try_parse_array(json_str)
            if result is not None:
                return self._normalize_keys(result)

        # Try wrapper object: {"entities": [...]} or {"results": [...]}
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            obj_str = text[brace_start : brace_end + 1]
            result = self._try_unwrap_object(obj_str)
            if result is not None:
                return self._normalize_keys(result)

        # Try numbered list: 1. {"name": ...}\n2. {"name": ...}
        result = self._try_parse_numbered_list(text)
        if result is not None:
            return self._normalize_keys(result)

        # Last resort: repair and retry
        if bracket_start != -1 and bracket_end > bracket_start:
            json_str = text[bracket_start : bracket_end + 1]
            result = self._repair_json(json_str)
            if result is not None:
                return self._normalize_keys(result)

        return None

    def _try_parse_array(self, text: str) -> Optional[List[Dict]]:
        """Try to parse text as a JSON array."""
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _try_unwrap_object(self, text: str) -> Optional[List[Dict]]:
        """Try to parse as a wrapper object and extract the array value."""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for val in obj.values():
                    if isinstance(val, list):
                        return val
        except json.JSONDecodeError:
            pass

        # Try with repair
        repaired = self._repair_json_str(text)
        if repaired:
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    for val in obj.values():
                        if isinstance(val, list):
                            return val
            except json.JSONDecodeError:
                pass

        return None

    def _try_parse_numbered_list(self, text: str) -> Optional[List[Dict]]:
        """Try to parse numbered list format: 1. {...}\n2. {...}"""
        pattern = re.compile(r"^\s*\d+[.)]\s*(\{.*\})\s*$", re.MULTILINE)
        matches = pattern.findall(text)
        if not matches:
            return None

        results = []
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                repaired = self._repair_json_str(match)
                if repaired:
                    try:
                        obj = json.loads(repaired)
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        continue

        return results if results else None

    def _repair_json(self, text: str) -> Optional[List[Dict]]:
        """Attempt to repair a malformed JSON array string."""
        repaired = self._repair_json_str(text)
        if repaired is None:
            return None

        try:
            result = json.loads(repaired)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return None

    def _repair_json_str(self, text: str) -> Optional[str]:
        """Apply common JSON repairs to a string."""
        # Replace single quotes with double quotes
        text = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', text)
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)

        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Replace literal newlines in strings
        text = text.replace("\\\n", " ")

        # Fix unquoted keys
        text = re.sub(r"([{,])\s*(\w+)\s*:", r'\1 "\2":', text)

        return text

    def _normalize_keys(self, entities: List[Dict]) -> List[Dict]:
        """Normalize entity dict keys to canonical {name, type} format."""
        _NAME_KEYS = {"name", "entity", "entity_name", "mention"}
        _TYPE_KEYS = {"type", "category", "label", "kind", "entity_type"}

        normalized = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue

            lower_keys = {k.lower(): v for k, v in ent.items()}

            name = None
            for key in _NAME_KEYS:
                if key in lower_keys:
                    name = lower_keys[key]
                    break

            etype = None
            for key in _TYPE_KEYS:
                if key in lower_keys:
                    etype = lower_keys[key]
                    break

            if name and isinstance(name, str):
                normalized.append({
                    "name": name.strip(),
                    "type": (
                        str(etype).strip().lower() if etype else "concept"
                    ),
                })

        return normalized
