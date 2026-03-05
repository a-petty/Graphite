"""Pass 3 — LLM entity tagging.

For each non-filler chunk, identifies all entity mentions (person,
project, technology, organization, location, decision, concept).
Returns a flat JSON array of {name, type} pairs. Includes post-
extraction validation (verify names appear in source text) and
inline entity disambiguation against the existing tag index.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cortex.config import CortexConfig
from cortex.extraction.classifier import ClassifiedChunk
from cortex.llm import LLMClient

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
        config: Optional[CortexConfig] = None,
    ):
        self.llm = llm_client
        self.kg = knowledge_graph
        self.embedding_manager = embedding_manager
        self.config = config or CortexConfig()
        self._prompt_template = self.config.get_prompt("tag")
        self.review_queue: List[ReviewItem] = []

    def tag_chunks(self, chunks: List[ClassifiedChunk]) -> List[TaggedChunk]:
        """Tag all chunks with entities. Implements circuit breaker.

        If more than circuit_breaker_failure_rate of chunks fail tagging,
        aborts early and returns whatever was successfully tagged.

        Args:
            chunks: Classified chunks from Pass 2.

        Returns:
            List of TaggedChunk with extracted entities.
        """
        results: List[TaggedChunk] = []
        failures = 0
        total = len(chunks)

        if total == 0:
            return results

        for i, chunk in enumerate(chunks):
            tagged = self._tag_single(chunk)
            if tagged is None:
                failures += 1
                # Circuit breaker: if failure rate exceeds threshold, abort
                if total > 2 and failures / (i + 1) > self.config.circuit_breaker_failure_rate:
                    logger.error(
                        "Circuit breaker tripped: %d/%d chunks failed (%.0f%%). "
                        "Aborting tagging for this document.",
                        failures,
                        i + 1,
                        (failures / (i + 1)) * 100,
                    )
                    break
            else:
                results.append(tagged)

        logger.info(
            "Tagging complete: %d/%d chunks tagged, %d failures",
            len(results),
            total,
            failures,
        )
        return results

    def _tag_single(self, chunk: ClassifiedChunk) -> Optional[TaggedChunk]:
        """Tag a single chunk via LLM call with retry."""
        prompt = self._prompt_template.replace("{chunk_text}", chunk.raw.text)

        for attempt in range(1 + self.config.max_tag_retries):
            try:
                raw_output = self.llm.chat([{"role": "user", "content": prompt}])
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

                # Validate and disambiguate each entity
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

    def _parse_entities_json(self, raw_output: str) -> Optional[List[Dict]]:
        """Parse LLM output into a list of entity dicts.

        Handles common LLM output quirks:
        - Markdown code fences (```json ... ```)
        - XML-like tags (<answer>...</answer>, <response>...</response>)
        - Text before/after the JSON array
        - Wrapper object ({"entities": [...]})
        - Trailing commas
        - Single quotes instead of double quotes
        - Smart/curly quotes
        - JS-style comments (// ...)
        - Numbered lists without array wrapper (1. {...}\n2. {...})
        - Inconsistent key names (entity/Entity → name, category/label → type)
        """
        text = raw_output.strip()

        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")

        # Strip XML-like wrapper tags
        text = re.sub(r"</?(?:answer|response|result|output|json)>", "", text, flags=re.IGNORECASE)

        # Replace smart/curly quotes with straight quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')  # " "
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # ' '

        # Remove JS-style line comments (// ...) but not inside strings
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
        """Try to parse text as a JSON array, with light cleanup."""
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
                # Look for the first list value
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
        # Match lines starting with a number followed by . or )
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
                # Try repair on individual object
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
        """Apply common JSON repairs to a string.

        Fixes:
        - Single quotes → double quotes (for keys and values)
        - Trailing commas before ] or }
        - Newlines inside string values
        - Unquoted keys
        """
        # Replace single quotes with double quotes
        # Handle keys: 'name': → "name":
        text = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', text)
        # Handle values: : 'some value' → : "some value"
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)

        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Replace literal newlines that appear between key-value pairs
        # (not between array elements) with spaces to fix broken strings
        text = text.replace("\\\n", " ")

        # Fix unquoted keys: {name: "..." } → {"name": "..."}
        text = re.sub(r"([{,])\s*(\w+)\s*:", r'\1 "\2":', text)

        return text

    def _normalize_keys(self, entities: List[Dict]) -> List[Dict]:
        """Normalize entity dict keys to canonical {name, type} format.

        Handles common LLM key variations:
        - entity/Entity/entity_name/Name → name
        - category/label/kind/entity_type/Type → type
        """
        _NAME_KEYS = {"name", "entity", "entity_name", "mention"}
        _TYPE_KEYS = {"type", "category", "label", "kind", "entity_type"}

        normalized = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue

            # Case-insensitive key lookup
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
                    "type": str(etype).strip().lower() if etype else "concept",
                })

        return normalized

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
           - >auto_merge_threshold → auto-link
           - >review_threshold → add to review queue
        3. No match → new entity
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
                        self.embedding_manager._cosine_similarity(query_emb, cand_emb)
                    )

                    if score > best_score:
                        best_score = score
                        best_match = existing

                if best_match and best_score >= self.config.disambiguation_auto_merge_threshold:
                    return ExtractedEntity(
                        name=name,
                        entity_type=entity_type.lower(),
                        is_new=False,
                        existing_entity_id=best_match.get("id"),
                    )

                if best_match and best_score >= self.config.disambiguation_review_threshold:
                    self.review_queue.append(
                        ReviewItem(
                            new_entity_name=name,
                            candidate_entity_id=best_match.get("id", ""),
                            candidate_entity_name=best_match.get("canonical_name", ""),
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
