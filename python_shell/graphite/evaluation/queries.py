"""Test query data types and loader for the evaluation framework."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """A single evaluation query with ground-truth expectations."""

    id: str
    type: str  # "retrieval", "temporal", "multi_hop"
    question: str
    expected_entities: List[str]
    expected_answer_keywords: List[str]
    expected_source_documents: List[str] = field(default_factory=list)
    min_relevant_chunks: int = 1
    time_context: Optional[Dict[str, int]] = None  # {"start": ..., "end": ...}
    hops_required: int = 1


class TestQueryLoader:
    """Loads and validates eval_queries.json."""

    VALID_TYPES = {"retrieval", "temporal", "multi_hop", "filler_rejection", "cross_document_chain"}

    @classmethod
    def load(cls, path: Path) -> List[TestQuery]:
        """Load queries from a JSON file.

        Args:
            path: Path to eval_queries.json.

        Returns:
            List of validated TestQuery objects.

        Raises:
            FileNotFoundError: If path doesn't exist.
            ValueError: If any query fails validation.
        """
        with open(path) as f:
            raw = json.load(f)

        queries = []
        for item in raw:
            cls._validate(item)
            queries.append(TestQuery(
                id=item["id"],
                type=item["type"],
                question=item["question"],
                expected_entities=item["expected_entities"],
                expected_answer_keywords=item["expected_answer_keywords"],
                expected_source_documents=item.get("expected_source_documents", []),
                min_relevant_chunks=item.get("min_relevant_chunks", 1),
                time_context=item.get("time_context"),
                hops_required=item.get("hops_required", 1),
            ))

        logger.info(f"Loaded {len(queries)} test queries from {path}")
        return queries

    @classmethod
    def filter_by_type(cls, queries: List[TestQuery], query_type: str) -> List[TestQuery]:
        """Filter queries by type."""
        return [q for q in queries if q.type == query_type]

    @classmethod
    def _validate(cls, item: dict):
        """Validate a single query dict."""
        required = ["id", "type", "question", "expected_entities", "expected_answer_keywords"]
        for key in required:
            if key not in item:
                raise ValueError(f"Query missing required field '{key}': {item.get('id', '?')}")

        if item["type"] not in cls.VALID_TYPES:
            raise ValueError(
                f"Query {item['id']} has invalid type '{item['type']}'. "
                f"Valid types: {cls.VALID_TYPES}"
            )

        if not item["expected_entities"]:
            raise ValueError(f"Query {item['id']} has empty expected_entities")
