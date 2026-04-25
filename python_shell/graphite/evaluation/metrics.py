"""Evaluation metrics for the Graphite evaluation framework.

Seven metrics comparing Graphite knowledge-graph retrieval against a
vector-only RAG baseline. Relevance scoring uses keyword/entity string
matching — no LLM-as-judge.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from graphite.config import GraphiteConfig
from graphite.evaluation.baseline_rag import SimpleRAGBaseline
from graphite.evaluation.queries import TestQuery, TestQueryLoader

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""

    name: str
    score: float  # 0.0 – 1.0 (or relevance-per-1k-tokens for efficiency)
    details: Dict  # metric-specific breakdown
    queries_evaluated: int = 0
    errors: List[str] = field(default_factory=list)

    # Baseline comparison (None if metric is Graphite-only)
    baseline_score: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_contains_keywords(text: str, keywords: List[str]) -> bool:
    """Check if chunk text contains any of the expected keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _count_keyword_hits(text: str, keywords: List[str]) -> int:
    """Count how many distinct keywords appear in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _load_expected_json(corpus_dir: Path) -> Dict[str, Dict]:
    """Load all .expected.json files from the corpus, keyed by relative path."""
    result = {}
    for expected_file in sorted(corpus_dir.rglob("*.expected.json")):
        doc_path = expected_file.with_suffix("").with_suffix(".md")
        rel = str(doc_path.relative_to(corpus_dir))
        with open(expected_file) as f:
            result[rel] = json.load(f)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: Entity Tagging Accuracy (requires full pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_entity_tagging_accuracy(
    knowledge_graph,
    corpus_dir: Path,
) -> MetricResult:
    """Precision/recall of extracted entities vs .expected.json ground truth.

    Compares entities in the knowledge graph against expected entities
    from all .expected.json files.
    """
    expected_data = _load_expected_json(corpus_dir)
    if not expected_data:
        return MetricResult(
            name="Entity Tagging Accuracy",
            score=0.0,
            details={"error": "No .expected.json files found"},
        )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_doc = {}

    # Get all entities from graph
    all_ids_json = knowledge_graph.all_entity_ids()
    all_ids = json.loads(all_ids_json)
    graph_entities = {}
    for eid in all_ids:
        entity_json = knowledge_graph.get_entity(eid)
        if entity_json:
            entity = json.loads(entity_json)
            name = entity["canonical_name"].lower()
            graph_entities[name] = entity

    for doc_path, expected in expected_data.items():
        expected_names = {e["name"].lower() for e in expected.get("entities", [])}

        # Find graph entities sourced from this document
        found_names = set()
        for name, entity in graph_entities.items():
            if doc_path in entity.get("source_documents", []):
                found_names.add(name)

        tp = len(expected_names & found_names)
        fp = len(found_names - expected_names)
        fn = len(expected_names - found_names)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_doc[doc_path] = {"tp": tp, "fp": fp, "fn": fn}

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return MetricResult(
        name="Entity Tagging Accuracy",
        score=f1,
        details={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "documents_evaluated": len(expected_data),
        },
        queries_evaluated=len(expected_data),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: Co-occurrence Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cooccurrence_accuracy(
    knowledge_graph,
    corpus_dir: Path,
) -> MetricResult:
    """Check that expected entity pairs are connected in the graph."""
    expected_data = _load_expected_json(corpus_dir)
    if not expected_data:
        return MetricResult(
            name="Co-occurrence Accuracy",
            score=0.0,
            details={"error": "No .expected.json files found"},
        )

    # Build entity name → id lookup
    all_ids = json.loads(knowledge_graph.all_entity_ids())
    name_to_id = {}
    for eid in all_ids:
        entity_json = knowledge_graph.get_entity(eid)
        if entity_json:
            entity = json.loads(entity_json)
            name_to_id[entity["canonical_name"].lower()] = eid

    total_expected = 0
    total_found = 0

    for doc_path, expected in expected_data.items():
        for cooc in expected.get("cooccurrences", []):
            a_name = cooc["entity_a"].lower()
            b_name = cooc["entity_b"].lower()
            total_expected += 1

            a_id = name_to_id.get(a_name)
            b_id = name_to_id.get(b_name)
            if not a_id or not b_id:
                continue

            # Check if there's an edge between a and b
            cooccurrences_json = knowledge_graph.get_cooccurrences(a_id)
            neighbors = json.loads(cooccurrences_json)
            neighbor_ids = {n[0] for n in neighbors}
            if b_id in neighbor_ids:
                total_found += 1

    score = total_found / total_expected if total_expected > 0 else 0.0

    return MetricResult(
        name="Co-occurrence Accuracy",
        score=round(score, 4),
        details={
            "expected_pairs": total_expected,
            "found_pairs": total_found,
            "missing_pairs": total_expected - total_found,
        },
        queries_evaluated=total_expected,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3: Retrieval Precision @k
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_retrieval_precision_at_k(
    context_manager,
    baseline: SimpleRAGBaseline,
    queries: List[TestQuery],
    k: int = 5,
) -> MetricResult:
    """Compare top-k retrieval relevance: Graphite vs baseline.

    A chunk is 'relevant' if it contains any expected_answer_keywords.
    """
    retrieval_queries = TestQueryLoader.filter_by_type(queries, "retrieval")
    if not retrieval_queries:
        return MetricResult(
            name=f"Retrieval Precision @{k}",
            score=0.0,
            details={"error": "No retrieval queries"},
        )

    graphite_scores = []
    baseline_scores = []
    errors = []

    for query in retrieval_queries:
        keywords = query.expected_answer_keywords + query.expected_entities

        # Graphite: use assemble_context, then check if keywords appear
        try:
            context = context_manager.assemble_context(query.question)
            if context:
                hit_count = _count_keyword_hits(context, keywords)
                # Precision = fraction of expected keywords found
                precision = hit_count / len(keywords) if keywords else 0.0
                graphite_scores.append(precision)
            else:
                graphite_scores.append(0.0)
        except Exception as e:
            errors.append(f"Graphite error on {query.id}: {e}")
            graphite_scores.append(0.0)

        # Baseline: top-k chunks
        try:
            results = baseline.retrieve(query.question, top_k=k)
            combined_text = " ".join(chunk.text for chunk, _ in results)
            hit_count = _count_keyword_hits(combined_text, keywords)
            precision = hit_count / len(keywords) if keywords else 0.0
            baseline_scores.append(precision)
        except Exception as e:
            errors.append(f"Baseline error on {query.id}: {e}")
            baseline_scores.append(0.0)

    graphite_avg = sum(graphite_scores) / len(graphite_scores) if graphite_scores else 0.0
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

    return MetricResult(
        name=f"Retrieval Precision @{k}",
        score=round(graphite_avg, 4),
        baseline_score=round(baseline_avg, 4),
        details={
            "graphite_per_query": [round(s, 4) for s in graphite_scores],
            "baseline_per_query": [round(s, 4) for s in baseline_scores],
            "k": k,
        },
        queries_evaluated=len(retrieval_queries),
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4: Temporal Reasoning
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_temporal_reasoning(
    context_manager,
    baseline: SimpleRAGBaseline,
    queries: List[TestQuery],
) -> MetricResult:
    """Test that time-filtered retrieval surfaces correct state.

    Graphite uses time_start/time_end in assemble_context.
    Baseline has no temporal filtering — we filter by timestamp manually.
    """
    temporal_queries = TestQueryLoader.filter_by_type(queries, "temporal")
    if not temporal_queries:
        return MetricResult(
            name="Temporal Reasoning",
            score=0.0,
            details={"error": "No temporal queries"},
        )

    graphite_scores = []
    baseline_scores = []
    errors = []

    for query in temporal_queries:
        keywords = query.expected_answer_keywords + query.expected_entities
        tc = query.time_context or {}
        time_start = tc.get("start")
        time_end = tc.get("end")

        # Graphite: with temporal filtering
        try:
            context = context_manager.assemble_context(
                query.question, time_start=time_start, time_end=time_end
            )
            if context:
                hit_count = _count_keyword_hits(context, keywords)
                score = hit_count / len(keywords) if keywords else 0.0
                graphite_scores.append(score)
            else:
                graphite_scores.append(0.0)
        except Exception as e:
            errors.append(f"Graphite temporal error on {query.id}: {e}")
            graphite_scores.append(0.0)

        # Baseline: no temporal filtering, just top-k
        try:
            results = baseline.retrieve(query.question, top_k=10)
            combined_text = " ".join(chunk.text for chunk, _ in results)
            hit_count = _count_keyword_hits(combined_text, keywords)
            score = hit_count / len(keywords) if keywords else 0.0
            baseline_scores.append(score)
        except Exception as e:
            errors.append(f"Baseline temporal error on {query.id}: {e}")
            baseline_scores.append(0.0)

    graphite_avg = sum(graphite_scores) / len(graphite_scores) if graphite_scores else 0.0
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

    return MetricResult(
        name="Temporal Reasoning",
        score=round(graphite_avg, 4),
        baseline_score=round(baseline_avg, 4),
        details={
            "queries_with_time_context": sum(1 for q in temporal_queries if q.time_context),
        },
        queries_evaluated=len(temporal_queries),
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 5: Multi-hop Reasoning
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_multihop_reasoning(
    knowledge_graph,
    context_manager,
    baseline: SimpleRAGBaseline,
    queries: List[TestQuery],
) -> MetricResult:
    """Test that 2+ hop BFS reaches expected entities.

    Graphite traverses the co-occurrence graph.
    Baseline can only find entities within the same retrieved chunks.
    """
    multihop_queries = TestQueryLoader.filter_by_type(queries, "multi_hop")
    if not multihop_queries:
        return MetricResult(
            name="Multi-hop Reasoning",
            score=0.0,
            details={"error": "No multi-hop queries"},
        )

    # Build name→id lookup
    all_ids = json.loads(knowledge_graph.all_entity_ids())
    name_to_id = {}
    for eid in all_ids:
        entity_json = knowledge_graph.get_entity(eid)
        if entity_json:
            entity = json.loads(entity_json)
            name_to_id[entity["canonical_name"].lower()] = eid

    graphite_scores = []
    baseline_scores = []
    errors = []

    for query in multihop_queries:
        expected = query.expected_entities
        keywords = query.expected_answer_keywords + expected

        # Graphite: assemble_context uses graph traversal
        try:
            context = context_manager.assemble_context(query.question)
            if context:
                hit_count = _count_keyword_hits(context, keywords)
                score = hit_count / len(keywords) if keywords else 0.0
                graphite_scores.append(score)
            else:
                graphite_scores.append(0.0)
        except Exception as e:
            errors.append(f"Graphite multihop error on {query.id}: {e}")
            graphite_scores.append(0.0)

        # Baseline: top-k chunks
        try:
            results = baseline.retrieve(query.question, top_k=10)
            combined_text = " ".join(chunk.text for chunk, _ in results)
            hit_count = _count_keyword_hits(combined_text, keywords)
            score = hit_count / len(keywords) if keywords else 0.0
            baseline_scores.append(score)
        except Exception as e:
            errors.append(f"Baseline multihop error on {query.id}: {e}")
            baseline_scores.append(0.0)

    graphite_avg = sum(graphite_scores) / len(graphite_scores) if graphite_scores else 0.0
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

    return MetricResult(
        name="Multi-hop Reasoning",
        score=round(graphite_avg, 4),
        baseline_score=round(baseline_avg, 4),
        details={
            "avg_hops_required": round(
                sum(q.hops_required for q in multihop_queries) / len(multihop_queries), 1
            ),
        },
        queries_evaluated=len(multihop_queries),
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 6: Context Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_context_efficiency(
    context_manager,
    baseline: SimpleRAGBaseline,
    queries: List[TestQuery],
) -> MetricResult:
    """Relevance per 1000 tokens: Graphite vs baseline.

    Higher is better — measures how efficiently the context budget
    is used to deliver relevant information.
    """
    if not queries:
        return MetricResult(
            name="Context Efficiency",
            score=0.0,
            details={"error": "No queries"},
        )

    graphite_efficiencies = []
    baseline_efficiencies = []
    errors = []

    for query in queries[:30]:  # Cap at 30 to keep runtime reasonable
        keywords = query.expected_answer_keywords + query.expected_entities

        # Graphite
        try:
            context = context_manager.assemble_context(query.question)
            if context:
                tokens = len(context.split()) * 1.3  # rough token estimate
                hits = _count_keyword_hits(context, keywords)
                efficiency = (hits / tokens * 1000) if tokens > 0 else 0.0
                graphite_efficiencies.append(efficiency)
            else:
                graphite_efficiencies.append(0.0)
        except Exception as e:
            errors.append(f"Graphite efficiency error on {query.id}: {e}")
            graphite_efficiencies.append(0.0)

        # Baseline
        try:
            results = baseline.retrieve(query.question, top_k=10)
            combined_text = " ".join(chunk.text for chunk, _ in results)
            if combined_text:
                tokens = len(combined_text.split()) * 1.3
                hits = _count_keyword_hits(combined_text, keywords)
                efficiency = (hits / tokens * 1000) if tokens > 0 else 0.0
                baseline_efficiencies.append(efficiency)
            else:
                baseline_efficiencies.append(0.0)
        except Exception as e:
            errors.append(f"Baseline efficiency error on {query.id}: {e}")
            baseline_efficiencies.append(0.0)

    graphite_avg = (sum(graphite_efficiencies) / len(graphite_efficiencies)
                  if graphite_efficiencies else 0.0)
    baseline_avg = (sum(baseline_efficiencies) / len(baseline_efficiencies)
                    if baseline_efficiencies else 0.0)

    return MetricResult(
        name="Context Efficiency",
        score=round(graphite_avg, 4),
        baseline_score=round(baseline_avg, 4),
        details={
            "unit": "relevant_keywords_per_1k_tokens",
        },
        queries_evaluated=min(len(queries), 30),
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric 7: LLM Error Resilience (requires full pipeline mode)
# ─────────────────────────────────────────────────────────────────────────────

class DegradedLLMClient:
    """Wraps an LLM client, randomly introducing failures and malformed JSON.

    Used by metric 7 to test pipeline resilience to LLM errors.
    """

    def __init__(self, base_client, failure_rate: float = 0.1,
                 malformed_rate: float = 0.15, seed: int = 42):
        self.base_client = base_client
        self.failure_rate = failure_rate
        self.malformed_rate = malformed_rate
        self.rng = random.Random(seed)
        self.call_count = 0
        self.failures_injected = 0
        self.malformed_injected = 0

    def chat(self, messages):
        """Call the base client, but randomly inject errors."""
        self.call_count += 1
        roll = self.rng.random()

        if roll < self.failure_rate:
            self.failures_injected += 1
            raise ConnectionError("Simulated LLM connection failure")

        if roll < self.failure_rate + self.malformed_rate:
            self.malformed_injected += 1
            return "{malformed json output that should not parse correctly"

        return self.base_client.chat(messages)


def evaluate_llm_error_resilience(
    knowledge_graph_factory,
    corpus_dir: Path,
    config: GraphiteConfig,
    base_llm_client=None,
) -> MetricResult:
    """Test that the pipeline degrades gracefully with a noisy LLM.

    Runs the ingestion pipeline with a DegradedLLMClient and checks
    that some entities are still extracted despite errors.
    """
    from graphite.ingestion.pipeline import IngestionPipeline

    if base_llm_client is None:
        from graphite.llm import StubClient
        base_llm_client = StubClient()

    degraded = DegradedLLMClient(base_llm_client, failure_rate=0.1, malformed_rate=0.15)
    kg = knowledge_graph_factory()

    pipeline = IngestionPipeline(
        knowledge_graph=kg,
        llm_client=degraded,
        config=config,
    )

    # Ingest a subset of documents
    md_files = sorted(corpus_dir.rglob("*.md"))[:10]
    total_entities = 0
    total_errors = 0
    results = []

    for md_file in md_files:
        try:
            result = pipeline.ingest_file(md_file)
            results.append(result)
            total_entities += result.entities_created + result.entities_linked
            total_errors += len(result.errors)
        except Exception as e:
            total_errors += 1

    # Score: fraction of documents that produced at least some entities
    docs_with_entities = sum(1 for r in results
                           if (r.entities_created + r.entities_linked) > 0)
    resilience_score = docs_with_entities / len(md_files) if md_files else 0.0

    return MetricResult(
        name="LLM Error Resilience",
        score=round(resilience_score, 4),
        details={
            "documents_processed": len(md_files),
            "documents_with_entities": docs_with_entities,
            "total_entities_extracted": total_entities,
            "llm_calls": degraded.call_count,
            "failures_injected": degraded.failures_injected,
            "malformed_injected": degraded.malformed_injected,
            "pipeline_errors": total_errors,
        },
        queries_evaluated=len(md_files),
    )
