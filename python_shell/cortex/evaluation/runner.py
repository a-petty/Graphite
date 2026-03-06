"""Evaluation runner — orchestrates metrics and produces an EvalReport."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from cortex.config import CortexConfig
from cortex.evaluation.baseline_rag import SimpleRAGBaseline
from cortex.evaluation.metrics import (
    MetricResult,
    evaluate_cooccurrence_accuracy,
    evaluate_context_efficiency,
    evaluate_entity_tagging_accuracy,
    evaluate_llm_error_resilience,
    evaluate_multihop_reasoning,
    evaluate_retrieval_precision_at_k,
    evaluate_temporal_reasoning,
)
from cortex.evaluation.queries import TestQuery, TestQueryLoader

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Complete evaluation report."""

    corpus_dir: str
    document_count: int
    entity_count: int
    edge_count: int
    chunk_count: int
    metrics: List[MetricResult]
    mode: str  # "graph_only" or "full_pipeline"
    duration_seconds: float
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "corpus_dir": self.corpus_dir,
            "document_count": self.document_count,
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
            "chunk_count": self.chunk_count,
            "mode": self.mode,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "baseline_score": m.baseline_score,
                    "queries_evaluated": m.queries_evaluated,
                    "details": m.details,
                    "errors": m.errors,
                }
                for m in self.metrics
            ],
        }


class EvalRunner:
    """Orchestrates evaluation metrics.

    Two modes:
    - graph_only (default): Loads pre-ingested graph. Runs metrics 2–6. No LLM.
    - full_pipeline: Runs extraction pipeline first, then all 7 metrics.
    """

    def __init__(
        self,
        corpus_dir: Path,
        project_root: Path,
        config: CortexConfig = None,
        mode: str = "graph_only",
        llm_client=None,
        k: int = 5,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.project_root = Path(project_root)
        self.config = config or CortexConfig()
        self.mode = mode
        self.llm_client = llm_client
        self.k = k

    def run(self) -> EvalReport:
        """Run the evaluation and return a report."""
        start = time.time()
        errors = []

        # Load knowledge graph
        from cortex.semantic_engine import PyKnowledgeGraph

        graph_path = self.project_root / ".cortex" / "graph.msgpack"
        if not graph_path.exists():
            if self.mode == "full_pipeline" and self.llm_client:
                logger.info("No existing graph — running ingestion pipeline first")
                kg = self._run_ingestion()
            else:
                raise FileNotFoundError(
                    f"No graph found at {graph_path}. Run 'cortex ingest' first, "
                    f"or use --full mode with an LLM provider."
                )
        else:
            logger.info(f"Loading graph from {graph_path}")
            kg = PyKnowledgeGraph.load(str(self.project_root))

        # Get graph statistics
        stats = json.loads(kg.get_statistics())

        # Load test queries
        queries_path = self.corpus_dir / "eval_queries.json"
        if not queries_path.exists():
            raise FileNotFoundError(
                f"No eval_queries.json found at {queries_path}. "
                f"Run generate_corpus.py first."
            )
        queries = TestQueryLoader.load(queries_path)

        # Build baseline index
        logger.info("Building baseline RAG index...")
        baseline = SimpleRAGBaseline(self.corpus_dir, self.config)
        baseline.build_index()

        # Build context manager
        from cortex.context import MemoryContextManager
        from cortex.embeddings import EmbeddingManager

        emb_manager = EmbeddingManager()
        context_manager = MemoryContextManager(
            knowledge_graph=kg,
            embedding_manager=emb_manager,
            config=self.config,
        )

        # Run metrics
        metrics: List[MetricResult] = []

        # Metric 1: Entity tagging (full mode only)
        if self.mode == "full_pipeline":
            logger.info("Running metric 1: Entity Tagging Accuracy")
            try:
                m1 = evaluate_entity_tagging_accuracy(kg, self.corpus_dir)
                metrics.append(m1)
            except Exception as e:
                errors.append(f"Metric 1 error: {e}")
                logger.error(f"Metric 1 failed: {e}")

        # Metric 2: Co-occurrence accuracy
        logger.info("Running metric 2: Co-occurrence Accuracy")
        try:
            m2 = evaluate_cooccurrence_accuracy(kg, self.corpus_dir)
            metrics.append(m2)
        except Exception as e:
            errors.append(f"Metric 2 error: {e}")
            logger.error(f"Metric 2 failed: {e}")

        # Metric 3: Retrieval precision @k
        logger.info(f"Running metric 3: Retrieval Precision @{self.k}")
        try:
            m3 = evaluate_retrieval_precision_at_k(
                context_manager, baseline, queries, k=self.k
            )
            metrics.append(m3)
        except Exception as e:
            errors.append(f"Metric 3 error: {e}")
            logger.error(f"Metric 3 failed: {e}")

        # Metric 4: Temporal reasoning
        logger.info("Running metric 4: Temporal Reasoning")
        try:
            m4 = evaluate_temporal_reasoning(context_manager, baseline, queries)
            metrics.append(m4)
        except Exception as e:
            errors.append(f"Metric 4 error: {e}")
            logger.error(f"Metric 4 failed: {e}")

        # Metric 5: Multi-hop reasoning
        logger.info("Running metric 5: Multi-hop Reasoning")
        try:
            m5 = evaluate_multihop_reasoning(kg, context_manager, baseline, queries)
            metrics.append(m5)
        except Exception as e:
            errors.append(f"Metric 5 error: {e}")
            logger.error(f"Metric 5 failed: {e}")

        # Metric 6: Context efficiency
        logger.info("Running metric 6: Context Efficiency")
        try:
            m6 = evaluate_context_efficiency(context_manager, baseline, queries)
            metrics.append(m6)
        except Exception as e:
            errors.append(f"Metric 6 error: {e}")
            logger.error(f"Metric 6 failed: {e}")

        # Metric 7: LLM error resilience (full mode only)
        if self.mode == "full_pipeline":
            logger.info("Running metric 7: LLM Error Resilience")
            try:
                def kg_factory():
                    return PyKnowledgeGraph(str(self.project_root))

                m7 = evaluate_llm_error_resilience(
                    kg_factory, self.corpus_dir, self.config, self.llm_client
                )
                metrics.append(m7)
            except Exception as e:
                errors.append(f"Metric 7 error: {e}")
                logger.error(f"Metric 7 failed: {e}")

        duration = time.time() - start

        return EvalReport(
            corpus_dir=str(self.corpus_dir),
            document_count=stats.get("documents_indexed", 0),
            entity_count=stats.get("entity_count", 0),
            edge_count=stats.get("edge_count", 0),
            chunk_count=stats.get("chunk_count", 0),
            metrics=metrics,
            mode=self.mode,
            duration_seconds=duration,
            errors=errors,
        )

    def _run_ingestion(self):
        """Run the ingestion pipeline (full_pipeline mode)."""
        from cortex.ingestion.pipeline import IngestionPipeline
        from cortex.semantic_engine import PyKnowledgeGraph

        kg = PyKnowledgeGraph(str(self.project_root))
        pipeline = IngestionPipeline(
            knowledge_graph=kg,
            llm_client=self.llm_client,
            config=self.config,
        )
        pipeline.ingest_directory(self.corpus_dir)
        pipeline.save_graph(str(self.project_root))
        return kg
