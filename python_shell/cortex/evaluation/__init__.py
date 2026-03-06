"""Cortex evaluation framework — Phase 7.

Provides benchmarking tools to compare Cortex's knowledge-graph-based
retrieval against a simple vector-only (RAG) baseline.
"""

from cortex.evaluation.queries import TestQuery, TestQueryLoader
from cortex.evaluation.metrics import MetricResult
from cortex.evaluation.runner import EvalRunner, EvalReport
from cortex.evaluation.report import ReportFormatter

__all__ = [
    "TestQuery",
    "TestQueryLoader",
    "MetricResult",
    "EvalRunner",
    "EvalReport",
    "ReportFormatter",
]
