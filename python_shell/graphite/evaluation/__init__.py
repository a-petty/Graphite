"""Graphite evaluation framework — Phase 7.

Provides benchmarking tools to compare Graphite's knowledge-graph-based
retrieval against a simple vector-only (RAG) baseline.
"""

from graphite.evaluation.queries import TestQuery, TestQueryLoader
from graphite.evaluation.metrics import MetricResult
from graphite.evaluation.runner import EvalRunner, EvalReport
from graphite.evaluation.report import ReportFormatter

__all__ = [
    "TestQuery",
    "TestQueryLoader",
    "MetricResult",
    "EvalRunner",
    "EvalReport",
    "ReportFormatter",
]
