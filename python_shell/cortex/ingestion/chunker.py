"""Document chunking for ingestion pipeline.

Chunking logic lives in cortex.extraction.structural_parser.
This module re-exports the key types for convenience.
"""

from cortex.extraction.structural_parser import RawChunk, StructuralParser

__all__ = ["RawChunk", "StructuralParser"]
