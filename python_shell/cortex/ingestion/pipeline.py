"""Three-pass document ingestion orchestrator.

Phase 2: Coordinates the tag-and-index pipeline:
1. Structural parse — deterministic chunking (no LLM)
2. Classify — LLM chunk classification (decision, discussion, etc.)
3. Tag — LLM entity tagging + disambiguation

Filler chunks are discarded between Pass 2 and Pass 3.
Co-occurrence edges are created for all entity pairs within each chunk.
"""
