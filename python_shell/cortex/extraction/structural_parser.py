"""Pass 1 — Deterministic structural parsing (no LLM).

Splits documents using their own structure: markdown headers,
speaker turns, date markers, paragraph breaks. Each chunk gets
metadata (section_name, speaker, timestamp, memory_category).
Fully deterministic and repeatable.
"""
