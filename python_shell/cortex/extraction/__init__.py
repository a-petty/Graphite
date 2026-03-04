# cortex.extraction — Three-pass tag-and-index pipeline (Phase 2)
#
# Pass 1: structural_parser.py — deterministic document chunking (no LLM)
# Pass 2: classifier.py — LLM chunk classification (single-word output)
# Pass 3: tagger.py — LLM entity tagging + disambiguation
