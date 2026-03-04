"""Pass 3 — LLM entity tagging.

For each non-filler chunk, identifies all entity mentions (person,
project, technology, organization, location, decision, concept).
Returns a flat JSON array of {name, type} pairs. Includes post-
extraction validation (verify names appear in source text) and
inline entity disambiguation against the existing tag index.
"""
