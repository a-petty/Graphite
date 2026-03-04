"""Pass 2 — LLM chunk classification.

Classifies each chunk into a fixed category set: decision, discussion,
action_item, status_update, preference, background, filler. Single-word
output — reliable even with small models. Chunks classified as "filler"
are discarded before Pass 3.
"""
