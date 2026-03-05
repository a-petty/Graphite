"""Tests for Phase 3: MemoryContextManager and entity embedding support.

Covers:
- Entity descriptor building and embedding caching
- Anchor phase (embedding search + PageRank re-ranking)
- Expand phase (neighborhood traversal, hop reconstruction)
- Chunk scoring (type weights, distance decay)
- Tier 1: Knowledge Map formatting and budget
- Tier 2: Evidence chunks ordering and formatting
- Tier 3: Peripheral entity summaries
- Budget allocation and adaptive params
- Full assembly end-to-end
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cortex.config import CortexConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Mock Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


def _deterministic_embedding(text: str) -> np.ndarray:
    """Generate a deterministic embedding from text for testing.

    Uses hash of text to seed a random generator, producing a
    consistent 384-dim vector (BGE-small dimensionality).
    """
    seed = hash(text) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    # Normalize to unit vector
    vec = vec / np.linalg.norm(vec)
    return vec


class FakeKnowledgeGraph:
    """Mock PyKnowledgeGraph returning JSON strings like the real Rust binding.

    Stores entities, chunks, and edges in dicts and implements the
    subset of PyKnowledgeGraph methods used by MemoryContextManager.
    """

    def __init__(self):
        self._entities = {}   # id -> entity dict
        self._chunks = {}     # id -> chunk dict
        self._edges = []      # [(source_id, target_id, edge_dict), ...]
        self._pagerank_computed = False
        self._document_hashes = {}  # document path -> content hash

    def add_test_entity(
        self, entity_id, canonical_name, entity_type="Person",
        source_chunks=None, source_documents=None,
        created_at=None, updated_at=None,
    ):
        """Helper to add an entity for testing."""
        now = int(datetime(2024, 6, 15, tzinfo=timezone.utc).timestamp())
        self._entities[entity_id] = {
            "id": entity_id,
            "canonical_name": canonical_name,
            "entity_type": entity_type,
            "aliases": [],
            "source_chunks": source_chunks or [],
            "source_documents": source_documents or [],
            "created_at": created_at or now,
            "updated_at": updated_at or now,
            "access_count": 0,
            "embedding": None,
            "rank": 0.0,
            "extraction_confidence": None,
            "merge_history": [],
        }

    def add_test_chunk(
        self, chunk_id, text, chunk_type="Discussion",
        source_document="doc.md", section_name=None,
        speaker=None, timestamp=None, tags=None,
        memory_category="Episodic",
    ):
        """Helper to add a chunk for testing."""
        now = int(datetime(2024, 6, 15, tzinfo=timezone.utc).timestamp())
        self._chunks[chunk_id] = {
            "id": chunk_id,
            "text": text,
            "chunk_type": chunk_type,
            "source_document": source_document,
            "section_name": section_name,
            "speaker": speaker,
            "timestamp": timestamp,
            "tags": tags or [],
            "memory_category": memory_category,
            "created_at": now,
        }

    def add_test_edge(
        self, source_id, target_id, chunk_id="",
        chunk_type="Discussion", timestamp=None,
        source_document="doc.md",
    ):
        """Helper to add a co-occurrence edge for testing."""
        self._edges.append((source_id, target_id, {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "memory_category": "Episodic",
            "timestamp": timestamp,
            "source_document": source_document,
            "weight": 1.0,
        }))

    # ── PyKnowledgeGraph API methods ──

    def compute_pagerank(self):
        """Returns JSON array of [id, score] pairs."""
        # Simple: assign rank based on edge count
        edge_counts = {}
        for s, t, _ in self._edges:
            edge_counts[s] = edge_counts.get(s, 0) + 1
            edge_counts[t] = edge_counts.get(t, 0) + 1

        total = sum(edge_counts.values()) or 1
        pairs = []
        for eid in self._entities:
            score = edge_counts.get(eid, 0) / total
            pairs.append([eid, score])

        self._pagerank_computed = True
        return json.dumps(pairs)

    def get_entity(self, entity_id):
        """Returns entity JSON or None."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return None
        return json.dumps(entity)

    def get_chunk(self, chunk_id):
        """Returns chunk JSON or None."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return None
        return json.dumps(chunk)

    def get_cooccurrences(self, entity_id):
        """Returns JSON array of [neighbor_id, {edge fields}]."""
        results = []
        for s, t, e in self._edges:
            if s == entity_id:
                results.append([t, e])
            elif t == entity_id:
                results.append([s, e])
        return json.dumps(results)

    def query_neighborhood(self, entity_id, hops, time_start=None, time_end=None):
        """BFS through edges, returning SubgraphResult as JSON."""
        visited = {entity_id}
        frontier = {entity_id}
        collected_edges = []
        has_time_filter = time_start is not None or time_end is not None

        for _ in range(hops):
            next_frontier = set()
            for s, t, e in self._edges:
                # Temporal filter (Bug #3 fix: exclude None-timestamp edges)
                ts = e.get("timestamp")
                if has_time_filter and ts is None:
                    continue
                if time_start is not None and ts is not None and ts < time_start:
                    continue
                if time_end is not None and ts is not None and ts > time_end:
                    continue

                if s in frontier and t not in visited:
                    next_frontier.add(t)
                    collected_edges.append([s, t, e])
                elif t in frontier and s not in visited:
                    next_frontier.add(s)
                    collected_edges.append([s, t, e])
            visited.update(next_frontier)
            frontier = next_frontier

        entities = [
            self._entities[eid] for eid in visited if eid in self._entities
        ]

        # Collect chunks tagged with any visited entity
        chunks = []
        for chunk in self._chunks.values():
            if any(tag in visited for tag in chunk.get("tags", [])):
                chunks.append(chunk)

        result = {
            "entities": entities,
            "edges": collected_edges,
            "chunks": chunks,
        }
        return json.dumps(result)

    def get_chunks_for_entities(self, entity_ids_json):
        """Returns JSON array of chunks."""
        ids = json.loads(entity_ids_json)
        ids_set = set(ids)
        chunks = [
            c for c in self._chunks.values()
            if any(t in ids_set for t in c.get("tags", []))
        ]
        return json.dumps(chunks)

    def search_entities(self, query, limit):
        """Substring search on entity canonical_name and aliases."""
        query_lower = query.lower()
        results = []
        for entity in self._entities.values():
            name = entity["canonical_name"].lower()
            aliases = [a.lower() for a in entity.get("aliases", [])]
            if query_lower in name or any(query_lower in a for a in aliases):
                results.append(entity)
                if len(results) >= limit:
                    break
        return json.dumps(results)

    def get_temporal_chain(self, entity_id):
        """Returns JSON array of chunks tagged with entity, sorted by timestamp desc."""
        tagged = [
            c for c in self._chunks.values()
            if entity_id in c.get("tags", [])
        ]
        # Sort most recent first (None timestamps last)
        tagged.sort(key=lambda c: c.get("timestamp") or 0, reverse=True)
        return json.dumps(tagged)

    def remove_entity(self, entity_id):
        """Remove entity and its edges. Returns True if entity existed."""
        if entity_id not in self._entities:
            return False
        del self._entities[entity_id]
        self._edges = [
            (s, t, e) for s, t, e in self._edges
            if s != entity_id and t != entity_id
        ]
        return True

    def all_entity_ids(self):
        """Returns JSON array of all entity IDs."""
        return json.dumps(list(self._entities.keys()))

    def find_orphan_entities(self):
        """Returns JSON array of entity IDs with no edges."""
        connected = set()
        for s, t, _ in self._edges:
            connected.add(s)
            connected.add(t)
        orphans = [eid for eid in self._entities if eid not in connected]
        return json.dumps(orphans)

    def merge_entities(self, keep_id, merge_id, confidence=None, method=None):
        """Absorb merge entity into keep entity. Returns kept ID."""
        if keep_id not in self._entities:
            raise RuntimeError(f"Entity not found: {keep_id}")
        if merge_id not in self._entities:
            raise RuntimeError(f"Entity not found: {merge_id}")

        keep = self._entities[keep_id]
        merge = self._entities[merge_id]

        # Absorb aliases
        if merge["canonical_name"] not in keep["aliases"]:
            keep["aliases"].append(merge["canonical_name"])
        for alias in merge.get("aliases", []):
            if alias not in keep["aliases"]:
                keep["aliases"].append(alias)

        # Absorb chunks and docs
        for c in merge.get("source_chunks", []):
            if c not in keep["source_chunks"]:
                keep["source_chunks"].append(c)
        for d in merge.get("source_documents", []):
            if d not in keep["source_documents"]:
                keep["source_documents"].append(d)

        keep["access_count"] = keep.get("access_count", 0) + merge.get("access_count", 0)

        # Record merge history (Bug #13 fix)
        keep.setdefault("merge_history", []).append({
            "merged_entity_id": merge_id,
            "merged_entity_name": merge["canonical_name"],
            "merged_at": int(time.time()),
            "confidence": confidence if confidence is not None else 0.0,
            "method": method if method is not None else "direct",
        })

        # Redirect edges
        new_edges = []
        for s, t, e in self._edges:
            ns = keep_id if s == merge_id else s
            nt = keep_id if t == merge_id else t
            if ns != nt:  # skip self-loops
                new_edges.append((ns, nt, e))
        self._edges = new_edges

        # Update chunk tags
        for chunk in self._chunks.values():
            if merge_id in chunk.get("tags", []):
                idx = chunk["tags"].index(merge_id)
                chunk["tags"][idx] = keep_id

        del self._entities[merge_id]
        return keep_id

    def get_top_entities(self, limit):
        """Returns JSON array of entities sorted by rank."""
        entities = sorted(
            self._entities.values(),
            key=lambda e: e.get("rank", 0.0),
            reverse=True,
        )[:limit]
        return json.dumps(entities)

    def recalculate_edge_weights(self):
        """No-op, returns edge count."""
        return len(self._edges)

    def store_chunk(self, chunk_json):
        """Store a chunk from JSON string (matches PyKnowledgeGraph API)."""
        chunk = json.loads(chunk_json) if isinstance(chunk_json, str) else chunk_json
        import uuid
        if "id" not in chunk:
            chunk["id"] = str(uuid.uuid4())
        if "created_at" not in chunk:
            chunk["created_at"] = int(time.time())
        chunk_id = chunk["id"]
        self._chunks[chunk_id] = chunk

        # Populate entity source_chunks (Bug #9 fix)
        for tag in chunk.get("tags", []):
            if tag in self._entities:
                if chunk_id not in self._entities[tag].get("source_chunks", []):
                    self._entities[tag].setdefault("source_chunks", []).append(chunk_id)

        return chunk_id

    def decay_scores(self, half_life_days):
        """Apply exponential decay to access counts in the fake graph."""
        import math
        now = int(time.time())
        half_life_secs = half_life_days * 86400
        for entity in self._entities.values():
            age_secs = now - entity.get("updated_at", now)
            decay_factor = math.exp(-math.log(2) * age_secs / half_life_secs)
            entity["access_count"] = round(entity.get("access_count", 0) * decay_factor)

    def save(self, path):
        """No-op save for testing."""
        pass

    def remove_document(self, document):
        """Remove a document and cascade-clean its chunks, edges, and orphaned entities.

        Returns JSON DocumentRemovalResult matching the Rust API.
        """
        # Collect chunk IDs for this document
        chunk_ids = {
            cid for cid, c in self._chunks.items()
            if c.get("source_document") == document
        }
        chunks_removed = len(chunk_ids)

        # Remove edges whose chunk_id is in the document's chunk set
        edges_before = len(self._edges)
        self._edges = [
            (s, t, e) for s, t, e in self._edges
            if e.get("chunk_id") not in chunk_ids
        ]
        edges_removed = edges_before - len(self._edges)

        # Remove chunks
        for cid in chunk_ids:
            del self._chunks[cid]

        # Update entities
        entities_to_remove = []
        entities_updated = 0
        for eid, entity in list(self._entities.items()):
            if document not in entity.get("source_documents", []):
                continue
            entity["source_documents"] = [
                d for d in entity["source_documents"] if d != document
            ]
            entity["source_chunks"] = [
                c for c in entity.get("source_chunks", []) if c not in chunk_ids
            ]
            if not entity["source_documents"] and not entity.get("source_chunks"):
                # Check for remaining edges
                has_edges = any(
                    s == eid or t == eid for s, t, _ in self._edges
                )
                if not has_edges:
                    entities_to_remove.append(eid)
                else:
                    entities_updated += 1
            else:
                entities_updated += 1

        entities_removed = len(entities_to_remove)
        for eid in entities_to_remove:
            del self._entities[eid]

        return json.dumps({
            "chunks_removed": chunks_removed,
            "edges_removed": edges_removed,
            "entities_removed": entities_removed,
            "entities_updated": entities_updated,
        })

    def get_chunks_by_document(self, document):
        """Get all chunks belonging to a document. Returns JSON array."""
        chunks = [
            c for c in self._chunks.values()
            if c.get("source_document") == document
        ]
        return json.dumps(chunks)

    def get_document_hash(self, document):
        """Get stored content hash for a document, or None."""
        return self._document_hashes.get(document)

    def set_document_hash(self, document, hash_val):
        """Store a content hash for a document."""
        self._document_hashes[document] = hash_val

    def remove_document_hash(self, document):
        """Remove the content hash for a document. Returns True if existed."""
        if document in self._document_hashes:
            del self._document_hashes[document]
            return True
        return False

    def tracked_documents(self):
        """Get all tracked document paths. Returns JSON array."""
        return json.dumps(list(self._document_hashes.keys()))

    def get_statistics(self):
        """Returns JSON statistics."""
        # Count entities by type
        by_type = {}
        for entity in self._entities.values():
            etype = entity.get("entity_type", "Unknown")
            by_type[etype] = by_type.get(etype, 0) + 1
        # Count unique source documents
        docs = set()
        for chunk in self._chunks.values():
            doc = chunk.get("source_document")
            if doc:
                docs.add(doc)
        return json.dumps({
            "entity_count": len(self._entities),
            "edge_count": len(self._edges),
            "chunk_count": len(self._chunks),
            "entities_by_type": by_type,
            "documents_indexed": len(docs),
        })


def _build_test_graph():
    """Build a small test graph with known structure.

    Graph:
      John (Person) --[Decision chunk]-- Dashboard (Project)
      John (Person) --[Discussion chunk]-- Jane (Person)
      Jane (Person) --[ActionItem chunk]-- Dashboard (Project)
      Dashboard (Project) --[Background chunk]-- React (Technology)
    """
    kg = FakeKnowledgeGraph()

    kg.add_test_entity("e-john", "John Doe", "Person",
                       source_chunks=["c1", "c2"],
                       source_documents=["meetings/q3-review.md"])
    kg.add_test_entity("e-jane", "Jane Smith", "Person",
                       source_chunks=["c2", "c3"],
                       source_documents=["meetings/q3-review.md"])
    kg.add_test_entity("e-dash", "Dashboard Redesign", "Project",
                       source_chunks=["c1", "c3", "c4"],
                       source_documents=["meetings/q3-review.md", "work/dashboard.md"])
    kg.add_test_entity("e-react", "React", "Technology",
                       source_chunks=["c4"],
                       source_documents=["work/dashboard.md"])

    ts = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())

    kg.add_test_chunk("c1",
                      'John Doe: "We need to abandon the dual-axis chart approach."',
                      chunk_type="Decision",
                      source_document="meetings/q3-review.md",
                      section_name="Q3 Design Review",
                      speaker="John Doe",
                      timestamp=ts,
                      tags=["e-john", "e-dash"])

    kg.add_test_chunk("c2",
                      "John and Jane discussed the timeline for the dashboard project.",
                      chunk_type="Discussion",
                      source_document="meetings/q3-review.md",
                      section_name="Q3 Design Review",
                      timestamp=ts,
                      tags=["e-john", "e-jane"])

    kg.add_test_chunk("c3",
                      "Action: Jane Smith to implement new bar chart component by October 21.",
                      chunk_type="ActionItem",
                      source_document="meetings/q3-review.md",
                      section_name="Q3 Design Review",
                      timestamp=ts,
                      tags=["e-jane", "e-dash"])

    kg.add_test_chunk("c4",
                      "The dashboard will be built using React with D3 for charting.",
                      chunk_type="Background",
                      source_document="work/dashboard.md",
                      timestamp=ts,
                      tags=["e-dash", "e-react"])

    # Edges matching the chunks
    kg.add_test_edge("e-john", "e-dash", "c1", "Decision", ts, "meetings/q3-review.md")
    kg.add_test_edge("e-john", "e-jane", "c2", "Discussion", ts, "meetings/q3-review.md")
    kg.add_test_edge("e-jane", "e-dash", "c3", "ActionItem", ts, "meetings/q3-review.md")
    kg.add_test_edge("e-dash", "e-react", "c4", "Background", ts, "work/dashboard.md")

    return kg


def _make_mock_embedding_manager():
    """Create an EmbeddingManager mock with deterministic embeddings."""
    mock = MagicMock()

    def fake_generate(texts):
        return [_deterministic_embedding(t) for t in texts]

    mock.generate_embedding.side_effect = fake_generate
    mock.entity_embeddings_cache = {}

    def fake_build_descriptor(canonical_name, entity_type, top_cooccurrences):
        descriptor = f"{canonical_name} ({entity_type})"
        if top_cooccurrences:
            descriptor += ": " + ", ".join(top_cooccurrences)
        return descriptor

    mock.build_entity_descriptor.side_effect = fake_build_descriptor

    def fake_embed_entities(entities, knowledge_graph):
        for entity in entities:
            eid = entity["id"]
            if eid not in mock.entity_embeddings_cache:
                descriptor = fake_build_descriptor(
                    entity["canonical_name"],
                    entity["entity_type"],
                    [],
                )
                mock.entity_embeddings_cache[eid] = _deterministic_embedding(descriptor)

    mock.embed_entities.side_effect = fake_embed_entities

    def fake_find_relevant(query, entity_ids, top_n=10):
        query_emb = _deterministic_embedding(query)
        scores = []
        for eid in entity_ids:
            if eid in mock.entity_embeddings_cache:
                emb = mock.entity_embeddings_cache[eid]
                sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
                scores.append((eid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    mock.find_relevant_entities_scored.side_effect = fake_find_relevant

    def fake_invalidate(entity_id=None):
        if entity_id is not None:
            mock.entity_embeddings_cache.pop(entity_id, None)
        else:
            mock.entity_embeddings_cache.clear()

    mock.invalidate_entity_cache.side_effect = fake_invalidate

    return mock


def _make_memory_context_manager(kg=None, config=None, max_tokens=10000):
    """Create a MemoryContextManager with mocked tiktoken and embedding manager."""
    if kg is None:
        kg = _build_test_graph()
    if config is None:
        config = CortexConfig()

    mock_embed = _make_mock_embedding_manager()

    with patch("cortex.context.tiktoken") as mock_tiktoken:
        mock_encoder = MagicMock()
        # Approximate token count: len(text) // 4
        mock_encoder.encode.side_effect = lambda text: list(range(len(text) // 4))
        mock_encoder.decode.side_effect = lambda tokens: "x" * (len(tokens) * 4)
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        from cortex.context import MemoryContextManager
        mcm = MemoryContextManager(
            knowledge_graph=kg,
            embedding_manager=mock_embed,
            config=config,
            model="gpt-4",
            max_tokens=max_tokens,
        )
        return mcm


# ═══════════════════════════════════════════════════════════════════════════════
# Entity Embedding Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEntityEmbeddings:
    def test_build_entity_descriptor_format(self):
        """Descriptor includes name, type, and co-occurrences."""
        from cortex.embeddings import EmbeddingManager

        # Test via the real method signature
        descriptor = EmbeddingManager.build_entity_descriptor(
            None,  # self
            "John Doe", "Person", ["Dashboard Redesign", "React"]
        )
        assert descriptor == "John Doe (Person): Dashboard Redesign, React"

    def test_build_entity_descriptor_no_cooccurrences(self):
        """Descriptor without co-occurrences omits the colon."""
        from cortex.embeddings import EmbeddingManager
        descriptor = EmbeddingManager.build_entity_descriptor(
            None, "React", "Technology", []
        )
        assert descriptor == "React (Technology)"

    def test_embed_entities_caches_results(self):
        """embed_entities populates the cache for uncached entities."""
        mock_embed = _make_mock_embedding_manager()
        kg = _build_test_graph()

        entities = [
            {"id": "e-john", "canonical_name": "John Doe", "entity_type": "Person"},
            {"id": "e-jane", "canonical_name": "Jane Smith", "entity_type": "Person"},
        ]
        mock_embed.embed_entities(entities, kg)
        assert "e-john" in mock_embed.entity_embeddings_cache
        assert "e-jane" in mock_embed.entity_embeddings_cache

    def test_find_relevant_entities_returns_sorted(self):
        """find_relevant_entities_scored returns results sorted by score descending."""
        mock_embed = _make_mock_embedding_manager()
        kg = _build_test_graph()

        entities = [
            {"id": "e-john", "canonical_name": "John Doe", "entity_type": "Person"},
            {"id": "e-dash", "canonical_name": "Dashboard Redesign", "entity_type": "Project"},
        ]
        mock_embed.embed_entities(entities, kg)
        results = mock_embed.find_relevant_entities_scored(
            "dashboard project", ["e-john", "e-dash"], top_n=2
        )
        assert len(results) == 2
        # Results should be sorted by score descending
        assert results[0][1] >= results[1][1]

    def test_invalidate_clears_cache(self):
        """invalidate_entity_cache clears specific or all entries."""
        mock_embed = _make_mock_embedding_manager()
        kg = _build_test_graph()

        entities = [
            {"id": "e-john", "canonical_name": "John Doe", "entity_type": "Person"},
            {"id": "e-jane", "canonical_name": "Jane Smith", "entity_type": "Person"},
        ]
        mock_embed.embed_entities(entities, kg)
        assert len(mock_embed.entity_embeddings_cache) == 2

        # Invalidate one
        mock_embed.invalidate_entity_cache("e-john")
        assert "e-john" not in mock_embed.entity_embeddings_cache
        assert "e-jane" in mock_embed.entity_embeddings_cache

        # Invalidate all
        mock_embed.invalidate_entity_cache()
        assert len(mock_embed.entity_embeddings_cache) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor Phase Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnchorPhase:
    def test_find_anchors_returns_entities(self):
        """_find_anchors returns entity IDs with scores."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()
        anchors = mcm._find_anchors("John Doe dashboard", anchor_count=3)
        assert len(anchors) > 0
        assert len(anchors) <= 3
        # Each anchor is (entity_id, score)
        for eid, score in anchors:
            assert isinstance(eid, str)
            assert isinstance(score, float)

    def test_find_anchors_respects_count(self):
        """_find_anchors returns at most anchor_count results."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()
        anchors = mcm._find_anchors("dashboard", anchor_count=2)
        assert len(anchors) <= 2

    def test_find_anchors_scores_sorted(self):
        """Anchor results are sorted by combined score descending."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()
        anchors = mcm._find_anchors("John dashboard project", anchor_count=4)
        if len(anchors) > 1:
            for i in range(len(anchors) - 1):
                assert anchors[i][1] >= anchors[i + 1][1]

    def test_find_anchors_empty_graph(self):
        """_find_anchors returns empty list for empty graph."""
        kg = FakeKnowledgeGraph()
        mcm = _make_memory_context_manager(kg=kg)
        mcm._ensure_entity_embeddings()
        anchors = mcm._find_anchors("anything", anchor_count=5)
        assert anchors == []


# ═══════════════════════════════════════════════════════════════════════════════
# Expand Phase Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpandPhase:
    def test_expand_returns_entities_and_chunks(self):
        """_expand_neighborhood returns entities, edges, and chunks."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        assert "entities" in expanded
        assert "edges" in expanded
        assert "chunks" in expanded
        # John should be in the result
        assert "e-john" in expanded["entities"]

    def test_expand_1hop_from_john(self):
        """1-hop from John finds Jane and Dashboard."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=1, neighborhood_max_hops=1,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        entity_ids = set(expanded["entities"].keys())
        # John's direct neighbors are Jane and Dashboard
        assert "e-john" in entity_ids
        assert "e-jane" in entity_ids or "e-dash" in entity_ids

    def test_expand_2hop_finds_react(self):
        """2-hop from John reaches React (via Dashboard)."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=1, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        entity_ids = set(expanded["entities"].keys())
        assert "e-react" in entity_ids

    def test_expand_temporal_filter(self):
        """Temporal filtering excludes edges outside the time range."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=1, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )

        # Set time_start far in the future — should find no neighbors
        future_ts = int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())
        expanded = mcm._expand_neighborhood(["e-john"], params, future_ts, None)
        # Only the anchor itself should remain
        assert len(expanded["entities"]) == 1

    def test_expand_entity_cap(self):
        """neighborhood_max_entities caps the result count."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=1, neighborhood_max_hops=3,
            neighborhood_max_entities=2,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        assert len(expanded["entities"]) <= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Hop Reconstruction Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHopReconstruction:
    def test_single_anchor_hops(self):
        """BFS from single anchor assigns correct hop distances."""
        mcm = _make_memory_context_manager()
        edges = [
            ("A", "B", {}),
            ("B", "C", {}),
            ("C", "D", {}),
        ]
        hops = mcm._reconstruct_hops(["A"], edges)
        assert hops["A"] == 0
        assert hops["B"] == 1
        assert hops["C"] == 2
        assert hops["D"] == 3

    def test_multiple_anchors_min_hop(self):
        """With multiple anchors, entities get the minimum hop distance."""
        mcm = _make_memory_context_manager()
        edges = [
            ("A", "C", {}),
            ("B", "C", {}),
            ("C", "D", {}),
        ]
        # C is 1 hop from both A and B
        hops = mcm._reconstruct_hops(["A", "B"], edges)
        assert hops["C"] == 1
        assert hops["D"] == 2

    def test_disconnected_entity(self):
        """Disconnected entities are not in hop map."""
        mcm = _make_memory_context_manager()
        edges = [
            ("A", "B", {}),
        ]
        hops = mcm._reconstruct_hops(["A"], edges)
        assert "A" in hops
        assert "B" in hops
        # Entity "C" not connected
        assert "C" not in hops


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk Scoring Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkScoring:
    def test_decision_scores_higher_than_background(self):
        """Decision chunks score higher than Background at same distance."""
        mcm = _make_memory_context_manager()
        hop_map = {"e-john": 0, "e-dash": 1}

        decision_chunk = {
            "chunk_type": "Decision",
            "tags": ["e-john"],
        }
        background_chunk = {
            "chunk_type": "Background",
            "tags": ["e-john"],
        }

        score_decision = mcm._score_chunk(decision_chunk, hop_map)
        score_background = mcm._score_chunk(background_chunk, hop_map)
        assert score_decision > score_background

    def test_closer_entity_scores_higher(self):
        """Chunk tagged with closer entity scores higher."""
        mcm = _make_memory_context_manager()
        hop_map = {"e-john": 0, "e-react": 3}

        close_chunk = {
            "chunk_type": "Discussion",
            "tags": ["e-john"],
        }
        far_chunk = {
            "chunk_type": "Discussion",
            "tags": ["e-react"],
        }

        score_close = mcm._score_chunk(close_chunk, hop_map)
        score_far = mcm._score_chunk(far_chunk, hop_map)
        assert score_close > score_far

    def test_untagged_chunk_scores_low(self):
        """Chunk with no tags gets a very low score."""
        mcm = _make_memory_context_manager()
        hop_map = {"e-john": 0}

        chunk = {"chunk_type": "Discussion", "tags": []}
        score = mcm._score_chunk(chunk, hop_map)
        assert score < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1: Knowledge Map Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTier1Map:
    def test_knowledge_map_format(self):
        """Tier 1 includes bold entity names and co-occurrence info."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=5000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        tier1 = mcm._build_knowledge_map(expanded, 5000)

        assert "**John Doe**" in tier1
        assert "(Person)" in tier1

    def test_knowledge_map_sorted_by_weight(self):
        """Entities with higher weight appear first in Tier 1."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=5000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=1, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        tier1 = mcm._build_knowledge_map(expanded, 5000)

        # John (anchor, hop 0) should appear before React (hop 2)
        if "**React**" in tier1:
            john_pos = tier1.index("**John Doe**")
            react_pos = tier1.index("**React**")
            assert john_pos < react_pos

    def test_knowledge_map_budget_truncation(self):
        """Tier 1 respects the token budget."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=5000, tier2_tokens=3000, tier3_tokens=1000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        # Very tight budget
        tier1 = mcm._build_knowledge_map(expanded, 10)
        # Should include at most 1 entity line or be empty
        lines = [l for l in tier1.split("\n") if l.startswith("- **")]
        assert len(lines) <= 2  # Might fit 1 entity with tight budget


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2: Evidence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTier2Evidence:
    def test_evidence_format(self):
        """Tier 2 includes chunk type tags and quoted text."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=5000, tier3_tokens=1000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        tier2, entity_ids = mcm._build_evidence_tier(expanded, 5000)

        assert "**[" in tier2  # Chunk type header
        assert ">" in tier2  # Quoted text
        assert isinstance(entity_ids, set)

    def test_evidence_returns_entity_ids(self):
        """Tier 2 returns the set of entity IDs mentioned in evidence chunks."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=5000, tier3_tokens=1000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        _, entity_ids = mcm._build_evidence_tier(expanded, 5000)
        # Should include entity IDs from chunk tags
        assert len(entity_ids) > 0

    def test_evidence_budget_limits_chunks(self):
        """Tier 2 stops adding chunks when budget is exhausted."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=5000, tier3_tokens=1000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)

        # Very tight budget
        tier2_tight, _ = mcm._build_evidence_tier(expanded, 10)
        # Full budget
        tier2_full, _ = mcm._build_evidence_tier(expanded, 50000)

        assert len(tier2_tight) <= len(tier2_full)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3: Entity Summaries Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTier3Summaries:
    def test_summaries_exclude_tier2_entities(self):
        """Tier 3 only includes entities NOT prominently in Tier 2."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=5000, tier3_tokens=5000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        _, tier2_ids = mcm._build_evidence_tier(expanded, 5000)
        tier3 = mcm._build_entity_summaries(expanded, tier2_ids, 5000)

        # Entities that are in tier2 should not appear in tier3
        for eid in tier2_ids:
            name = expanded["entities"].get(eid, {}).get("canonical_name", "")
            if name:
                assert f"**{name}**" not in tier3

    def test_summaries_format(self):
        """Tier 3 entries start with '- **Name** (Type)'."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=5000, tier3_tokens=5000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)
        # Use empty tier2_ids so all entities go to tier3
        tier3 = mcm._build_entity_summaries(expanded, set(), 5000)

        if tier3:
            lines = [l for l in tier3.split("\n") if l.strip()]
            for line in lines:
                assert line.startswith("- **")

    def test_summaries_budget(self):
        """Tier 3 respects token budget."""
        mcm = _make_memory_context_manager()
        mcm._ensure_entity_embeddings()

        from cortex.context import MemoryContextParams
        params = MemoryContextParams(
            tier1_tokens=1000, tier2_tokens=5000, tier3_tokens=5000,
            anchor_count=3, neighborhood_max_hops=2,
            neighborhood_max_entities=20,
        )
        expanded = mcm._expand_neighborhood(["e-john"], params, None, None)

        tight = mcm._build_entity_summaries(expanded, set(), 5)
        full = mcm._build_entity_summaries(expanded, set(), 50000)
        assert len(tight) <= len(full)


# ═══════════════════════════════════════════════════════════════════════════════
# Budget Allocation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBudgetAllocation:
    def test_tier_budgets_sum_to_total(self):
        """Tier 1 + Tier 2 + Tier 3 budgets equal total budget."""
        mcm = _make_memory_context_manager()
        params = mcm._compute_adaptive_params(10000)
        assert params.tier1_tokens + params.tier2_tokens + params.tier3_tokens == 10000

    def test_default_percentages(self):
        """Default config allocates 10%/60%/30%."""
        mcm = _make_memory_context_manager()
        params = mcm._compute_adaptive_params(10000)
        assert params.tier1_tokens == 1000
        assert params.tier2_tokens == 6000
        assert params.tier3_tokens == 3000

    def test_custom_percentages(self):
        """Custom config overrides tier budgets."""
        config = CortexConfig(
            tier1_budget_pct=0.20,
            tier2_budget_pct=0.50,
        )
        mcm = _make_memory_context_manager(config=config)
        params = mcm._compute_adaptive_params(10000)
        assert params.tier1_tokens == 2000
        assert params.tier2_tokens == 5000
        assert params.tier3_tokens == 3000


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Params Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdaptiveParams:
    def test_tiny_graph_params(self):
        """Tiny graph (4 entities) gets minimum anchor count."""
        mcm = _make_memory_context_manager()
        params = mcm._compute_adaptive_params(10000)
        assert params.anchor_count == 3  # min

    def test_medium_graph_params(self):
        """Graph with 50 entities gets scaled parameters."""
        kg = FakeKnowledgeGraph()
        for i in range(50):
            kg.add_test_entity(f"e-{i}", f"Entity {i}")
        # Add enough edges to make density > 3
        for i in range(160):
            kg.add_test_edge(f"e-{i % 50}", f"e-{(i + 1) % 50}")

        mcm = _make_memory_context_manager(kg=kg)
        params = mcm._compute_adaptive_params(10000)
        assert params.anchor_count == 5  # 50 // 10
        assert params.neighborhood_max_hops == 2  # dense graph

    def test_large_graph_params(self):
        """Graph with 100+ entities gets larger params."""
        kg = FakeKnowledgeGraph()
        for i in range(120):
            kg.add_test_entity(f"e-{i}", f"Entity {i}")
        # Sparse graph (few edges)
        for i in range(50):
            kg.add_test_edge(f"e-{i}", f"e-{i + 1}")

        mcm = _make_memory_context_manager(kg=kg)
        params = mcm._compute_adaptive_params(10000)
        assert params.anchor_count == 10  # capped at max
        assert params.neighborhood_max_hops == 3  # sparse graph


# ═══════════════════════════════════════════════════════════════════════════════
# Full Assembly Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullAssembly:
    def test_assemble_returns_markdown(self):
        """assemble_context returns markdown with Knowledge Context header."""
        mcm = _make_memory_context_manager(max_tokens=50000)
        result = mcm.assemble_context("What did John decide about the dashboard?")
        assert "## Knowledge Context" in result

    def test_assemble_has_all_tiers(self):
        """Output contains all three tier headers."""
        mcm = _make_memory_context_manager(max_tokens=50000)
        result = mcm.assemble_context("John Doe dashboard decisions")
        assert "### Key Entities" in result
        assert "### Evidence" in result
        # Tier 3 may or may not be present depending on whether
        # entities are in Tier 2

    def test_assemble_empty_graph(self):
        """Empty graph returns empty string."""
        kg = FakeKnowledgeGraph()
        mcm = _make_memory_context_manager(kg=kg, max_tokens=50000)
        result = mcm.assemble_context("anything")
        assert result == ""

    def test_assemble_time_filter(self):
        """Temporal filter restricts results."""
        mcm = _make_memory_context_manager(max_tokens=50000)

        # All data is from 2024-10-14
        # Filter to only 2030+ — should get very limited results
        future_ts = int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())
        result = mcm.assemble_context("John dashboard", time_start=future_ts)

        # Result should be minimal — anchors exist but no neighborhood
        # Still gets the header since anchors are found
        if result:
            # Evidence section should be empty or minimal
            assert "### Evidence" not in result or result.count(">") == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Date Formatting Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDateFormatting:
    def test_format_date_valid(self):
        """Valid timestamp formats correctly."""
        mcm = _make_memory_context_manager()
        ts = int(datetime(2024, 10, 14, tzinfo=timezone.utc).timestamp())
        assert mcm._format_date(ts) == "2024-10-14"

    def test_format_date_none(self):
        """None timestamp returns 'unknown'."""
        mcm = _make_memory_context_manager()
        assert mcm._format_date(None) == "unknown"

    def test_format_date_invalid(self):
        """Invalid timestamp returns 'unknown'."""
        mcm = _make_memory_context_manager()
        # Very large timestamp that would overflow
        assert mcm._format_date(99999999999999) == "unknown"
