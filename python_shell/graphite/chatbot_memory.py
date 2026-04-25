"""Graphite Memory Engine for Chatbot LLMs.

Drop-in replacement for a flat memory system (e.g., MemoryManager) that uses
a knowledge graph for contextual recall instead of cosine similarity search.

Memories are ingested through the Graphite extraction pipeline (structural parse
→ classify → tag entities → build co-occurrence graph) on a background thread.
Retrieval uses the AgentContextAssembler (~200ms, no LLM calls) for fast,
graph-based context assembly every chat turn.

Usage:
    from graphite.chatbot_memory import GraphiteMemory

    memory = GraphiteMemory(lore_file="lore.json")

    memory.add_episodic_memory("User mentioned they love hiking in Colorado")
    memory.add_core_memory("User is a software engineer who enjoys outdoor activities")

    episodic, core = memory.get_relevant_memories("Tell me about hiking trails")
    context = memory.get_context("Tell me about hiking trails")

    memory.ingest_conversation_turns([
        {"role": "user", "content": "I went hiking in Rocky Mountain National Park"},
        {"role": "assistant", "content": "That sounds amazing! Which trail?"},
    ])

    memory.close()  # drain queue, save graph
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GraphiteMemory:
    """Knowledge graph memory engine for chatbot LLMs.

    Replaces flat JSONL + embedding similarity memory with a knowledge graph
    that captures entities, relationships, and co-occurrences from conversations.
    Retrieval uses graph traversal + PageRank re-ranking instead of cosine
    similarity, enabling contextual recall that vector search cannot provide.

    This class matches the MemoryManager interface for drop-in replacement:
        add_episodic_memory(text)
        add_core_memory(text)
        get_relevant_memories(query) → (episodic_list, core_list)
        get_relevant_lore(query) → list
        get_recent_episodic_memories(count) → list
        clear_episodic_memories()
        get_all_memories_text() → str

    Plus enhanced methods:
        get_context(query) → str  (rich graph context for prompt injection)
        ingest_conversation_turns(messages)  (raw turn ingestion)
        save() / close()
    """

    def __init__(
        self,
        graph_path: Optional[str] = None,
        config_path: Optional[str] = None,
        lore_file: Optional[str] = None,
        llm_model: str = "qwen2.5:14b",
        llm_provider: str = "ollama",
        auto_save_interval: int = 5,
    ):
        """Initialize Graphite memory engine.

        Args:
            graph_path: Path to persist the graph. Defaults to .graphite/graph.msgpack
                        in the current directory.
            config_path: Optional path to .graphite.toml for fine-tuning.
            lore_file: Optional path to a JSON array of background lore strings.
            llm_model: Ollama model name for extraction pipeline.
            llm_provider: "ollama" or "mlx".
            auto_save_interval: Save graph to disk every N ingestions.
        """
        from graphite.config import GraphiteConfig

        # ── Configuration ──
        if config_path and Path(config_path).exists():
            self._config = GraphiteConfig.from_toml(Path(config_path))
        else:
            self._config = GraphiteConfig()

        # Override LLM settings from constructor args
        self._config.llm_model = llm_model
        self._config.llm_provider = llm_provider

        # ── Graph path resolution ──
        # An explicit graph_path wins. Otherwise use config.graph_root (default ~).
        if graph_path:
            self._root_path = str(Path(graph_path).parent.parent)
            self._graph_path = graph_path
        else:
            self._root_path = str(Path(self._config.graph_root).expanduser().resolve())
            self._graph_path = os.path.join(self._root_path, ".graphite", "graph.msgpack")

        # Ensure .graphite directory exists
        os.makedirs(os.path.dirname(self._graph_path), exist_ok=True)

        # ── Knowledge Graph ──
        from graphite.semantic_engine import PyKnowledgeGraph

        if Path(self._graph_path).exists():
            logger.info("Loading existing knowledge graph from %s", self._graph_path)
            self._kg = PyKnowledgeGraph.from_path(self._root_path)
            logger.info("Graph loaded")
        else:
            logger.info("Creating new knowledge graph")
            self._kg = PyKnowledgeGraph(self._root_path)

        # ── Embedding Manager ──
        from graphite.embeddings import EmbeddingManager

        self._embedding_manager = EmbeddingManager()

        # ── Agent Context Assembler (retrieval — no LLM) ──
        from graphite.agent_context import AgentContextAssembler

        self._assembler = AgentContextAssembler(
            knowledge_graph=self._kg,
            embedding_manager=self._embedding_manager,
            config=self._config,
        )

        # ── LLM Client (for extraction pipeline) ──
        if llm_provider == "mlx":
            from graphite.llm import MLXClient
            self._llm_client = MLXClient(model=llm_model)
        else:
            from graphite.llm import OllamaClient
            self._llm_client = OllamaClient(model=llm_model)

        # ── Ingestion Pipeline ──
        from graphite.ingestion.pipeline import IngestionPipeline

        self._pipeline = IngestionPipeline(
            knowledge_graph=self._kg,
            llm_client=self._llm_client,
            embedding_manager=self._embedding_manager,
            config=self._config,
        )

        # ── Background Ingestion Thread ──
        self._auto_save_interval = auto_save_interval
        self._ingest_count = 0
        self._queue: queue.Queue = queue.Queue()
        self._shutdown = threading.Event()
        self._worker = threading.Thread(
            target=self._ingestion_worker, daemon=True, name="graphite-ingestion"
        )
        self._worker.start()

        # ── Lore Loading ──
        self._lore_file = lore_file
        if lore_file:
            self._load_lore(lore_file)

        # ── Save on exit ──
        atexit.register(self.close)

        # Log stats
        try:
            stats = json.loads(self._kg.get_statistics())
            logger.info(
                "Graphite memory ready: %d entities, %d chunks, %d documents",
                stats.get("entity_count", 0),
                stats.get("chunk_count", 0),
                stats.get("documents_indexed", 0),
            )
            print(
                f"[Graphite] Memory loaded: {stats.get('entity_count', 0)} entities, "
                f"{stats.get('chunk_count', 0)} chunks"
            )
        except Exception:
            print("[Graphite] Memory initialized (empty graph)")

    # ═══════════════════════════════════════════════════════════════════════
    # MemoryManager-Compatible Interface
    # ═══════════════════════════════════════════════════════════════════════

    def add_episodic_memory(self, summary_text: str) -> None:
        """Ingest a short-term memory into the knowledge graph.

        Queues the text for background extraction (classify → tag → graph write).
        Returns immediately without blocking.
        """
        if not summary_text or not summary_text.strip():
            return

        source_id = f"chatbot://episodic/{int(time.time() * 1000)}"
        self._queue.put((summary_text, source_id, "Episodic"))
        print(f"\n[Graphite] Queued episodic memory for extraction: {summary_text[:80]}...\n")

    def add_core_memory(self, summary_text: str) -> None:
        """Ingest a consolidated long-term memory into the knowledge graph.

        Queues the text for background extraction. Core memories are stored
        as Semantic category, giving them higher persistence.
        """
        if not summary_text or not summary_text.strip():
            return

        source_id = f"chatbot://core/{int(time.time() * 1000)}"
        self._queue.put((summary_text, source_id, "Semantic"))
        print(f"\n[Graphite] Queued core memory for extraction: {summary_text[:80]}...\n")

    def get_relevant_memories(
        self,
        query: str,
        top_k_episodic: int = 2,
        top_k_core: int = 2,
        min_similarity: float = 0.5,
    ) -> Tuple[List[str], List[str]]:
        """Find contextually relevant memories using knowledge graph traversal.

        Unlike vector similarity search, this finds memories connected through
        shared entities and co-occurrence relationships in the graph.

        Returns:
            (episodic_memories, core_memories) — lists of memory text strings,
            matching the MemoryManager return format.
        """
        if not query:
            return [], []

        try:
            # Use the agent context assembler for graph-based retrieval
            ctx = self._assembler.assemble(query, depth="full")

            episodic_list = []
            core_list = []

            # Extract evidence from recent events (these are chunk summaries)
            for event in ctx.recent_events:
                source = event.source if hasattr(event, "source") else ""
                if "episodic" in source or "conversation" in source:
                    episodic_list.append(event.summary)
                else:
                    core_list.append(event.summary)

            # Also build entity-level context strings for core memories
            for entity in ctx.entities:
                connections = ", ".join(entity.top_connections) if entity.top_connections else ""
                last = f" (last: {entity.last_seen})" if entity.last_seen else ""
                entry = f"{entity.name} ({entity.type})"
                if connections:
                    entry += f" — connected to {connections}"
                entry += last
                core_list.append(entry)

            return episodic_list[:top_k_episodic], core_list[:top_k_core]

        except Exception as e:
            logger.warning("Memory retrieval failed: %s", e)
            return [], []

    def get_relevant_lore(
        self, query: str, top_k: int = 2, min_similarity: float = 0.3
    ) -> List[str]:
        """Retrieve relevant background lore from the knowledge graph.

        Searches for chunks sourced from lore:// documents.
        """
        if not query:
            return []

        try:
            # Use the assembler to find entities related to the query,
            # then filter for lore-sourced chunks
            ctx = self._assembler.assemble(query, depth="full")
            lore_entries = []

            for event in ctx.recent_events:
                source = event.source if hasattr(event, "source") else ""
                if "lore" in source:
                    lore_entries.append(event.summary)

            # If no lore found via events, try direct chunk search
            if not lore_entries:
                lore_entries = self._search_lore_chunks(query, top_k)

            return lore_entries[:top_k]

        except Exception as e:
            logger.warning("Lore retrieval failed: %s", e)
            return []

    def get_recent_episodic_memories(self, count: int) -> List[Dict]:
        """Return the N most recent episodic memory chunks.

        Returns list of dicts with 'summary' and 'timestamp' keys,
        matching the MemoryManager format.
        """
        try:
            docs_json = self._kg.tracked_documents()
            docs = json.loads(docs_json)

            # Find episodic documents
            episodic_docs = [d for d in docs if d.startswith("chatbot://episodic/")]
            # Sort by timestamp in source_id (descending)
            episodic_docs.sort(reverse=True)

            recent = []
            for doc in episodic_docs[:count]:
                chunks_json = self._kg.get_chunks_by_document(doc)
                chunks = json.loads(chunks_json)
                for chunk in chunks:
                    recent.append({
                        "summary": chunk.get("text", ""),
                        "timestamp": chunk.get("timestamp"),
                    })

            # Sort by timestamp descending, limit to count
            recent.sort(
                key=lambda m: m.get("timestamp") or 0, reverse=True
            )
            return recent[:count]

        except Exception as e:
            logger.warning("Failed to get recent memories: %s", e)
            return []

    def clear_episodic_memories(self) -> None:
        """Remove all episodic memories from the knowledge graph.

        Cascade-removes chunks, co-occurrence edges, and orphaned entities.
        Entities shared with core memories or conversation turns are kept.
        """
        try:
            docs_json = self._kg.tracked_documents()
            docs = json.loads(docs_json)

            episodic_docs = [d for d in docs if d.startswith("chatbot://episodic/")]
            removed_count = 0

            for doc in episodic_docs:
                self._kg.remove_document(doc)
                self._kg.remove_document_hash(doc)
                removed_count += 1

            if removed_count > 0:
                self._assembler.invalidate_caches()
                self._embedding_manager.invalidate_entity_cache()
                self._kg.save(self._root_path)

            print(
                f"\n[Graphite] Cleared {removed_count} episodic memory documents "
                f"from knowledge graph.\n"
            )

        except Exception as e:
            logger.error("Failed to clear episodic memories: %s", e)

    def get_all_memories_text(self) -> str:
        """Return a formatted display of all knowledge in the graph."""
        try:
            stats = json.loads(self._kg.get_statistics())
            entity_count = stats.get("entity_count", 0)

            lines = [
                "--- Graphite Knowledge Graph ---",
                f"Entities: {entity_count}",
                f"Chunks: {stats.get('chunk_count', 0)}",
                f"Documents: {stats.get('documents_indexed', 0)}",
            ]

            # Show entities by type
            by_type = stats.get("entities_by_type", {})
            if by_type:
                lines.append("\nEntities by type:")
                for etype, count in sorted(by_type.items()):
                    lines.append(f"  {etype}: {count}")

            # Show top entities by PageRank
            if entity_count > 0:
                pagerank_json = self._kg.compute_pagerank()
                ranked = json.loads(pagerank_json)

                lines.append("\nTop entities (by importance):")
                for entity_id, score in ranked[:20]:
                    entity_json = self._kg.get_entity(entity_id)
                    if entity_json:
                        entity = json.loads(entity_json)
                        name = entity.get("canonical_name", entity_id)
                        etype = entity.get("entity_type", "Unknown")
                        chunks = len(entity.get("source_chunks", []))
                        lines.append(
                            f"  - {name} ({etype}): "
                            f"{chunks} mentions, importance {score:.3f}"
                        )

            return "\n".join(lines)

        except Exception as e:
            return f"Error displaying memories: {e}"

    # ═══════════════════════════════════════════════════════════════════════
    # Enhanced Methods (beyond MemoryManager)
    # ═══════════════════════════════════════════════════════════════════════

    def get_context(self, query: str, depth: str = "brief") -> str:
        """Return rich graph context formatted for prompt injection.

        This is the recommended retrieval method for chatbots that want
        structured knowledge context (entities, relationships, evidence)
        rather than flat memory lists.

        Args:
            query: The user's input or current conversation context.
            depth: "brief" (~100-200 tokens, every turn) or
                   "full" (~1000-5000 tokens, deep reasoning).

        Returns:
            Formatted markdown text ready for system prompt injection.
            Returns empty string if graph has no relevant knowledge.
        """
        if not query:
            return ""

        try:
            ctx = self._assembler.assemble(query, depth=depth)

            # Check if there's actually any content
            if not ctx.entities and not ctx.recent_events and not ctx.pending_items:
                return ""

            return ctx.to_injection_text()

        except Exception as e:
            logger.warning("Context assembly failed: %s", e)
            return ""

    def ingest_conversation_turns(self, messages: List[Dict]) -> None:
        """Ingest raw conversation turns for richer entity extraction.

        Call this when the chatbot's context window is trimmed and messages
        are about to be discarded. The raw turns contain far more entity
        signal than one-sentence memory summaries.

        Args:
            messages: List of {"role": "user|assistant", "content": "..."} dicts.
                      System messages are filtered out.
        """
        if not messages:
            return

        # Filter out system messages and format as speaker-labeled text
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system" or not content.strip():
                continue
            speaker = "User" if role == "user" else "Assistant"
            conversation_parts.append(f"**{speaker}:** {content}")

        if not conversation_parts:
            return

        formatted = "\n\n".join(conversation_parts)
        source_id = f"chatbot://conversation/{int(time.time() * 1000)}"
        self._queue.put((formatted, source_id, "Episodic"))
        logger.info("Queued %d conversation turns for extraction", len(conversation_parts))

    def save(self) -> None:
        """Persist the knowledge graph to disk immediately."""
        try:
            self._kg.save(self._root_path)
            logger.info("Graph saved to %s", self._graph_path)
        except Exception as e:
            logger.error("Failed to save graph: %s", e)

    def close(self) -> None:
        """Drain the ingestion queue, save, and shut down the background thread."""
        if self._shutdown.is_set():
            return

        self._shutdown.set()

        # Drain remaining items
        remaining = 0
        while not self._queue.empty():
            remaining += 1
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if remaining > 0:
            logger.info(
                "Shutdown: %d pending ingestion items were not processed", remaining
            )

        # Signal worker to stop
        self._queue.put(None)
        self._worker.join(timeout=5)

        # Final save
        self.save()
        print("[Graphite] Memory saved and shut down.")

    # ═══════════════════════════════════════════════════════════════════════
    # Internal Methods
    # ═══════════════════════════════════════════════════════════════════════

    def _ingestion_worker(self) -> None:
        """Background thread that processes the ingestion queue."""
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break  # Shutdown signal

            text, source_id, category = item

            try:
                result = self._pipeline.ingest_text(text, source_id, category)
                self._ingest_count += 1

                if result.status in ("complete", "partial"):
                    logger.info(
                        "Ingested %s: %d entities, %d edges (%.1fs)",
                        source_id,
                        result.entities_created + result.entities_linked,
                        result.edges_created,
                        result.duration_seconds,
                    )

                    # Invalidate caches so next retrieval sees new data
                    self._assembler.invalidate_caches()
                    self._embedding_manager.invalidate_entity_cache()

                    # Periodic save
                    if self._ingest_count % self._auto_save_interval == 0:
                        self.save()
                else:
                    errors = "; ".join(result.errors) if result.errors else "unknown"
                    logger.warning("Ingestion failed for %s: %s", source_id, errors)

            except Exception as e:
                logger.error("Ingestion error for %s: %s", source_id, e)

    def _load_lore(self, lore_file: str) -> None:
        """Load background lore from a JSON file into the knowledge graph.

        Only loads if the lore hasn't been ingested yet (checks document hash).
        """
        lore_path = Path(lore_file)
        if not lore_path.exists():
            logger.info("No lore file found at %s", lore_file)
            return

        try:
            with open(lore_path, "r") as f:
                lore_data = json.load(f)

            if not isinstance(lore_data, list) or not lore_data:
                return

            # Check if already loaded
            existing_hash = self._kg.get_document_hash("lore://manifest")
            import hashlib
            lore_hash = hashlib.sha256(json.dumps(lore_data).encode()).hexdigest()

            if existing_hash == lore_hash:
                logger.info("Lore already loaded (hash match)")
                return

            # Ingest each lore entry
            logger.info("Loading %d lore entries...", len(lore_data))
            for i, entry in enumerate(lore_data):
                if isinstance(entry, str) and entry.strip():
                    source_id = f"lore://entry/{i}"
                    self._queue.put((entry, source_id, "Semantic"))

            # Store manifest hash to skip re-loading next time
            self._kg.set_document_hash("lore://manifest", lore_hash)

            print(f"[Graphite] Queued {len(lore_data)} lore entries for extraction")

        except Exception as e:
            logger.warning("Failed to load lore: %s", e)

    def _search_lore_chunks(self, query: str, top_k: int) -> List[str]:
        """Direct search for lore-sourced chunks in the graph."""
        try:
            docs_json = self._kg.tracked_documents()
            docs = json.loads(docs_json)
            lore_docs = [d for d in docs if d.startswith("lore://")]

            all_chunks = []
            for doc in lore_docs:
                chunks_json = self._kg.get_chunks_by_document(doc)
                chunks = json.loads(chunks_json)
                all_chunks.extend(chunks)

            if not all_chunks:
                return []

            # Use embedding similarity to rank lore chunks against query
            texts = [c.get("text", "") for c in all_chunks]
            if not texts:
                return []

            embeddings = self._embedding_manager.generate_embedding(texts)
            query_embedding = self._embedding_manager.generate_embedding([query])[0]

            import numpy as np
            scores = []
            for i, emb in enumerate(embeddings):
                sim = float(np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
                ))
                scores.append((sim, texts[i]))

            scores.sort(key=lambda x: x[0], reverse=True)
            return [text for _, text in scores[:top_k]]

        except Exception as e:
            logger.warning("Lore chunk search failed: %s", e)
            return []
