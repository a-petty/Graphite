# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prime Directive

**This project is an active transformation of Atlas (a code intelligence system) into Cortex (a knowledge graph memory system for LLMs).** The codebase currently contains Atlas code that is being incrementally rewritten. Every task should be evaluated against the implementation plan in `cortex-implementation-plan-2.md`.

**Key framing:**
- Atlas builds dependency graphs from **source code** using Tree-sitter parsing.
- Cortex builds knowledge graphs from **natural language documents** (meetings, associate profiles, completed work) using a **tag-and-index** architecture: LLM-based entity tagging + co-occurrence edges, with relationships inferred at query time from evidence chunks.
- The Rust+Python+PyO3 architecture, petgraph core, PageRank, embeddings, Anchor & Expand context assembly, and MCP server framework all carry over. The **domain** changes, not the architecture.

**When working on any task, always ask:** "Does this move us from Atlas toward Cortex, or does it entrench Atlas patterns we plan to remove?"

## Working Style

The project owner is a systems thinker without formal SWE background. Their engineering team validates code but Claude is the primary day-to-day collaborator. This means:

- **Explain what you built.** Every code change gets a plain-English walkthrough — what it does, why, and any caveats. Do not make the user ask for this.
- **Ask before assuming.** If requirements are ambiguous or context is missing (data structures, constraints, how a feature connects to existing code), stop and ask. Do not silently fill gaps.
- **One thing at a time.** Deliver one piece, explain it, let the user confirm it works, then move on. Compound deliverables create compound errors.
- **State your confidence level.** Say "I'm confident this is correct" vs "this should work but test with X" vs "I'm uncertain here — flag for engineer review." Never present uncertain output with certain tone.
- **Flag for engineer review:** authentication, passwords, payments, sensitive data, file uploads, destructive operations, external API integrations, database migrations, deployment config, or anything you're not fully confident in.
- **Prefer simple over clever.** Verbose and obvious beats compact and subtle. Well-established over novel. Explicit error handling over assuming success.
- **Never skip validation or error handling**, even in prototypes. Never store passwords in plain text, build SQL from string concatenation, or expose internal errors to users.

## Implementation Phases (from cortex-implementation-plan-2.md)

The transformation follows 8 phases. Track current progress against this list:

| Phase | Summary | Status |
|-------|---------|--------|
| **Phase 0** | Fork & rename (`atlas` → `cortex`), strip tree-sitter, stub modules | **Complete** |
| **Phase 1** | Core graph schema + persistence — `EntityNode`/`CoOccurrenceEdge` replace `FileNode`/`EdgeKind`, `KnowledgeGraph` replaces `RepoGraph`, chunk storage, save/load, PyO3 bindings | **Complete** |
| **Phase 2** | Three-pass tag-and-index pipeline — structural parse (no LLM) → classify (LLM) → tag entities (LLM), co-occurrence graph construction, test corpus | **Complete** |
| **Phase 3** | Context assembly adaptation — Anchor & Expand for knowledge, evidence chunk retrieval, temporal filtering, entity summaries | **Complete** |
| **Phase 4** | MCP server tool redesign — new knowledge-centric tool set | **Complete** |
| **Phase 5** | Reflection & consolidation — entity merging, decay scoring, stale evidence detection, synthesis | **Complete** |
| **Phase 6** | Incremental update handling — two-tier document change/delete model | Not started |
| **Phase 7** | Evaluation framework — synthetic test corpus, benchmark suite, RAG baseline comparison | Not started |

**Update the Status column as phases are completed.** The plan document has full details for each phase.

## Build & Development

Cortex is a hybrid Rust + Python project using PyO3/Maturin for FFI (inherited from Atlas).

```bash
# Initial setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
maturin develop            # Build Rust core as Python extension

# After ANY Rust code change, rebuild before testing:
maturin develop            # Must run from project root (where pyproject.toml lives)
```

### Testing

```bash
# Rust tests (run from rust_core/)
cd rust_core && cargo test
cargo test --test test_graph_construction                 # Single test file
cargo test --test test_graph_updates test_update_file_no_change  # Single test function

# Python tests (run from project root)
pytest python_shell/tests/
pytest python_shell/tests/test_context.py                # Single file
pytest python_shell/tests/test_context.py::test_name     # Single test

# Rust benchmarks (Criterion)
cd rust_core && cargo bench
```

### Linting & Formatting

```bash
# Rust (from rust_core/)
cargo fmt --check          # Check formatting (rustfmt)
cargo fmt                  # Auto-format
cargo clippy               # Lint

# Python
mypy python_shell/cortex/
```

### Running

```bash
cortex-mcp --project-root /path/to/repo          # MCP server
cortex-mcp --project-root /path/to/repo --verbose # With debug logging
cortex watch /path/to/repo                        # File watcher mode
cortex query "question" -p /path/to/repo          # One-shot query
```

## Architecture

### Current state (Phase 0 complete)

Two layers connected via PyO3. Tree-sitter and code-analysis modules (CPG, CFG, dataflow, callgraph) have been stripped. Graph infrastructure (petgraph, PageRank, file watcher) is retained but stubbed — nodes exist, no edges until Phase 1.

**Rust core** (`rust_core/src/`, edition 2024) — Stubbed graph with petgraph DiGraph, PageRank, file watcher. Compiled extension importable as `cortex.semantic_engine`.

**Python shell** (`python_shell/cortex/`) — MCP server (graph tools retained, CPG tools removed), context assembly, embeddings, LLM clients.

### Target state (Cortex)

Same two-layer Rust+Python architecture, but the domain shifts from code to knowledge:

**Rust core** — `KnowledgeGraph` wrapping `petgraph::DiGraph<EntityNode, CoOccurrenceEdge>`, tag index with alias/embedding lookup, chunk storage, temporal filtering, decay scoring, PageRank, persistence (MessagePack). No Tree-sitter.

**Python shell** (`python_shell/cortex/`) — Three-pass tag-and-index pipeline (structural parse → classify → tag), MCP server (knowledge tools), Anchor & Expand with evidence chunk retrieval, reflection/consolidation agent, embeddings (reused).

### Key data flow (target)

1. Documents land in `memory/{meetings,associates,work}/`
2. **Three-pass pipeline** (Python): structural parse (deterministic chunking) → classify chunks via LLM → tag entities via LLM → disambiguate against existing graph
3. **KnowledgeGraph** (Rust) stores `EntityNode`s and `CoOccurrenceEdge`s in a petgraph DiGraph. Edges link entity pairs that co-occur within the same chunk. Chunk text is stored alongside the graph for O(1) evidence retrieval.
4. **EmbeddingManager** (Python, reused) embeds entity contextual descriptors with BAAI/bge-small-en-v1.5 via FastEmbed
5. **ContextManager** (Python, adapted) runs Anchor & Expand: semantic search finds relevant entities (anchor), BFS walks the co-occurrence graph with temporal filtering (expand), retrieves evidence chunks, assembles 3-tier response:
   - **Tier 1** (~10% budget): PageRank-ordered knowledge map
   - **Tier 2** (40-70%): Evidence chunks (actual text where entities co-occur)
   - **Tier 3** (remainder): Compressed entity summaries

### Atlas → Cortex component mapping

| Atlas Component | Cortex Replacement | Action |
|---|---|---|
| Tree-sitter parser (`parser.rs`, `queries/`) | Three-pass tag-and-index pipeline (Python) | **BUILD** |
| `RepoGraph` / `FileNode` / `EdgeKind` (`graph.rs`) | `KnowledgeGraph` / `EntityNode` / `CoOccurrenceEdge` | **ADAPT** |
| `CpgLayer` (`cpg.rs`) | *Removed — not needed for tag-and-index* | **REMOVED** |
| `CFG` / `dataflow` / `callgraph` | *Removed — co-occurrence replaces all* | **REMOVED** |
| `SymbolIndex` (`symbol_table.rs`) | Tag index + disambiguation | **ADAPT** |
| Import resolvers (`import_resolver.rs`) | *Removed — coreference optional, handled in Python* | **REMOVED** |
| Context manager (`context.py`) | Memory context assembler + evidence chunk retrieval | **ADAPT** |
| MCP server (`mcp_server.py`) | MCP server — knowledge tools | **ADAPT** |
| Agent (`agent.py`) | Memory agent + reflection loop | **ADAPT** |
| Embeddings (`embeddings.py`) | Entity + chunk embeddings | **REUSE** |
| LLM clients (`llm.py`) | LLM clients (classification + tagging) | **REUSE** |
| File watcher (`watcher.rs`) | Directory watcher | **REUSE** |
| PageRank (`graph.rs`) | Entity importance scoring | **REUSE** |

### Design invariants (carried over and adapted)

- **Lazy initialization**: Graph builds on first tool call, embeddings on first semantic search. Never eagerly.
- **Tag-and-index, not triple extraction**: The graph stores co-occurrence edges (which entities appear together in which chunks), not typed predicates. Relationships are inferred at query time by the consuming LLM reading evidence chunks.
- **Conservative disambiguation**: Only auto-merge entities at >0.90 confidence. Maintain audit trail. Borderline cases (0.7–0.85) go to human review queue.
- **Post-extraction validation**: Verify tagged entity names appear in source chunk text. Remove hallucinated entities.
- **Incremental updates**: Two-tier model — DocumentChanged (re-run pipeline for that file) and DocumentDeleted (clean up chunks/edges/entities).
- **petgraph swap-remove**: `remove_node()` uses swap-remove, so all side-maps (tag index, chunk references) must be explicitly remapped when removing nodes.
- **Persistence required**: Graph is saved after ingestion and periodically during watch mode. Can't cheaply rebuild since extraction requires LLM calls.

## Target directory structure

```
cortex/
├── rust_core/src/
│   ├── knowledge_graph.rs   # KnowledgeGraph (replaces RepoGraph)
│   ├── entity.rs            # EntityNode, EntityType, MemoryCategory
│   ├── chunk.rs             # Chunk storage and metadata
│   ├── cooccurrence.rs      # CoOccurrenceEdge types
│   ├── persistence.rs       # Save/load/crash recovery (MessagePack)
│   ├── graph.rs             # Retained: core petgraph utilities
│   ├── watcher.rs           # Retained: file system monitoring
│   ├── symbol_table.rs      # Adapted → tag_index.rs
│   └── lib.rs               # Adapted: PyKnowledgeGraph PyO3 bindings
├── python_shell/cortex/
│   ├── extraction/          # Three-pass tag-and-index pipeline
│   │   ├── structural_parser.py  # Pass 1: deterministic chunking (no LLM)
│   │   ├── classifier.py         # Pass 2: LLM chunk classification
│   │   ├── tagger.py             # Pass 3: LLM entity tagging
│   │   └── prompts/              # LLM prompt templates
│   │       ├── classify.txt
│   │       └── tag.txt
│   ├── ingestion/           # Document ingestion
│   │   ├── pipeline.py      # 3-pass orchestrator
│   │   ├── chunker.py       # Document chunking utilities
│   │   └── categorizer.py   # Directory → memory category
│   ├── reflection/          # Background consolidation
│   │   ├── consolidator.py  # Merge, prune, decay
│   │   └── synthesizer.py   # Higher-order insights
│   ├── context.py           # Adapted Anchor & Expand + evidence chunk retrieval
│   ├── embeddings.py        # Retained: FastEmbed
│   ├── mcp_server.py        # Knowledge-centric tools
│   ├── agent.py             # Memory agent + reflection loop
│   ├── cli.py               # cortex ingest/query/watch/reflect
│   ├── llm.py               # Retained: Ollama/MLX clients
│   └── config.py            # .cortex.toml management
├── memory/                  # Default memory repository
│   ├── meetings/            # Episodic memory
│   ├── associates/          # Semantic memory
│   └── work/                # Procedural memory
├── .cortex.toml
├── pyproject.toml
└── cortex-implementation-plan-2.md  # Full implementation details (v4)
```

## Project config files

- `.cortex.toml` — Per-project config (models, extraction thresholds, reflection schedule, context budgets)
- `.cortexignore` — Per-project ignore patterns (like `.gitignore`)
- Prompt templates in `python_shell/cortex/extraction/prompts/`
