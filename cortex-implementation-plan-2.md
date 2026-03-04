# Cortex: Implementation Plan v4

## From Atlas (Code Intelligence) → Cortex (Tag-and-Index Knowledge Memory for LLMs)

> **This is a revision of `cortex-implementation-plan.md` (v3).** It is a complete, standalone document — not a diff. See [Changelog from v3](#9-changelog-from-v3) at the end for a summary of what changed and why. For changes between earlier versions, see [Changelog from v2](#10-changelog-from-v2) and [Changelog from v1](#11-changelog-from-v1).

---

## Executive Summary

Cortex is a fork of Atlas — a Rust+Python MCP tool that builds dependency graphs from codebases — adapted to build persistent knowledge graphs from natural language documents (meetings, associate profiles, completed work). This plan maps every Atlas component to its Cortex counterpart, identifies what can be reused, what must be modified, and what must be built from scratch.

**Key difference from v3: Tag-and-index replaces triple extraction.** v3 proposed a five-stage LLM extraction pipeline culminating in subject-predicate-object triple extraction — the same approach used by LightRAG, Cognee, and Microsoft GraphRAG. Feasibility analysis revealed that triple extraction at achievable precision (~70%) compounds errors through multi-hop traversals (0.7³ ≈ 34% accuracy on 3-hop queries), making the graph confidently wrong rather than vaguely approximate. The tag-and-index architecture sidesteps this by asking a fundamentally easier question: "What entities are mentioned here?" (95%+ accuracy) instead of "What is the precise relationship between these entities?" (70% accuracy, noisy). Relationships are inferred from co-occurrence within chunks and resolved at query time by the consuming LLM reading the underlying evidence text.

**What this changes:** The extraction pipeline shrinks from five LLM-dependent stages to a three-pass pipeline where only two passes require LLM calls (and both are classification/tagging tasks, not generation tasks). The graph schema simplifies — edges carry chunk-type labels and evidence text rather than typed predicates. The graph becomes a retrieval index (like Atlas already is for code) rather than a knowledge base. Coreference resolution drops from critical-path risk to optional enhancement. Timeline compresses from 16–18 weeks to 10–12 weeks.

**What doesn't change:** The Rust core graph infrastructure, petgraph, PageRank, persistence strategy, Anchor & Expand context assembly, MCP server architecture, reflection loop concept, and incremental update model all carry forward from v3. Atlas's architecture remains the foundation.

---

## 1. Architectural Mapping: Atlas → Cortex

The following table maps every major Atlas subsystem to its Cortex equivalent. Components marked **REUSE** require no or minimal changes. Components marked **ADAPT** require structural modifications to existing code. Components marked **BUILD** are net-new.

**Note on effort ratings:** "Medium" means the existing logic transfers with moderate refactoring. "Medium-High" means the existing code structure is helpful but the type signatures change throughout, requiring careful propagation across call sites and tests.

| Atlas Component | File(s) | Cortex Equivalent | Action | Effort |
|---|---|---|---|---|
| Tree-sitter parser | `rust_core/src/parser.rs`, `queries/` | Three-pass tag-and-index pipeline | **BUILD** | Medium |
| File-level DiGraph | `rust_core/src/graph.rs` (RepoGraph) | Knowledge DiGraph (KnowledgeGraph) | **ADAPT** | Medium-High |
| FileNode / EdgeKind types | `rust_core/src/graph.rs` | EntityNode / CoOccurrenceEdge types | **ADAPT** | Medium |
| CPG overlay | `rust_core/src/cpg.rs` | *Removed — not needed for tag-and-index* | **REMOVE** | — |
| CFG builder | `rust_core/src/cfg.rs` | *Removed — not needed for tag-and-index* | **REMOVE** | — |
| Dataflow analysis | `rust_core/src/dataflow.rs` | *Removed — relationships inferred at query time* | **REMOVE** | — |
| Call graph builder | `rust_core/src/callgraph.rs` | *Removed — co-occurrence replaces explicit relation graph* | **REMOVE** | — |
| Symbol table / index | `rust_core/src/symbol_table.rs` | Tag index + disambiguation | **ADAPT** | Medium |
| Import resolver | `rust_core/src/import_resolver.rs` | *Removed — coreference is optional, handled in Python* | **REMOVE** | — |
| File watcher | `rust_core/src/watcher.rs` | Directory watcher (meetings/associates/work) | **REUSE** | Low |
| PyO3 bindings | `rust_core/src/lib.rs` | PyO3 bindings (new types) | **ADAPT** | Medium |
| Context manager | `python_shell/atlas/context.py` | Memory context assembler | **ADAPT** | Medium |
| Embedding manager | `python_shell/atlas/embeddings.py` | Entity + chunk embeddings | **REUSE** | Low |
| MCP server (12 tools) | `python_shell/atlas/mcp_server.py` | MCP server (new tool set) | **ADAPT** | Medium |
| Agent orchestrator | `python_shell/atlas/agent.py` | Memory agent + reflection loop | **ADAPT** | Medium |
| CLI | `python_shell/atlas/cli.py` | CLI (ingest, query, watch, reflect) | **ADAPT** | Low |
| Tool executor | `python_shell/atlas/tools.py` | Document processor | **ADAPT** | Low |
| LLM clients | `python_shell/atlas/llm.py` | LLM clients (classification + tagging) | **REUSE** | Low |
| PageRank scoring | `rust_core/src/graph.rs` | Entity importance scoring | **REUSE** | Low |
| Incremental updates | `rust_core/src/graph.rs` | Incremental knowledge updates | **ADAPT** | Low-Medium |
| Skeleton generation | `rust_core/src/parser.rs` | Entity summary generation | **BUILD** | Medium |
| `.atlas.toml` config | root | `.cortex.toml` config | **ADAPT** | Low |
| `.atlasignore` | root | `.cortexignore` | **REUSE** | Trivial |
| Tree-sitter queries | `rust_core/queries/` | LLM prompt templates | **BUILD** | Low-Medium |

**Why this is simpler than v3:** Four Atlas modules (CPG, CFG, dataflow, callgraph) are removed entirely — they existed to model code relationships that have no analogue in the tag-and-index architecture. The extraction pipeline drops from High effort to Medium because tagging is a classification task, not a generation task. The entity types and edge types are simpler, so the type-signature rewrite in the Rust core is less invasive.

---

## 2. Phased Implementation Plan

### Phase 0a: Fork and Rename (Week 1)

**Objective:** Clean fork, rename all references, verify the build. Atlas still works, just under Cortex names.

**Why this is separate from stripping:** Renaming is mechanically simple and produces a verifiable checkpoint — the full Atlas test suite should still pass under the new names. Stripping tree-sitter and stubbing modules (Phase 0b) intentionally breaks things. Keeping these apart means you can always roll back to "Atlas-under-Cortex-names" if Phase 0b goes wrong.

**Tasks:**

1. Fork `a-petty/Atlas` to your `cortex` repository.
2. Global rename: `atlas` → `cortex`, `Atlas` → `Cortex` across all files — Cargo.toml, pyproject.toml, CLI entry points, MCP server name, module names.
3. Rename directories: `python_shell/atlas/` → `python_shell/cortex/`.
4. Update `pyproject.toml` entry points: `atlas-mcp` → `cortex-mcp`, `atlas` CLI → `cortex`.
5. Rename config files: `.atlas.toml` → `.cortex.toml`, `.atlasignore` → `.cortexignore`.
6. Verify `maturin develop` builds cleanly, `pytest` passes, MCP server starts.

**Deliverable:** Cortex builds and passes all existing Atlas tests. Every binary, import path, and config file uses the Cortex name. Nothing has been removed yet.

**Exit criteria:** `maturin develop && pytest` passes. `cortex-mcp --project-root .` starts and responds to tool calls.

---

### Phase 0b: Strip and Stub (Week 1–2)

**Objective:** Remove tree-sitter dependencies and code-specific logic. Stub modules that will be rebuilt in later phases. Create directory scaffolding for the new Cortex modules.

**Tasks:**

1. Strip tree-sitter language grammars from `Cargo.toml` dependencies (Python, Rust, JS, TS, Go, Java grammar crates). Remove `tree-sitter` core as well — Cortex will not parse source code.
2. Remove code-specific modules entirely: `cpg.rs`, `cfg.rs`, `dataflow.rs`, `callgraph.rs`, `import_resolver.rs`. These have no analogue in the tag-and-index architecture — unlike v3, which adapted them, v4 removes them.
3. Gut `parser.rs`: remove all tree-sitter parsing logic. Leave a stub module so `lib.rs` compiles.
4. Remove tree-sitter query files from `rust_core/queries/`.
5. Create the new directory scaffolding:

```
cortex/
├── rust_core/
│   ├── src/
│   │   ├── knowledge_graph.rs    # New: replaces graph.rs for knowledge domain
│   │   ├── entity.rs             # New: entity/tag types
│   │   ├── chunk.rs              # New: chunk storage and metadata
│   │   ├── cooccurrence.rs       # New: co-occurrence edge types
│   │   ├── persistence.rs        # New: save/load/crash recovery
│   │   ├── graph.rs              # Retained: core petgraph DiGraph utilities
│   │   ├── watcher.rs            # Retained: file system monitoring
│   │   ├── symbol_table.rs       # Adapted → tag_index.rs
│   │   └── lib.rs                # Adapted: new PyO3 types
│   └── tests/
├── python_shell/
│   └── cortex/
│       ├── extraction/           # New: three-pass tag-and-index pipeline
│       │   ├── __init__.py
│       │   ├── structural_parser.py   # Pass 1: deterministic structural chunking
│       │   ├── classifier.py          # Pass 2: LLM chunk classification
│       │   ├── tagger.py              # Pass 3: LLM entity tagging
│       │   └── prompts/               # LLM prompt templates
│       │       ├── classify.txt
│       │       └── tag.txt
│       ├── ingestion/            # New: directory-aware document ingestion
│       │   ├── __init__.py
│       │   ├── pipeline.py       # Orchestrates the 3-pass pipeline
│       │   └── categorizer.py    # Maps directory → memory category
│       ├── reflection/           # New: background consolidation agent
│       │   ├── __init__.py
│       │   ├── consolidator.py   # Merge, prune, decay
│       │   └── synthesizer.py    # Generate higher-order insights
│       ├── context.py            # Adapted: Anchor & Expand for knowledge
│       ├── embeddings.py         # Retained: FastEmbed vector search
│       ├── mcp_server.py         # Adapted: new tool set
│       ├── agent.py              # Adapted: memory agent loop
│       ├── cli.py                # Adapted: new subcommands
│       ├── llm.py                # Retained: Ollama/MLX clients
│       └── config.py             # New: Cortex configuration management
├── memory/                       # New: default memory repository root
│   ├── meetings/
│   ├── associates/
│   └── work/
├── test_corpus/                  # New: synthetic test documents (see Phase 2D)
│   ├── meetings/
│   ├── associates/
│   └── work/
├── .cortex.toml
├── pyproject.toml
└── README.md
```

6. Verify `maturin develop` still builds (with stubs). Tests that depend on tree-sitter or removed modules will fail — that's expected. Remove or `#[ignore]` those tests explicitly so the remaining suite is green.

**Deliverable:** Cortex builds without tree-sitter. Code-specific modules are removed. Directory structure for all new modules exists. Test suite is green (with code-specific tests removed or ignored).

**Exit criteria:** `maturin develop && cargo test` passes (ignoring removed tests). No tree-sitter crates in `Cargo.lock`.

---

### Phase 1: Core Graph Schema Redesign + Persistence (Weeks 2–4)

**Objective:** Replace Atlas's code-centric graph types with knowledge-centric types in the Rust core. Define and implement the persistence strategy.

**Why 2–3 weeks:** This phase requires changing the type signature of the central data structure that everything else touches. However, the type system is simpler than v3's because co-occurrence edges are structurally uniform — unlike v3's 18+ typed relation variants, every edge in v4 has the same shape. Additionally, persistence must be settled here — Cortex can't cheaply rebuild from source because re-extraction requires LLM calls.

#### 1A. Define new node and edge types

Atlas currently defines `FileNode` and `EdgeKind` (Import, SymbolUsage) in `graph.rs`. Replace with:

```rust
// entity.rs — New file

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Project,
    Technology,
    Organization,
    Location,
    Decision,
    Concept,
    Document,      // The source document itself
    Custom(String), // User-defined or LLM-suggested types
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub id: String,                          // Stable unique ID (UUID)
    pub canonical_name: String,              // Disambiguated primary name
    pub aliases: Vec<String>,                // "J. Doe", "John", "the lead engineer"
    pub entity_type: EntityType,
    pub source_chunks: Vec<String>,          // Which chunks contributed to this tag
    pub source_documents: Vec<String>,       // Which files contributed to this tag
    pub created_at: i64,                     // Unix timestamp
    pub updated_at: i64,
    pub access_count: u32,                   // For importance/decay scoring
    pub embedding: Option<Vec<f32>>,         // Cached embedding vector (see 1C)
}
```

```rust
// chunk.rs — New file

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkType {
    Decision,
    Discussion,
    ActionItem,
    StatusUpdate,
    Preference,
    Background,     // Useful context but not a specific event type
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryCategory {
    Episodic,    // meetings/
    Semantic,    // associates/
    Procedural,  // work/
}

/// A chunk is a segment of a source document with metadata from all three passes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,                      // Stable unique ID
    pub source_document: String,         // File path
    pub section_name: Option<String>,    // From structural parse (e.g., "## Q3 Review")
    pub speaker: Option<String>,         // From structural parse (speaker turns)
    pub timestamp: Option<i64>,          // From structural parse (date markers)
    pub chunk_type: ChunkType,           // From Pass 2 (LLM classification)
    pub memory_category: MemoryCategory, // From directory location
    pub text: String,                    // The original chunk text
    pub tags: Vec<String>,              // Entity IDs tagged in Pass 3
    pub created_at: i64,
}
```

```rust
// cooccurrence.rs — New file

/// An edge in the co-occurrence graph.
/// Created when two entities are tagged within the same chunk.
/// The edge is labeled with the chunk type and carries the evidence text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoOccurrenceEdge {
    pub chunk_id: String,               // Which chunk produced this co-occurrence
    pub chunk_type: ChunkType,          // Decision, Discussion, ActionItem, etc.
    pub memory_category: MemoryCategory,
    pub timestamp: Option<i64>,         // When this co-occurrence was observed
    pub source_document: String,        // Which file
    pub weight: f32,                    // Importance weight (for PageRank)
}
```

**Key design difference from v3:** There are no typed predicates (Attended, Decided, ReportsTo, etc.). The graph says "John Doe and Dashboard Redesign co-occur in a Decision chunk from 2024-10-14." It does *not* say John "led" the project — that inference lives in the chunk text and is resolved at query time by the consuming LLM. This is the fundamental tradeoff: edges are semantically thin, but the system is far more reliable because tagging is easier than relation extraction.

#### 1B. Adapt the core graph structure

Atlas's `RepoGraph` in `graph.rs` wraps a `petgraph::DiGraph<FileNode, EdgeKind>`. Create `KnowledgeGraph` wrapping `DiGraph<EntityNode, CoOccurrenceEdge>`:

- **Retain:** petgraph DiGraph, parallel processing with rayon, PageRank implementation (edge weights are uniform by default, but chunk-type-specific weights can be configured — e.g., Decision edges carry 2x weight), content hash-based change detection.
- **Retain:** The side-maps pattern (`HashMap<PathBuf, NodeIndex>` becomes `HashMap<String, NodeIndex>` keyed by entity ID).
- **Remove:** All tree-sitter-specific logic, import resolution, symbol harvesting, CPG/CFG/dataflow/callgraph references.
- **Add:** Temporal filtering on graph traversal (only follow edges whose timestamp falls within a query-specified window), entity merge operations (combine two nodes into one, redirect all edges), decay scoring (access_count × recency weighting), chunk storage (a separate `HashMap<String, Chunk>` for retrieving evidence text at query time).

**Why chunks are stored alongside the graph:** The graph is a retrieval index. When the context assembler finds relevant entities via graph traversal, it needs to pull the underlying chunk text to include in the LLM prompt. Without chunk storage, the system would have to re-read and re-parse source documents at query time. Storing chunks in memory alongside the graph makes retrieval O(1).

#### 1C. Adapt the tag index and define the embedding strategy

Atlas's `SymbolIndex` in `symbol_table.rs` provides `name → files` and `file → symbols` mappings. Adapt to `TagIndex`:

- `canonical_name → NodeIndex` (primary lookup)
- `alias → NodeIndex` (fuzzy lookup)
- `source_document → Vec<NodeIndex>` (which entities came from which file)
- `entity_type → Vec<NodeIndex>` (type-based filtering)
- Embedding-based nearest-neighbor lookup for disambiguation

**Entity embedding strategy (carried forward from v3):**

The `embedding` field on `EntityNode` stores a vector computed from a contextual descriptor string — not just the entity name. Entity names alone (e.g., "John") produce poor embeddings because they lack distinguishing context. The embedding input string is constructed as:

```
"{canonical_name} ({entity_type}): {top 3 chunk texts where this entity is tagged}"
```

For example:
```
"John Doe (Person): Lead engineer on the Dashboard Redesign project. Prefers Python with strict linting. Relocated from Seattle to Tokyo in November 2024."
```

This descriptor is recomputed whenever an entity's tags, co-occurrences, or evidence chunks change significantly (triggered during ingestion and reflection). The `embeddings.py` module (reused from Atlas's FastEmbed integration) generates the vector from this string.

**Why this matters for disambiguation:** When Pass 3 (tagging) extracts a "John" mention, comparing embeddings of contextual descriptors is far more reliable than comparing embeddings of bare names. The descriptors encode what you *know* about each entity, which is exactly the information needed to judge whether two mentions refer to the same real-world referent.

**Embedding refresh policy:** Recompute the descriptor and embedding when: (a) a new chunk tags the entity, (b) the reflection loop modifies the entity's aliases or merges entities, or (c) manually triggered. Do *not* recompute on every access_count increment.

#### 1D. Persistence strategy

The knowledge graph is the core data asset — if Cortex crashes or restarts, the graph must survive. Atlas didn't need this because it could rebuild from source code. Cortex can't cheaply rebuild because re-extracting from documents requires LLM calls.

**Storage format:** Serde serialization of the full `KnowledgeGraph` state (including chunk storage) to a binary file (MessagePack via `rmp-serde` for speed and compactness, with a JSON fallback for debugging).

**File layout:**
```
.cortex/
├── graph.msgpack          # Primary graph state (entities + edges + chunks)
├── graph.msgpack.bak      # Previous snapshot (crash recovery)
├── entity_embeddings.bin  # Embedding vectors (separate for size)
└── review_queue.jsonl     # Human review items (see Phase 2C)
```

**Save strategy:**
- **Auto-save after ingestion:** Every successful `cortex ingest` triggers a full graph snapshot.
- **Periodic save during watch mode:** Every 5 minutes (configurable) during `cortex watch`.
- **Save before shutdown:** Graceful shutdown always saves.
- **Crash recovery:** On startup, load `graph.msgpack`. If corrupted, fall back to `graph.msgpack.bak`. If both are gone, the graph starts empty (documents are still on disk and can be re-ingested).

**Why not SQLite/a database?** The graph is an in-memory data structure (petgraph DiGraph). Serializing it is simpler than maintaining a database schema, and in-memory operations are faster. If the graph eventually grows beyond memory (unlikely for the target use case of personal/team knowledge), this decision can be revisited.

**Implementation:**
```rust
// persistence.rs — New file

pub struct GraphStore {
    path: PathBuf,
}

impl GraphStore {
    pub fn save(&self, graph: &KnowledgeGraph) -> Result<(), PersistenceError>;
    pub fn load(&self) -> Result<KnowledgeGraph, PersistenceError>;
    pub fn save_with_backup(&self, graph: &KnowledgeGraph) -> Result<(), PersistenceError>;
    pub fn recover(&self) -> Result<KnowledgeGraph, PersistenceError>;
    pub fn export_json(&self, graph: &KnowledgeGraph) -> Result<(), PersistenceError>; // debugging
}
```

Expose via PyO3: `save()`, `load()`, `export_json()`.

#### 1E. Update PyO3 bindings

In `lib.rs`, create `PyKnowledgeGraph` wrapping `KnowledgeGraph`, exposing methods:

- `add_entity(entity_json: &str) -> String` (returns node ID)
- `add_cooccurrence(entity_a_id: &str, entity_b_id: &str, edge_json: &str)` (adds co-occurrence edge)
- `store_chunk(chunk_json: &str)` (stores a chunk in the chunk map)
- `get_chunk(chunk_id: &str) -> Option<String>` (retrieves chunk text)
- `merge_entities(id_a: &str, id_b: &str) -> String` (returns surviving ID)
- `query_neighborhood(entity_id: &str, hops: usize, time_filter: Option<(i64, i64)>) -> String` (JSON subgraph)
- `get_entity(id: &str) -> Option<String>` (JSON)
- `get_cooccurrences(entity_id: &str) -> Vec<String>` (all co-occurring entities with evidence)
- `search_entities(query: &str, limit: usize) -> Vec<String>`
- `compute_pagerank() -> Vec<(String, f64)>`
- `get_temporal_chain(entity_id: &str) -> Vec<String>` (chunks involving entity, ordered by timestamp)
- `get_chunks_for_entities(entity_ids: Vec<String>) -> Vec<String>` (retrieve evidence chunks)
- `decay_scores(half_life_days: f64)` (reduce access_count by time-based decay)
- `export_subgraph(entity_ids: Vec<String>) -> String` (JSON for LLM context injection)
- `save(path: &str)` / `load(path: &str)` — persistence

**Deliverable:** `maturin develop` builds, Python can create a KnowledgeGraph, add entities and co-occurrence edges, store and retrieve chunks, traverse neighborhoods, run PageRank, save to disk, and load from disk. All via the Rust core.

**Exit criteria:** Rust tests cover add/query/merge/save/load round-trip. Python tests confirm PyO3 bindings work end-to-end.

---

### Phase 2: Three-Pass Tag-and-Index Pipeline (Weeks 4–7)

**Objective:** Build the Python-side extraction pipeline that replaces tree-sitter parsing.

This is the core innovation of v4. Unlike v3's five-stage pipeline (which required coreference resolution, entity extraction, triple extraction, and disambiguation — all generation tasks), the three-pass pipeline asks fundamentally easier questions of the LLM. Pass 1 requires no LLM at all. Passes 2 and 3 are classification and tagging tasks — the LLM identifies and labels, it doesn't generate structured triples.

**Why 3 weeks instead of v3's 6:** Coreference resolution (v3's riskiest sub-problem) is removed from the critical path. Triple extraction (v3's highest-effort stage) is eliminated entirely. The two LLM passes are classification tasks with simpler prompts, more predictable outputs, and higher baseline accuracy. The main iteration time is on Pass 3 (tagging) prompt quality and entity disambiguation.

#### 2A. Pass 1 — Structural Parse (deterministic, no LLM)

`python_shell/cortex/extraction/structural_parser.py`

Split each document using its own structure. This pass is entirely deterministic — no LLM calls, no probabilistic output, fully repeatable.

**Chunking strategy by document type:**

- **Markdown documents:** Split on headers (`##`, `###`), horizontal rules, and double-newline paragraph breaks. Each section becomes a chunk. Preserve the header text as the chunk's `section_name` metadata.
- **Meeting transcripts with speaker turns:** Split on speaker labels (e.g., `**John:**`, `Speaker 1:`). Each speaker turn becomes a chunk. Attach the speaker name as metadata.
- **Date-structured documents:** Split on date markers (ISO dates, "October 14, 2024", "Monday", etc.). Attach parsed timestamps as metadata.
- **Plain text:** Split on paragraph breaks (double newline). Fall back to token-based splitting (~800 tokens per chunk, with 100-token overlap) if paragraphs are too long.

**What Pass 1 produces for each chunk:**
```python
{
    "id": "chunk_uuid",
    "source_document": "memory/meetings/q3-design-review.md",
    "section_name": "## Chart Design Discussion",   # from markdown headers
    "speaker": "John Doe",                           # from speaker turns
    "timestamp": 1728864000,                         # from date markers
    "memory_category": "episodic",                   # from directory location
    "text": "The original chunk text...",
}
```

**Why structural parsing first:** By splitting on the document's own structure rather than arbitrary token windows, chunks are semantically coherent. A decision expressed across two paragraphs stays in one chunk. A speaker's complete statement isn't split mid-sentence. This makes Passes 2 and 3 more accurate because the LLM sees complete thoughts, not arbitrary text fragments.

#### 2B. Pass 2 — Classify (LLM, easy task)

`python_shell/cortex/extraction/classifier.py`

For each chunk from Pass 1, ask the LLM: "What kind of information is this?"

**LLM prompt:**
```
Classify the following text chunk into exactly one category.
Categories: decision, discussion, action_item, status_update, preference, background, filler

If the chunk contains no meaningful information (greetings, pleasantries, 
scheduling logistics), classify as "filler".

Respond with ONLY the category name, nothing else.

Text:
{chunk_text}
```

**Why this is reliable:** Classification into a small fixed set of categories is one of the easiest tasks for any LLM. Even small models (3b-class) handle this well because the output space is constrained — there's no JSON to malform, no entities to hallucinate, just a single word from a known list.

**Filler detection:** Chunks classified as "filler" are discarded before Pass 3. This prevents the graph from filling with noise. Typical discard rate: 15–30% of chunks in meeting transcripts (greetings, "can you hear me?", scheduling chatter).

**Validation:** If the LLM returns a category not in the fixed set, default to "background" and log a warning. No retry needed — this failure mode is rare and the default is safe.

#### 2C. Pass 3 — Tag (LLM, moderate task)

`python_shell/cortex/extraction/tagger.py`

For each non-filler chunk, ask the LLM: "What entities are mentioned here?"

**LLM prompt:**
```
List every person, project, technology, organization, location, decision, 
and concept mentioned in the following text.

For each entity, provide:
- name: the most complete form of the name used in the text
- type: person, project, technology, organization, location, decision, or concept

Return JSON array: [{"name": "...", "type": "..."}]
Return ONLY the JSON array, no other text.

Text:
{chunk_text}
```

**Why this is more reliable than triple extraction:** The LLM is identifying what's mentioned, not interpreting how things relate. "John Doe" is mentioned — that's factual and verifiable against the source text. "John Doe *led* the Dashboard Redesign" is an interpretation that may or may not be accurate. Tagging precision is typically 90–95%, vs. 65–75% for triple extraction.

**Post-extraction validation:** For each tagged entity, verify that the name appears in the source chunk text (exact or fuzzy match). Remove entities that don't appear — these are hallucinations. This simple check catches the most common LLM extraction failure mode.

**Entity disambiguation (inline, not a separate stage):**

When Pass 3 produces a tag like "John", check the existing tag index:
1. Exact match on canonical_name or aliases → link to existing entity
2. Embedding cosine similarity > 0.85 on contextual descriptors → link to existing entity
3. No match → create new entity

For borderline cases (similarity between 0.7 and 0.85), add to the human review queue rather than guessing. The review queue follows the same lifecycle policy as v3: items auto-expire after 30 days, auto-promote if corroborated by a different source, and queue size is visible via `cortex_status`.

**Graph construction from tags:**

After Pass 3 processes a chunk:
1. For each tagged entity, ensure it exists in the graph (create or link to existing)
2. For every *pair* of entities tagged in the same chunk, add a `CoOccurrenceEdge` between them
3. Store the chunk in the chunk map

For a chunk with entities [A, B, C], this creates edges: A↔B, A↔C, B↔C — all labeled with the chunk type, timestamp, and chunk ID. The chunk text is stored separately and retrieved at query time.

**LLM output error handling:**

| Error | Strategy | Fallback |
|---|---|---|
| **Malformed JSON** | Retry once with repair prompt | Skip chunk, log raw output, mark document partial |
| **Hallucinated entities** (name not in source text) | Remove from tag list | Proceed with remaining valid tags |
| **LLM timeout** | Retry once | Skip chunk, mark document partial |
| **Rate limit / model unavailable** | Exponential backoff: 1s, 5s, 30s, abort | Queue document for retry next cycle |
| **Empty tag list** (LLM returns `[]`) | Accept — some chunks genuinely contain no taggable entities | Log for review if the chunk seemed entity-rich |

**Circuit breaker:** If more than 50% of chunks in a single document fail tagging, abort that document. Something is systematically wrong.

**Partial extraction policy:** A document can be "partially extracted." The graph contains whatever was successfully tagged. The document is marked with `extraction_status`: `complete`, `partial`, or `failed`.

#### 2D. Test corpus

Create a controlled test dataset in `test_corpus/` at the start of Phase 2:
- 10 meeting transcripts (with speaker turns, dates, decisions)
- 5 associate profiles
- 5 completed project summaries
- Deliberately include: temporal state changes (person moves cities), ambiguous entity references ("the project" referring to different projects), at least one multi-hop reasoning chain (A connected to B connected to C)

**Ground truth annotations:** For each test document, create a `.expected.json` sidecar file containing the entities a human would tag and the co-occurrences they'd expect. Use these for development validation.

#### 2E. Model routing

**Start with the reasoning model (e.g., `llama3.3:70b`) for both passes.** Once prompts are stable, test downgrading:

- **Pass 2 (classification)** is the strongest candidate for a small model — it's a single-word classification task. Test with 3b-class models early.
- **Pass 3 (tagging)** requires more judgment. Test downgrading but expect that larger models will produce meaningfully better entity lists, especially for implicit mentions.

**Decision rule:** Downgrade only if tagging precision drops by less than 10% and JSON malformed rate stays below 15%.

**Deliverable:** Drop a markdown meeting transcript into `memory/meetings/`, run `cortex ingest`, and see entities and co-occurrence edges populated in the knowledge graph, with chunk text stored for retrieval. LLM errors handled gracefully.

**Exit criteria:** All 10 test meeting transcripts ingest without crashing. Entity tagging precision >85% against ground truth (note: higher bar than v3's 70% for triple extraction, because tagging is inherently more accurate). No unhandled LLM errors.

---

### Phase 3: Context Assembly Adaptation (Weeks 7–8)

**Objective:** Adapt Atlas's Anchor & Expand context assembly for knowledge graph retrieval.

Atlas's `ContextManager` in `context.py` implements three-tier token budgeting. The architecture transfers almost directly — this is where Atlas's heritage provides the strongest differentiation over existing NL knowledge graph tools.

#### 3A. Tier mapping

| Atlas Tier | Budget | Cortex Tier | Budget |
|---|---|---|---|
| Tier 1: Repository Map (PageRank-ranked file overview) | ~8% | Tier 1: Knowledge Map (PageRank-ranked entity overview with types and co-occurrence counts) | ~10% |
| Tier 2: Full File Content (anchor files + BFS neighborhood) | 40–75% | Tier 2: Evidence Chunks (chunks containing anchor entities + co-occurring entities' chunks) | 40–70% |
| Tier 3: Architectural Skeletons (signatures only) | Remaining | Tier 3: Entity Summaries (compressed profiles of peripherally relevant entities) | Remaining |

#### 3B. Anchor & Expand adaptation

**Anchor phase** (find relevant entities):

1. Embed the user query using FastEmbed (Atlas's `embeddings.py` — reuse as-is)
2. Cosine similarity search against entity contextual descriptor embeddings (see Phase 1C) to find top-k most relevant entities
3. Also search against chunk text embeddings for full-text matches

**Expand phase** (gather context neighborhood):

1. Multi-hop BFS from anchor entities through the co-occurrence graph
2. **Temporal filtering:** Only traverse edges whose timestamp falls within a query-specified window, unless the query explicitly asks about history
3. **Edge-type-aware traversal** (mirrors Atlas's existing pattern):
   - Decision/ActionItem co-occurrences: full hop depth (these are the strongest signal)
   - Discussion co-occurrences: 1 hop (to prevent explosion through casual mentions)
   - Background co-occurrences: 1 hop
4. **Distance decay weighting:** hop n → weight 1/2^(n-1) (reuse Atlas's existing formula)

**Chunk retrieval** (the key difference from v3):

Once the Expand phase identifies relevant entities, retrieve the underlying chunks:
1. For each relevant entity pair, get the `chunk_id` from their `CoOccurrenceEdge`
2. Fetch the chunk text from the chunk store
3. Deduplicate chunks (the same chunk may be referenced by multiple edges)
4. Rank chunks by relevance (distance from anchor entities × chunk type weight)
5. Fill Tier 2 budget with ranked chunk texts

This is where the "graph as retrieval index" architecture pays off — the graph finds the right chunks, the chunks contain the actual evidence, and the consuming LLM reads the evidence to answer the question. The graph never claims to know *what* the relationship is; it just points the LLM at the right text.

**Entity summary generation** (Tier 3, replaces skeleton generation):

- For entities that made it into the expanded set but whose chunks didn't fit in Tier 2's budget
- Generate a one-line summary: "{name} ({type}): co-occurs with {top 3 co-occurring entities} in {N} chunks"
- Cache these summaries (Atlas caches skeletons in an LRU with 500 entries — reuse)

#### 3C. Context serialization

Cortex serializes the retrieved knowledge into a structured format the LLM can reason over:

```markdown
## Knowledge Context

### Key Entities
- **John Doe** (Person): Appears in 14 chunks across 8 documents.
  Co-occurs most frequently with: Dashboard Redesign, Jane Smith, React.
  Known since: 2024-03-15. Last referenced: 2025-02-28.

### Evidence (most relevant chunks, ordered by relevance)

**[Decision] Q3 Design Review — 2024-10-14**
> John Doe: "We need to abandon the dual-axis chart. The user testing showed
> people couldn't read it. Let's go with separate bar charts instead."
> Jane Smith: "Agreed. John, can you have the new component ready by Oct 21?"

**[ActionItem] Q3 Design Review — 2024-10-14**
> Action: John Doe to implement new bar chart component by October 21.
> Depends on: Data Pipeline v2 providing the split dataset endpoint.

**[StatusUpdate] Weekly Standup — 2024-11-18**
> John Doe: Reporting from Tokyo office this week. Bar chart component shipped
> last Friday. Starting on the dashboard layout refactor.

### Peripheral Entities (summaries only)
- **Data Pipeline v2** (Project): co-occurs with Dashboard Redesign, John Doe in 4 chunks
- **React** (Technology): co-occurs with Dashboard Redesign, John Doe in 6 chunks
```

**Why this is better than v3's approach:** v3 serialized *interpreted* relationships ("John Doe → LED → Dashboard Redesign"). If the extraction got "LED" wrong, the consuming LLM would confidently repeat the error. v4 serializes *evidence chunks* — the actual text where entities co-occur. The consuming LLM reads the evidence and draws its own conclusions about what the relationship is. This means the answer quality is bounded by the LLM's reading comprehension (which is excellent) rather than by the extraction pipeline's relation-typing accuracy (which is mediocre).

**Deliverable:** `cortex query "What did John decide about the chart design?"` returns a correctly assembled context window with the right entities and evidence chunks, within the model's token budget.

---

### Phase 4: MCP Server Tool Redesign (Weeks 8–9)

**Objective:** Replace Atlas's 12 code-centric MCP tools with knowledge-centric equivalents.

| Atlas Tool | Cortex Replacement | Description |
|---|---|---|
| `atlas_status` | `cortex_status` | Graph stats: entity count, edge count, chunk count, memory categories, last ingestion time, persistence status, review queue size |
| `get_repository_map` | `get_knowledge_map` | PageRank-ranked entity overview grouped by memory category |
| `get_dependencies` | `get_cooccurrences` | All co-occurring entities for a given entity, with chunk types and timestamps |
| `get_dependents` | `get_entity_mentions` | All chunks where an entity is tagged |
| `get_top_ranked_files` | `get_key_entities` | Most central entities by PageRank score, filterable by type/category |
| `find_relevant_files` | `find_relevant_entities` | Semantic search via contextual descriptor embeddings |
| `assemble_context` | `assemble_memory` | Core intelligence: Anchor & Expand with three-tier budgeting and chunk retrieval |
| `get_file_symbols` | `get_entity_profile` | Full entity profile with all co-occurrences, tagged chunks, and aliases |
| `get_callees` | `get_timeline` | Chunks involving an entity ordered chronologically |
| `get_callers` | `get_evidence` | Retrieve specific chunk texts for a pair of co-occurring entities |
| `get_file_skeleton` | `get_entity_summary` | Compressed entity profile (one-liner for context stuffing) |
| `atlas_refresh` | `cortex_reingest` | Re-process all source documents and rebuild the graph from scratch |
| — | `cortex_ingest` | **NEW:** Trigger ingestion of a specific document or directory |
| — | `cortex_reflect` | **NEW:** Trigger the reflection/consolidation loop manually |
| — | `cortex_forget` | **NEW:** Mark an entity as decayed/archived |
| — | `cortex_review` | **NEW:** Present human review queue items for confirmation/rejection |

**Deliverable:** MCP server starts, Claude Code/Desktop can call all tools, and the full memory lifecycle (ingest → query → reflect → forget) is accessible via MCP.

---

### Phase 5: Reflection and Consolidation (Weeks 9–10)

**Objective:** Build the background agent that maintains graph health over time.

This is the component that distinguishes a "knowledge index" from a "knowledge dump." Without it, the graph bloats with noise, duplicate entities accumulate, and stale chunks persist. Note that v4's reflection loop is simpler than v3's because there are no typed relation edges to resolve contradictions on — the "contradiction" problem is reframed as "stale evidence" and handled by temporal decay.

#### 5A. Consolidation operations

`python_shell/cortex/reflection/consolidator.py`:

1. **Entity merging:** Periodically scan for entity pairs with high alias overlap or embedding similarity > 0.9 (using contextual descriptor embeddings). Present candidates to the reasoning LLM for confirmation. Merge confirmed duplicates using the Rust core's `merge_entities()`.

2. **Stale evidence detection:** For entities with chunks spanning a long time range, identify chunks where the information has been superseded by newer chunks. Example: a chunk saying "John is in Seattle" from June, followed by "John is in Tokyo" from November. The system doesn't delete the old chunk (it's still true that John *was* in Seattle), but marks it as historical and deprioritizes it in context assembly. The consuming LLM can still see the history if the query asks for it.

3. **Decay scoring:** Implement an exponential decay on `access_count`:
   ```
   effective_score = access_count × e^(-λ × days_since_last_access)
   ```
   where λ is configurable in `.cortex.toml`. Entities that fall below a threshold are flagged for archival.

4. **Orphan cleanup:** Remove entities that have no co-occurrence edges and no recent chunks. These are extraction artifacts that were never corroborated.

#### 5B. Synthesis operations

`python_shell/cortex/reflection/synthesizer.py`:

1. **Cross-chunk inference:** After N new documents are ingested, run a synthesis pass:
   "Given these {N} chunks involving {entity}, what patterns, recurring concerns, or trajectory do you see?"
   Store results as new "synthesis" chunks linked to the entity, tagged with chunk_type = Background.

2. **Entity profile enrichment:** After each ingestion cycle, check if new chunks reveal additional context for existing entities. Update aliases, refine entity types if needed. Trigger embedding refresh for affected entities.

3. **Co-occurrence strength update:** Recalculate edge weights based on frequency — entity pairs that co-occur across many chunks and many documents get stronger edges, which influences PageRank and context assembly priority.

#### 5C. Scheduling

The reflection loop should run:
- **After every ingestion** (lightweight: disambiguation check, orphan scan)
- **On a configurable schedule** (heavy: full consolidation, synthesis, decay). Default: daily.
- **On-demand** via the `cortex_reflect` MCP tool or CLI subcommand.

Atlas's `agent.py` already has a watch loop pattern that can be extended with a periodic timer.

**Deliverable:** The graph self-maintains. Duplicate entities merge automatically. Stale evidence is deprioritized. Unused entities decay. The system can answer "What patterns do you see across John's last 10 meetings?" with synthesized insights drawn from evidence chunks.

---

### Phase 6: Incremental Update Handling (Weeks 10–11)

**Objective:** Handle document changes efficiently without full re-ingestion.

**Two-tier model (carried forward from v3):**

| Tier | Trigger | Action |
|---|---|---|
| **DocumentChanged** | A watched file's content hash changes | Re-run the full three-pass pipeline for that document. Diff the new tags against the existing chunks sourced from this document. Add new entities/co-occurrences, remove chunks that no longer exist, update changed chunks. Trigger a graph save. |
| **DocumentDeleted** | A watched file is removed | Remove all chunks sourced from the deleted file. Remove co-occurrence edges whose only supporting chunk was from this file. Remove entities whose only source_document was this file (after checking for co-occurrences from other documents). Trigger a graph save. |

**Why this is sufficient:** The three-pass pipeline is cheaper than v3's five-stage pipeline (no coreference resolution, no triple extraction), so re-processing a single document is faster. At the target scale (personal/team knowledge, hundreds to low thousands of documents), re-processing one document takes a few seconds of LLM time.

**Deliverable:** Editing a meeting transcript triggers re-extraction for that document only. Deleting a file cleanly removes its contributions from the graph.

---

### Phase 7: Evaluation Framework (Weeks 11–12)

**Objective:** Build the test harness that proves Cortex works better than vector-only retrieval.

#### 7A. Expand test corpus

Expand the Phase 2D corpus to full benchmark scale:
- 50 meeting transcripts spanning 6 months, with 20 recurring participants
- 20 associate profiles with evolving preferences
- 15 completed project summaries with cross-references

All new documents get `.expected.json` ground truth annotations (entities and expected co-occurrences).

#### 7B. Evaluation metrics

| Metric | Method |
|---|---|
| **Entity tagging accuracy** | Ground-truth entity mapping. Measure precision/recall of entity identification per chunk. |
| **Co-occurrence accuracy** | For 10 documents, verify that expected entity pairs are connected in the graph. |
| **Retrieval precision @k** | For 50 test queries, measure whether the top-k retrieved chunks contain the answer. |
| **Temporal reasoning accuracy** | 20 questions requiring state-at-time-T reasoning. Compare Cortex vs vector baseline. |
| **Multi-hop reasoning accuracy** | 20 questions requiring 2+ hops (e.g., "Who else works on the project that John presented at the Q3 review?"). |
| **Context efficiency** | Tokens consumed per query vs answer quality. Compare to naive RAG with same token budget. |
| **LLM error resilience** | Run the pipeline with intentionally degraded LLM responses. Measure graceful degradation vs crash. |

#### 7C. Baseline comparison

Implement a simple RAG baseline using the same document corpus with chunking + FastEmbed + cosine similarity retrieval. Run the same test queries against both systems. This is the comparison that makes the work publishable and defensible.

**Deliverable:** A benchmark suite that runs with `cortex eval` and produces a comparison report.

---

## 3. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Entity tagging quality too low for reliable graph construction | Medium | High | Tagging is inherently more reliable than triple extraction (95% vs 70% baseline). Post-extraction validation catches hallucinations. Phase 2D test corpus validates quality before proceeding. |
| R2 | Entity disambiguation errors compound over time | High | High | Conservative merging (only merge at >0.9 confidence). Maintain audit trail. Build an "undo merge" operation. Human review queue for borderline cases. |
| R3 | Graph bloat from noisy tagging | Medium | Medium | Aggressive decay scoring. Filler chunk detection in Pass 2 removes 15–30% of chunks before tagging. Minimum co-occurrence threshold for weak entities. |
| R4 | Co-occurrence edges too semantically thin for useful retrieval | Medium | Medium | This is the core tradeoff of the architecture. Mitigated by including chunk text in context assembly — the consuming LLM reads the evidence and infers relationships. If co-occurrence proves insufficient, typed edges can be added incrementally for high-value chunk types (Decisions, ActionItems) without redesigning the system. |
| R5 | Rust refactoring scope larger than expected | Medium | Medium | Keep Atlas's graph.rs as-is for the first phase. Build KnowledgeGraph as a separate struct that wraps the same petgraph primitives. Only refactor into graph.rs once the schema is stable. |
| R6 | MCP protocol changes break integration | Low | Medium | Pin MCP SDK versions. Atlas already handles this. |
| R7 | Persistence data loss or corruption | Medium | High | Backup file before every save. JSON export as emergency fallback. Source documents are always on disk — worst case, re-ingest from scratch. |
| R8 | LLM output unreliability (malformed JSON in Pass 3) | Medium | Medium | Pass 2 output is a single word (no JSON). Pass 3 JSON is a flat array (simpler than v3's nested triple schema). Retry with repair prompt. Circuit breaker for systematic failures. |
| R9 | Consuming LLM can't infer relationships from evidence chunks | Low | High | This depends on the quality of the context assembly, not the graph. Test with representative queries during Phase 3 development. If a specific relationship type is consistently missed, add a typed co-occurrence variant for that type only. |
| R10 | Existing systems (LightRAG, Cognee) ship similar features during development | Medium | Medium | The differentiation is the context assembly layer (Anchor & Expand with three-tier budgeting), not the graph construction. Even if graph construction converges, the retrieval quality from Atlas's heritage remains unique. |
| R11 | Chunk storage grows too large for in-memory persistence | Low | Medium | At target scale (1000 documents × 20 chunks × 500 bytes avg), total chunk storage is ~10MB. Well within memory. Revisit only if scaling to enterprise corpus sizes. |

---

## 4. Dependency Changes

### Rust (Cargo.toml)

**Remove:**
- `tree-sitter` and all language grammar crates
- Related test fixtures

**Add:**
- `uuid` — for stable entity and chunk IDs
- `chrono` — for timestamp handling
- `serde_json` — likely already present, ensure it's a direct dependency
- `rmp-serde` — MessagePack serialization for graph persistence

**Retain:**
- `petgraph` — graph implementation (core dependency)
- `pyo3` — Python bindings
- `rayon` — parallel processing
- `notify` + `notify-debouncer-full` — file watching
- `ignore` — gitignore-aware file walking
- `parking_lot` — sync primitives
- `crossbeam-channel` — thread messaging
- `lru` — caching
- `thiserror` — error types
- `serde` + `serde_derive` — serialization framework

### Python (pyproject.toml)

**Remove:**
- Nothing (Atlas's Python deps are all general-purpose)

**Add:**
- `pyyaml` — for prompt template management
- `anthropic` or keep `ollama` — for LLM extraction calls
- `jsonschema` — for validating LLM extraction output (Pass 3 only)

**Retain:**
- `tiktoken` — token counting
- `fastembed` — vector embeddings
- `rich` — terminal display
- `pydantic` — config validation
- `ollama` — local LLM client
- `numpy` — vector operations
- `mcp[cli]` — MCP server framework

---

## 5. Configuration Schema

```toml
# .cortex.toml

[project]
name = "My Project Memory"
memory_root = "./memory"        # Root directory for the three memory categories

[directories]
meetings = "meetings"           # Relative to memory_root
associates = "associates"
work = "work"

[models]
classification_provider = "ollama"  # "ollama", "mlx", "anthropic"
classification_model = "llama3.2:3b"  # Pass 2 is simple enough for small models
tagging_provider = "ollama"
tagging_model = "llama3.3:70b"       # Start with reasoning model; downgrade after testing
reasoning_provider = "ollama"
reasoning_model = "llama3.3:70b"     # For reflection/synthesis

[extraction]
min_chunk_tokens = 50              # Skip chunks shorter than this
max_chunk_tokens = 1200            # Re-split chunks longer than this
disambiguation_threshold = 0.85    # Embedding similarity threshold for entity matching
max_retries = 1                    # LLM call retries on failure (reduced from v3's 2)
circuit_breaker_pct = 50           # Abort document if >N% of chunks fail

[persistence]
format = "msgpack"              # "msgpack" or "json"
auto_save_interval_minutes = 5  # During watch mode
backup_count = 1                # Number of .bak files to keep

[reflection]
enabled = true
schedule = "daily"              # "after_ingest", "hourly", "daily", "manual"
decay_half_life_days = 90       # How quickly unused entities decay
merge_threshold = 0.90          # Confidence required for automatic entity merge
max_synthesis_sources = 10      # Max chunks per synthesis pass

[review]
review_queue_ttl_days = 30      # Auto-expire unreviewed items after this many days
max_queue_size_warning = 100    # Warn when queue exceeds this size

[context]
default_budget = 32000          # Default token budget for context assembly
map_budget_pct = 10             # Percent of budget for Tier 1 (knowledge map)
full_budget_pct = 60            # Percent of remaining for Tier 2 (evidence chunks)
max_hops = 3                    # Maximum graph traversal depth
temporal_default = "current"    # "current" (recent chunks first) or "all"
```

---

## 6. Success Criteria

The Cortex project is considered successful when:

1. **Ingestion works end-to-end:** A markdown file placed in `memory/meetings/` is automatically parsed, classified, tagged, and indexed in the knowledge graph, with no manual intervention.

2. **Temporal reasoning is correct:** When asked "Where does John live?" after multiple location mentions across different chunks, Cortex retrieves the most recent evidence chunk and the consuming LLM returns the current answer.

3. **Multi-hop queries succeed:** "Who else works on the project that John presented at the Q3 review?" traverses co-occurrence edges across multiple entities and returns relevant evidence chunks.

4. **Context assembly is efficient:** For a graph with 1000+ entities, context assembly produces a focused, token-budgeted prompt in under 2 seconds.

5. **Measurably better than vector RAG:** On the evaluation benchmark, Cortex outperforms the vector-only baseline on temporal and multi-hop questions by at least 20% accuracy.

6. **MCP integration works:** Claude Code or Desktop can use all Cortex tools to maintain and query persistent memory across sessions.

7. **Persistence works:** The graph survives process restarts. A crash during ingestion does not corrupt the graph (it falls back to the last good snapshot).

8. **LLM errors don't cause silent data loss:** Every LLM failure is logged, and the pipeline either retries, skips gracefully, or flags for review — never silently drops data.

9. **Review queue doesn't accumulate silently:** Unreviewed items auto-expire. Corroborated items auto-promote. Queue size is visible via `cortex_status`.

---

## 7. Timeline Summary

| Phase | Scope | Weeks | Calendar |
|---|---|---|---|
| **0a** | Fork & Rename | 0.5 | Week 1 |
| **0b** | Strip & Stub | 0.5 | Week 1 |
| **1** | Core Graph Schema + Persistence | 2–3 | Weeks 2–4 |
| **2** | Three-Pass Tag-and-Index Pipeline | 3 | Weeks 4–7 |
| **3** | Context Assembly | 1 | Weeks 7–8 |
| **4** | MCP Server Tools | 1 | Weeks 8–9 |
| **5** | Reflection & Consolidation | 1–2 | Weeks 9–10 |
| **6** | Incremental Updates | 1 | Weeks 10–11 |
| **7** | Evaluation Framework | 1–2 | Weeks 11–12 |
| | **Total** | **~10–12** | |

The optimistic target is 10 weeks. The realistic target is 12 weeks. The reduction from v3's 16–18 weeks comes from:

- **Phase 0b saves time:** Removing modules outright (CPG, CFG, dataflow, callgraph, import resolver) is faster than adapting them.
- **Phase 2 compresses by 3 weeks:** Coreference resolution is removed from the critical path. Triple extraction is eliminated. The two LLM passes are classification/tagging tasks with simpler prompts and more predictable outputs. No model validation sub-phase needed because the baseline quality of tagging is higher.
- **Phase 5 compresses by 1 week:** No typed-relation contradiction resolution. Temporal conflicts become "stale evidence detection," which is simpler.
- **Phase 7 compresses by 1 week:** Evaluation is simpler because there are fewer moving parts to benchmark (no triple extraction P/R/F1, no coreference accuracy).

The biggest schedule risk is now Phase 1 (type-signature rewrite in the Rust core) rather than Phase 2 (which was the high-risk phase in v3). If the graph schema needs iteration, budget an extra week.

---

## 8. Future Work (Explicitly Out of Scope for v1)

1. **Write-ahead log (WAL) for persistence:** Snapshot-only persistence is sufficient for v1. A WAL would reduce worst-case data loss from 5 minutes to near-zero.

2. **Typed co-occurrence edges:** If specific relationship types (e.g., "led", "decided", "blocked by") prove important for retrieval quality, add typed edge variants for high-value chunk types. This is an incremental addition, not a redesign — the co-occurrence graph structure supports it naturally.

3. **Coreference resolution as optional enhancement:** v3 treated this as critical-path. v4 drops it. As a future enhancement, a lightweight coreference pass between Pass 1 and Pass 2 could improve tagging accuracy for documents with heavy pronoun usage. But the system works without it.

4. **Multi-user / multi-tenant support:** v1 assumes a single user with a single knowledge graph.

5. **Graph visualization:** A web UI or CLI-rendered graph for exploring the knowledge graph interactively.

6. **Streaming ingestion:** v1 processes documents batch-style. Streaming from live transcripts is a different ingestion mode.

7. **Custom ontology definition:** v1 uses a fixed set of entity types with `Custom(String)` escape hatches. A future version could allow user-defined entity types in `.cortex.toml`.

8. **LightRAG-style dual-level retrieval:** LightRAG's low-level (specific entities) + high-level (abstract themes) retrieval pattern could be layered on top of the tag-and-index graph. The current architecture supports it — entity nodes serve the low level, and theme/concept nodes serve the high level — but it's not implemented in v1.

---

## 9. Changelog from v3

This section documents what changed between `cortex-implementation-plan.md` (v3) and this document (v4), and why.

| # | Change | Rationale |
|---|---|---|
| 1 | **Architecture shift: tag-and-index replaces triple extraction** | Feasibility analysis showed triple extraction at ~70% precision compounds through multi-hop traversals (0.7³ ≈ 34%). Tagging at 95%+ precision is fundamentally more reliable. The graph becomes a retrieval index (pointing to evidence chunks) rather than a knowledge base (storing interpreted relationships). This is the same architectural role Atlas already plays for code. |
| 2 | **Extraction pipeline: 3 passes replace 5 stages** | Pass 1 (structural parse) is deterministic — no LLM. Pass 2 (classify) is single-word classification — trivial for LLMs. Pass 3 (tag) is entity identification — reliable and verifiable. Coreference resolution and triple extraction are eliminated. |
| 3 | **Graph schema: CoOccurrenceEdge replaces RelationEdge with 18+ typed variants** | Edges now carry chunk type, timestamp, and evidence chunk ID — not typed predicates. This makes edges semantically thinner but eliminates the precision problem. Relationships are inferred at query time by the consuming LLM reading the evidence text. |
| 4 | **Coreference resolution removed from critical path** | v3 identified this as the riskiest sub-problem (Phase 2A). v4 drops it entirely — tagging works without it because entity names appear directly in text. Listed as future enhancement. |
| 5 | **Four Atlas modules removed (CPG, CFG, dataflow, callgraph)** | v3 adapted these for NL analogues (temporal overlay, event sequences, belief propagation, relationship graph). v4 removes them — co-occurrence edges replace all of these functions with a simpler mechanism. |
| 6 | **Chunk storage added to graph** | The graph now stores chunk text alongside entities and edges. Required because the graph is a retrieval index — when context assembly finds relevant entities, it needs to pull the underlying evidence text. |
| 7 | **Timeline compressed from 16–18 weeks to 10–12 weeks** | Phase 2 shrinks from 6 weeks to 3 (simpler pipeline). Phase 5 shrinks by 1 week (no contradiction resolution). Phase 0b is faster (removing modules vs adapting them). Phase 7 shrinks by 1 week (fewer metrics to benchmark). |
| 8 | **Risk profile substantially changed** | R1 (extraction quality) drops from High/Critical to Medium/High — tagging is inherently more reliable. R9 (coreference quality) is removed — no longer on critical path. New R4 (co-occurrence edges too thin) and R9 (consuming LLM can't infer from evidence) address the core tradeoff of the new architecture. New R10 (competitors ship similar features) reflects market awareness from feasibility analysis. |
| 9 | **Model routing simplified** | v3 had extraction_model vs reasoning_model with a complex decision rule. v4 has classification_model (can be tiny), tagging_model (start large, test small), and reasoning_model (stays large). Classification is simple enough that small models work from the start. |
| 10 | **Context serialization serves evidence chunks, not interpreted relations** | v3 serialized "John Doe → LED → Dashboard Redesign". v4 serializes the actual chunk text where John and Dashboard Redesign co-occur. The consuming LLM reads the evidence and infers the relationship — more reliable because reading comprehension is easier than relation typing. |
| 11 | **Human review queue scope narrowed** | v3 routed low-confidence triples, entity merges, and disambiguation to review. v4 routes only entity disambiguation borderline cases — tagging and classification are reliable enough that other review triggers are unnecessary. |
| 12 | **Temporal conflict model replaced with stale evidence detection** | v3 used ConflictsWith/SupersededBy edges to model contradictions between typed relations. v4 doesn't have typed relations to contradict — instead, the reflection loop detects when newer chunks supersede older chunks for the same entity, and deprioritizes the old chunks in context assembly. Simpler, and the consuming LLM can still see the full history if asked. |

---

## 10. Changelog from v2

Preserved from v3 for full audit trail. These changes were made between v2 and v3.

| # | Change | Rationale |
|---|---|---|
| 1 | **Entity embedding strategy made explicit (Phase 1C)** | v2 had `embedding: Option<Vec<f32>>` on EntityNode but never specified what gets embedded. |
| 2 | **Incremental updates simplified from four-tier to two-tier (Phase 6)** | Four-tier change classification doesn't transfer to NL documents because tier classification itself requires LLM calls. |
| 3 | **Human review queue lifecycle policy added (Phase 2D)** | v2 defined how items enter the queue but not what happens if nobody reviews them. |
| 4 | **Write-ahead log removed from v1 scope** | Half-commitment created ambiguity. Snapshot-only persistence is sufficient. |
| 5 | **Model routing revised: start with reasoning model** | Establishes quality ceiling before testing downgrades. |
| 6 | **Timeline extended to 16–18 weeks** | Honest accounting of extraction pipeline complexity. |
| 7 | **New risk R11 added** (JSON malformed rates) | Tied to model routing revision. |
| 8 | **Success criterion 9 added** (review queue lifecycle) | Follows from review queue lifecycle policy. |
| 9 | **Future Work section added** | Made scope boundaries explicit. |
| 10 | **`cortex_review` added to MCP tool table** | Users need MCP-accessible way to review queued items. |
| 11 | **Embedding references updated throughout** | Phases now reference contextual descriptor embedding strategy. |
| 12 | **Configuration schema expanded** | Added `[review]` section. Updated `[models]` defaults. |

---

## 11. Changelog from v1

Preserved from v2/v3 for full audit trail. These changes were made between v1 and v2.

| # | Change | Rationale |
|---|---|---|
| 1 | **Phase 0 split into 0a and 0b** | Separate rollback points for rename vs strip. |
| 2 | **Persistence strategy added to Phase 1** | Can't rebuild from source without LLM calls. |
| 3 | **Phase 2 time estimate doubled and split into sub-phases** | Three weeks for the highest-risk component was unrealistic. |
| 4 | **Temporal conflict model made explicit** | v1 had ConflictsWith but no resolution strategy. |
| 5 | **Effort estimates recalibrated** | Type-signature changes labeled "Medium-High". |
| 6 | **Phase 7A (test corpus) pulled into Phase 2** | Can't develop extraction without test data. |
| 7 | **LLM output error handling section added** | v1 described only the happy path. |
| 8 | **Risk register expanded** | Added persistence, LLM reliability, coreference, temporal conflict risks. |
| 9 | **Timeline updated from 12 to 14–16 weeks** | Honest accounting. |
| 10 | **Success criteria expanded** | Added persistence and LLM error resilience. |
| 11 | **Context serialization includes conflicts** | Consuming LLMs need to see contradictions. |
| 12 | **Configuration schema expanded** | Added persistence and extraction error-handling settings. |