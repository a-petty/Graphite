# Cortex

A knowledge graph memory system for LLMs. Cortex ingests natural language documents — meeting notes, associate profiles, completed work — and builds a persistent knowledge graph that any LLM can query via the [Model Context Protocol](https://modelcontextprotocol.io/).

Instead of parsing source code, Cortex uses a **tag-and-index architecture**: an LLM-based extraction pipeline tags entities in document chunks and links them through co-occurrence edges. Relationships aren't stored as typed predicates — they're inferred at query time by the consuming LLM reading the actual evidence text where entities appear together.

Built on a high-performance **Rust core** (petgraph, PageRank, persistence) with a **Python orchestration layer** (extraction pipeline, context assembly, MCP server), connected via PyO3.

## Key Capabilities

- **Tag-and-index extraction pipeline** — Three-pass document processing: deterministic structural parsing (no LLM) splits documents into chunks, LLM classification labels chunk types, LLM tagging extracts and disambiguates entities. Co-occurrence edges connect entity pairs that appear in the same chunk.

- **Knowledge graph with PageRank** — Entities (people, projects, technologies, organizations, decisions, concepts) are nodes in a petgraph DiGraph. Co-occurrence edges carry chunk references, timestamps, and memory categories. PageRank surfaces the most connected entities.

- **Anchor & Expand context assembly** — Semantic vector search finds query-relevant entities (anchor), BFS walks the co-occurrence graph with temporal filtering (expand), retrieves evidence chunks, and assembles a three-tier response: knowledge map, evidence chunks, and peripheral entity summaries.

- **MCP server with 18 tools** — Entity lookup, co-occurrence analysis, semantic search, timeline queries, evidence retrieval, document ingestion, reflection/consolidation, and graph management. Integrates with Claude Code, VS Code, and any MCP-compatible client.

- **Reflection and consolidation** — Background operations merge duplicate entities (alias overlap + embedding similarity), apply temporal decay to access counts, prune orphan entities, and synthesize higher-order insights via LLM.

- **Incremental updates** — Two-tier document change model: content hashing detects modified documents and re-runs the extraction pipeline; document removal cascade-cleans chunks, edges, and orphaned entities.

- **Persistence** — MessagePack serialization with backup/recovery. Graph state persists across sessions — no need to re-run expensive LLM extraction.

## Architecture

```
                        ┌───────────────────────────────────────────────┐
                        │          MCP Server (mcp_server.py)           │
                        │  18 tools: status, knowledge map, entities,   │
                        │  evidence, timeline, ingest, reflect, ...     │
                        └─────────────────────┬─────────────────────────┘
                                              │
┌─────────────────────────────────────────────┼─────────────────────────────────────┐
│  Python Shell (python_shell/cortex/)        │                                     │
│                                             │                                     │
│  ┌──────────────────┐  ┌──────────────────┐ │   ┌────────────┐  ┌─────────────┐  │
│  │   Extraction     │  │ Context Manager  │ │   │  Embedding │  │    LLM      │  │
│  │   Pipeline       │  │ (anchor+expand,  │◄┘   │  Manager   │  │  Clients    │  │
│  │  structural_     │  │  3-tier budget,  │     │ (FastEmbed,│  │ (Ollama,    │  │
│  │  parser →        │  │  temporal filter,│     │  cosine    │  │  MLX)       │  │
│  │  classifier →    │  │  adaptive params)│     │  similarity│  │             │  │
│  │  tagger          │  └────────┬─────────┘     └────────────┘  └─────────────┘  │
│  └──────┬───────────┘           │                                                │
│         │   ┌───────────────────┤                                                │
│         │   │  Reflection       │                                                │
│         │   │  consolidator +   │           PyO3 boundary                        │
│         │   │  synthesizer      │                                                │
│         │   └───────────────────┘                                                │
├─────────┼───────────────────────┼────────────────────────────────────────────────┤
│  Rust Core (rust_core/src/)     │                                                │
│                                 │                                                │
│  ┌──────────────────────────────┼─────────────────────────────────────────────┐  │
│  │  KnowledgeGraph (knowledge_graph.rs)                                       │  │
│  │  ┌────────────────┐  ┌──────┴──────┐  ┌─────────────┐  ┌──────────────┐   │  │
│  │  │  EntityNode    │  │  PageRank   │  │  Tag Index  │  │   Chunk      │   │  │
│  │  │  DiGraph with  │  │ (weighted,  │  │ name→entity │  │   Storage    │   │  │
│  │  │  CoOccurrence  │  │  iterative) │  │ alias→node  │  │ id→text+meta │   │  │
│  │  │  Edges         │  │             │  │ doc→entities│  │              │   │  │
│  │  └────────────────┘  └─────────────┘  └─────────────┘  └──────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  ┌─────────────────────────┐  ┌────────────────────────────────────────────┐     │
│  │  Persistence            │  │  Watcher (watcher.rs)                     │     │
│  │  (persistence.rs)       │  │  FSEvents (macOS) + notify crate          │     │
│  │  MessagePack save/load  │  │  100ms debouncing, .gitignore-aware       │     │
│  │  Backup + crash recovery│  │  crossbeam-channel to main thread         │     │
│  └─────────────────────────┘  └────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. Documents land in `memory/{meetings,associates,work}/`
2. **Three-pass pipeline** (Python): structural parse (deterministic chunking) → classify chunks via LLM → tag entities via LLM → disambiguate against existing graph
3. **KnowledgeGraph** (Rust) stores `EntityNode`s and `CoOccurrenceEdge`s in a petgraph DiGraph. Edges link entity pairs that co-occur within the same chunk. Chunk text is stored alongside the graph for O(1) evidence retrieval.
4. **EmbeddingManager** (Python) embeds entity contextual descriptors with BAAI/bge-small-en-v1.5 via FastEmbed
5. **ContextManager** (Python) runs Anchor & Expand: semantic search finds relevant entities (anchor), BFS walks the co-occurrence graph with temporal filtering (expand), retrieves evidence chunks, assembles a three-tier response:
   - **Tier 1** (~10% budget): PageRank-ordered knowledge map
   - **Tier 2** (40-70%): Evidence chunks (actual text where entities co-occur)
   - **Tier 3** (remainder): Compressed entity summaries

## Quick Start

### Prerequisites

- Python >= 3.10
- Rust toolchain ([rustup](https://rustup.rs/))
- [Maturin](https://www.maturin.rs/) (`pip install maturin`)
- [Ollama](https://ollama.ai/) (for LLM-powered extraction)

### Installation

```bash
git clone https://github.com/a-petty/Cortex.git
cd Cortex

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -e ".[dev]"

# Build the Rust core as a Python extension
maturin develop
```

After any change to Rust code, you must run `maturin develop` before Python code will reflect those changes.

### Optional Dependencies

```bash
# MCP server support (for Claude Code / VS Code integration)
pip install -e ".[mcp]"

# Apple Silicon local LLM inference via MLX
pip install -e ".[mlx]"
```

## Usage

### MCP Server (Claude Code, VS Code, etc.)

The primary way to use Cortex is as an MCP server that provides knowledge graph memory to AI tools.

```bash
# Start the MCP server
cortex-mcp --project-root /path/to/project

# With debug logging
cortex-mcp --project-root /path/to/project --verbose
```

**Claude Code configuration** (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "cortex": {
      "command": "/path/to/Cortex/.venv/bin/cortex-mcp",
      "args": ["--project-root", "/path/to/your/project"]
    }
  }
}
```

**VS Code configuration** (`.vscode/mcp.json`):
```json
{
  "servers": {
    "cortex": {
      "command": "/path/to/Cortex/.venv/bin/cortex-mcp",
      "args": ["--project-root", "${workspaceFolder}"]
    }
  }
}
```

The MCP server uses lazy initialization — the graph loads from disk on first tool call, embeddings initialize on first semantic search, and LLM clients initialize on first ingestion.

### MCP Tools

Cortex exposes 18 tools over MCP, organized into five groups:

**Entity & Graph Lookup**

| Tool | Description |
|------|-------------|
| `cortex_status` | Graph statistics and readiness status |
| `get_knowledge_map` | PageRank-ordered knowledge map of entities |
| `get_cooccurrences` | Co-occurring entities and their relationship evidence |
| `get_entity_mentions` | Chunks where an entity appears |
| `get_key_entities` | Top entities by PageRank, optionally filtered by type |
| `get_entity_profile` | Full profile: type, aliases, documents, co-occurrences |
| `get_timeline` | Temporal chain of an entity's appearances |
| `get_evidence` | Evidence chunks where two specific entities co-occur |
| `get_entity_summary` | Synthesized summary of an entity |

**Search & Context Assembly**

| Tool | Description |
|------|-------------|
| `find_relevant_entities` | Semantic search via vector embeddings |
| `assemble_memory` | Core intelligence: Anchor & Expand with three-tier budgeting and temporal filtering |

**Document Ingestion**

| Tool | Description |
|------|-------------|
| `cortex_ingest` | Ingest a file or directory into the knowledge graph |
| `cortex_reingest` | Re-ingest all tracked documents from scratch |
| `cortex_update_document` | Re-ingest a single changed document (content-hash based) |
| `cortex_remove_document` | Remove a document and cascade-clean its artifacts |

**Reflection & Management**

| Tool | Description |
|------|-------------|
| `cortex_reflect` | Run entity merging, orphan cleanup, decay, and synthesis |
| `cortex_forget` | Permanently remove an entity from the graph |
| `cortex_review` | Show merge candidates for human review |

### Standalone CLI

```bash
# Ingest documents into the knowledge graph
cortex ingest /path/to/project

# One-shot query against the knowledge graph
cortex query "What decisions were made about authentication?" -p /path/to/project

# Watch a directory for changes and auto-ingest
cortex watch /path/to/project
```

The CLI requires [Ollama](https://ollama.ai/) (default) or MLX for LLM inference.

## Configuration

### `.cortex.toml` (optional)

Place at the project root to customize extraction, context assembly, and reflection behavior:

```toml
[llm]
provider = "ollama"
model = "llama3.3:70b"
temperature = 0.1
max_tokens = 4096

[extraction]
auto_merge_threshold = 0.85
review_threshold = 0.70
max_chunk_tokens = 800

[context]
tier1_budget_pct = 0.10
tier2_budget_pct = 0.60
similarity_weight = 0.80
pagerank_weight = 0.20

[reflection]
decay_half_life_days = 30.0
orphan_max_age_days = 7
merge_embedding_threshold = 0.90
merge_alias_overlap_threshold = 0.80
lightweight_reflection_on_ingest = true

[paths]
memory_root = "memory"
```

### `.cortexignore` (optional)

Exclude directories from document scanning. One pattern per line; `#` for comments:

```
# Exclude non-document directories
node_modules/
.git/
__pycache__/
```

## How It Works — A Concrete Example

Scenario: Your team has 50 meeting transcripts, 20 associate profiles, and 30 project documents in a `memory/` directory. You ask a Cortex-powered Claude Code: *"What has Sarah said about the authentication redesign?"*

**1. Ingestion** (already done): Cortex parsed all documents through the three-pass pipeline. The structural parser split meeting transcripts on speaker turns and markdown documents on headers. The LLM classified chunks (decisions, discussions, action items) and tagged entities (Sarah Chen → Person, auth-redesign → Project). Co-occurrence edges link "Sarah Chen" and "auth-redesign" in every chunk where they appear together.

**2. Anchor**: Embedding search finds the entities most relevant to the query: `Sarah Chen`, `auth-redesign`, `authentication`, `OAuth 2.0`.

**3. Expand**: BFS walks the co-occurrence graph from anchors — Sarah co-occurs with "auth-redesign" in 8 chunks, with "OAuth 2.0" in 3 chunks, with "security-review" in 2 chunks. Temporal filtering ensures results are from the relevant time period.

**4. Context assembly**: Tier 1 gets the knowledge map (~2K tokens showing key entities and their connections). Tier 2 gets the actual evidence chunks (~15K tokens of meeting transcript excerpts where Sarah discusses authentication). Tier 3 gets summaries of peripheral entities (~5K tokens about related projects and decisions).

**5. Result**: The model receives Sarah's exact words from meeting transcripts, ordered chronologically, with surrounding context about the decisions made and who else was involved. It can answer with specific quotes and dates rather than generic summaries.

## Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Build Rust core and install as Python extension
maturin develop

# Run all Rust tests (70 tests)
cd rust_core && cargo test

# Run a single Rust test file
cargo test --test test_knowledge_graph

# Run a single Rust test by name
cargo test --test test_knowledge_graph test_merge_entities

# Run all Python tests (288 tests)
pytest python_shell/tests/

# Run a single Python test file
pytest python_shell/tests/test_memory_context.py

# Rust linting and formatting
cd rust_core && cargo fmt --check && cargo clippy

# Python type checking
mypy python_shell/cortex/
```

### Test Suites

**Rust** (`rust_core/tests/`): 8 test files — knowledge graph operations (entities, edges, chunks, merging, temporal queries, document removal, PageRank), persistence roundtrip, tag index, graph construction, incremental updates, repo map generation, PageRank, and file watching.

**Python** (`python_shell/tests/`): 9 test files — memory context assembly (anchor/expand, tiered budgeting, temporal filtering, adaptive params), extraction pipeline (structural parser, classifier, tagger, disambiguation), reflection (merge candidates, orphan cleanup, decay, synthesis), ingestion pipeline, incremental updates, embeddings, chat/agent loop, bug fix regression tests.

### Project Structure

```
Cortex/
├── rust_core/
│   ├── src/
│   │   ├── knowledge_graph.rs  # KnowledgeGraph: entity/edge graph, PageRank, queries
│   │   ├── entity.rs           # EntityNode, EntityType, MergeRecord
│   │   ├── chunk.rs            # Chunk storage and metadata
│   │   ├── cooccurrence.rs     # CoOccurrenceEdge types
│   │   ├── tag_index.rs        # Tag index: name/alias/type/document → entity lookup
│   │   ├── persistence.rs      # Save/load/backup (MessagePack)
│   │   ├── watcher.rs          # File system monitoring via notify crate
│   │   ├── graph.rs            # Legacy RepoGraph (retained, not used by Cortex)
│   │   └── lib.rs              # PyO3 bindings: PyKnowledgeGraph, PyFileWatcher
│   └── tests/                  # 8 integration test files
├── python_shell/
│   └── cortex/
│       ├── mcp_server.py       # MCP server: 18 tools, lazy init, stdio transport
│       ├── context.py          # MemoryContextManager: Anchor & Expand, 3-tier budgeting
│       ├── embeddings.py       # EmbeddingManager: FastEmbed entity/chunk vectors
│       ├── config.py           # CortexConfig: validated settings with .cortex.toml support
│       ├── extraction/         # Three-pass tag-and-index pipeline
│       │   ├── structural_parser.py  # Pass 1: deterministic chunking
│       │   ├── classifier.py         # Pass 2: LLM chunk classification
│       │   └── tagger.py             # Pass 3: LLM entity tagging
│       ├── ingestion/          # Document ingestion orchestration
│       │   ├── pipeline.py     # IngestionPipeline: 3-pass orchestrator
│       │   ├── categorizer.py  # Directory → memory category mapping
│       │   └── chunker.py      # Chunking utilities
│       ├── reflection/         # Background consolidation
│       │   ├── consolidator.py # Entity merging, orphan cleanup, decay
│       │   └── synthesizer.py  # LLM-powered higher-order insights
│       ├── llm.py              # LLM clients: Ollama, MLX
│       ├── agent.py            # CortexAgent: orchestrator, watch loop
│       └── cli.py              # CLI: ingest, query, watch subcommands
├── memory/                     # Default document repository
│   ├── meetings/               # Episodic memory (meeting transcripts)
│   ├── associates/             # Semantic memory (people profiles)
│   └── work/                   # Procedural memory (completed work)
├── pyproject.toml
├── .cortex.toml                # Per-project configuration
└── CLAUDE.md                   # AI coding assistant instructions
```

### Key Design Decisions

- **Tag-and-index, not triple extraction**: The graph stores co-occurrence edges (which entities appear together in which chunks), not typed predicates. Relationships are inferred at query time by the consuming LLM reading evidence chunks. This avoids the brittleness of LLM-extracted relationship types.
- **Conservative disambiguation**: Only auto-merge entities at >0.90 embedding similarity or >0.95 alias overlap. Borderline cases go to a human review queue via `cortex_review`.
- **Post-extraction validation**: Tagged entity names are verified against source chunk text. Hallucinated entities are removed before graph insertion.
- **Swap-remove safety**: `petgraph::DiGraph::remove_node` uses swap-remove, so all side-maps (entity index, tag index, chunk references) are explicitly remapped when removing nodes.
- **Persistence required**: The graph is saved after ingestion and periodically during watch mode. Unlike code graphs, knowledge graphs can't be cheaply rebuilt since extraction requires LLM calls.
- **Lazy initialization**: Graph loads on first tool call, embeddings on first semantic search. Never eagerly.
- **PyO3 boundary**: Rust types have `Py*` wrapper structs in `lib.rs` (e.g., `PyKnowledgeGraph` wraps `KnowledgeGraph`). All data crosses the boundary as JSON strings.

## Rust Core Dependencies

| Crate | Purpose |
|-------|---------|
| `petgraph` | Directed graph for entities and co-occurrence edges |
| `pyo3` | Python-Rust FFI bindings |
| `serde` + `rmp-serde` | MessagePack serialization for persistence |
| `uuid` | Entity ID generation |
| `chrono` | Timestamp handling |
| `notify` + `notify-debouncer-full` | File system monitoring with 100ms debouncing |
| `ignore` | .gitignore-aware file walking |
| `crossbeam-channel` | Thread-safe message passing for watcher events |
| `rayon` | Parallel processing |
| `parking_lot` | Efficient mutex/sync primitives |

## Python Dependencies

| Package | Purpose |
|---------|---------|
| `tiktoken` | Token counting for context budget management |
| `fastembed` | Vector embeddings (BAAI/bge-small-en-v1.5) for semantic search |
| `ollama` | LLM client for extraction pipeline and synthesis |
| `numpy` | Embedding vector operations |
| `rich` | Terminal formatting |
| `pydantic` | Data validation |
| `mcp[cli]` | MCP server framework (optional) |
| `mlx-lm` | Apple Silicon LLM inference (optional) |

## License

MIT License. See [LICENSE](LICENSE) for details.
