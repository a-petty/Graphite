# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atlas is a local-first autonomous coding agent that combines a high-performance Rust core with Python orchestration. It builds symbol-aware semantic graphs of repositories and provides intelligent context to LLMs for code modifications.

## Build & Development

The project uses **Maturin** to bridge Rust and Python. The Rust core compiles into a Python extension module (`atlas.semantic_engine`).

```bash
# Activate virtual environment first
source .venv/bin/activate

# Build Rust core and install as Python extension (run from project root)
maturin develop

# Run Rust tests
cd rust_core && cargo test

# Run a single Rust test
cargo test --test test_graph_construction

# Run Python tests
cd python_shell && pytest

# Run a single Python test
pytest python_shell/tests/test_tools.py

# Run Rust benchmarks
cd rust_core && cargo bench

# Run the CLI
atlas watch /path/to/repo
atlas query "your question" -p /path/to/repo
```

After any change to Rust code, you must run `maturin develop` before Python tests will reflect those changes.

## Architecture

### Hybrid Rust/Python Model

**Rust Core (`rust_core/src/`)** — Performance-critical code:
- `graph.rs` — `RepoGraph` built on petgraph `DiGraph<FileNode, EdgeKind>`. Nodes are files, edges are `Import` or `SymbolUsage` relationships. Includes PageRank scoring.
- `parser.rs` — Tree-sitter parsing across 6 languages (Python, Rust, JS, TS, Go, Java). `ParserPool` manages thread-safe parser instances. Also generates code "skeletons" (signatures only).
- `symbol_table.rs` — `SymbolIndex` mapping symbol names to file locations.
- `import_resolver.rs` — Resolves import statements to file paths per language.
- `watcher.rs` — File system monitoring via `notify` crate with debouncing and filtering.
- `incremental_parser.rs` — Change-aware re-parsing to avoid full rebuilds.
- `lib.rs` — PyO3 bindings exposing `RepoGraph`, `FileWatcher`, `check_syntax`, `scan_repository`, etc. to Python.

**Python Shell (`python_shell/atlas/`)** — Orchestration layer:
- `agent.py` — `AtlasAgent` orchestrator: initializes graph, manages file watching, handles queries, implements the "Reflexive Sensory Loop" (syntax validation before saving files).
- `context.py` — `ContextManager` using "Anchor & Expand" strategy: vector search finds relevant code (anchor), then graph traversal pulls in dependencies (expand). Three-tier token budgeting: Tier 1 (5%) repo map, Tier 2 (50%) full file content, Tier 3 (45%) skeletons of high-PageRank files.
- `llm.py` — LLM clients (Ollama for local models).
- `tools.py` — `ToolExecutor` for file read/write with syntax checking.
- `embeddings.py` — FastEmbed-based vector search.
- `cli.py` — Entry point (`atlas` command). Subcommands: `watch`, `query`.

### Key Design Patterns

- **All parsing goes through Tree-sitter** via the Rust core — never parse source code in Python.
- **PyO3 boundary**: Rust types have `Py*` wrapper structs in `lib.rs` (e.g., `PyRepoGraph` wraps `graph::RepoGraph`). Error conversion uses `From<graph::GraphError> for PyErr`.
- **Incremental updates**: File changes are classified as structural (import changes) vs. local (body-only). Only structural changes trigger edge recalculation and PageRank recomputation.
- **Python syntax checking** uses Python's native `compile()` via PyO3 for accuracy; other languages use tree-sitter.

## Conventions

- Performance-sensitive code belongs in `rust_core/`. Agent orchestration, LLM interaction, and tool implementation belong in `python_shell/`.
- Verify the current state of any file before proposing edits. Do not assume file contents, function signatures, or variable states.
- Prefer small, reversible atomic changes over sweeping architectural updates.
- Rust edition is 2024. Python requires >=3.9.
