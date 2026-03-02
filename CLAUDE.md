# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

The project owner is a systems thinker without formal SWE background. Their engineering team validates code but Claude is the primary day-to-day collaborator. This means:

- **Explain what you built.** Every code change gets a plain-English walkthrough — what it does, why, and any caveats. Do not make the user ask for this.
- **Ask before assuming.** If requirements are ambiguous or context is missing (data structures, constraints, how a feature connects to existing code), stop and ask. Do not silently fill gaps.
- **One thing at a time.** Deliver one piece, explain it, let the user confirm it works, then move on. Compound deliverables create compound errors.
- **State your confidence level.** Say "I'm confident this is correct" vs "this should work but test with X" vs "I'm uncertain here — flag for engineer review." Never present uncertain output with certain tone.
- **Flag for engineer review:** authentication, passwords, payments, sensitive data, file uploads, destructive operations, external API integrations, database migrations, deployment config, or anything you're not fully confident in.
- **Prefer simple over clever.** Verbose and obvious beats compact and subtle. Well-established over novel. Explicit error handling over assuming success.
- **Never skip validation or error handling**, even in prototypes. Never store passwords in plain text, build SQL from string concatenation, or expose internal errors to users.

## Build & Development

Atlas is a hybrid Rust + Python project using PyO3/Maturin for FFI.

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
cargo test --test test_callgraph                         # Single test file
cargo test --test test_callgraph test_resolve_same_file  # Single test function

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
mypy python_shell/atlas/   # Type checking (mypy is in dev deps)
```

### Running

```bash
atlas-mcp --project-root /path/to/repo          # MCP server (primary use case)
atlas-mcp --project-root /path/to/repo --verbose # With debug logging
atlas watch /path/to/repo                        # File watcher mode
atlas query "question" -p /path/to/repo          # One-shot query
```

## Architecture

Two layers connected via PyO3:

**Rust core** (`rust_core/src/`, edition 2024) — Parsing, graph construction, PageRank, CPG analysis. All Tree-sitter parsing happens here. The compiled extension is importable as `atlas.semantic_engine` (configured via `tool.maturin.module-name` in `pyproject.toml`).

**Python shell** (`python_shell/atlas/`) — MCP server, context assembly, embeddings, LLM clients. Orchestrates the Rust core and exposes 12 MCP tools.

### Key data flow

1. `RepoGraph` (Rust, `graph.rs`) scans files with Tree-sitter, builds a `petgraph::DiGraph<FileNode, EdgeKind>` with Import (weight 2.0) and SymbolUsage (weight 1.0) edges, runs PageRank (20 iterations, damping 0.85)
2. `EmbeddingManager` (`embeddings.py`) embeds file skeletons with BAAI/bge-small-en-v1.5 via FastEmbed, using chunk-level max-similarity scoring
3. `ContextManager` (`context.py`) runs Anchor & Expand: semantic search finds relevant files (anchor), BFS walks the dependency graph (expand), then assembles a 3-tier response:
   - **Tier 1** (~8% budget): PageRank-ordered repo map
   - **Tier 2** (40-75%): Full content of anchor files + graph neighbors
   - **Tier 3** (remainder): Skeletons of architecturally relevant files

### Critical Rust modules

| Module | Owns |
|--------|------|
| `graph.rs` | `RepoGraph`, `FileNode`, `EdgeKind`, PageRank, incremental updates (`UpdateTier`) |
| `cpg.rs` | `CpgLayer` — sub-file CFG/dataflow/call-graph (Python only) |
| `callgraph.rs` | Call site extraction + cross-file resolution using import bindings |
| `parser.rs` | Tree-sitter parsing, `SymbolHarvester`, skeleton generation, 5 languages (Python, Rust, JS/JSX, TS/TSX, Go) |
| `import_resolver.rs` | `PythonImportResolver`, `JsTsImportResolver` — maps import statements to file paths |
| `symbol_table.rs` | `SymbolIndex` — bidirectional name↔file mappings |
| `lib.rs` | PyO3 bindings (`PyRepoGraph`), error type mapping |

### Critical Python modules

| Module | Owns |
|--------|------|
| `mcp_server.py` | FastMCP server, 12 tool definitions, lazy graph/CPG/embedding init |
| `context.py` | `ContextManager`, Anchor & Expand, adaptive 3-tier budgeting |
| `embeddings.py` | `EmbeddingManager`, FastEmbed ONNX, skeleton-based chunked embeddings |
| `agent.py` | `AtlasAgent`, file watch loop, tool orchestration |

### Design invariants

- **Lazy initialization**: Graph builds on first tool call, CPG on first CPG tool call, embeddings on first semantic search. Never eagerly.
- **Conservative call resolution**: CPG only creates call edges for unambiguously resolved calls. No guessing.
- **Incremental updates**: File changes are classified as Local/FileScope/GraphScope/FullRebuild. Only GraphScope+ triggers PageRank recalc.
- **petgraph swap-remove**: `remove_node()` uses swap-remove, so all side-maps (symbol index, CPG `file_to_nodes`, `path_to_idx`) must be explicitly remapped when removing nodes.
- **Tree-sitter queries**: Language-specific queries live in `rust_core/queries/`. These define how symbols are harvested from each language's AST.

## Project config files

- `.atlas.toml` — Optional per-repo config (custom `source_roots`)
- `.atlasignore` — Optional per-repo ignore patterns (like `.gitignore`)
- Tree-sitter queries in `rust_core/queries/{python,rust,javascript,typescript}/`
