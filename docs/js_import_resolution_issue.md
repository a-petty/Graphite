# Issue: `RepoGraph` Generation Fails for JavaScript/TypeScript Projects

## 1. Summary

The Atlas agent's Rust core, specifically the `RepoGraph` module, is designed to be language-agnostic. However, testing has revealed that its current implementation only functions correctly for Python projects. When attempting to generate a repository map for a JavaScript/TypeScript project (`test_repo`), the process completes but produces a graph with zero dependency edges, resulting in a meaningless PageRank score where all files are ranked as 0.

This fundamentally blocks the agent's ability to reason about non-Python codebases.

## 2. Discovery Process

The issue was discovered through a series of step-by-step tests intended to debug the connection between the Python orchestration shell and the Rust core.

1.  **Initial Test:** A Python script (`python_shell/generate_map_script.py`) was created to call the `RepoGraph::generate_map` method on the `test_repo`.
2.  **First Failure:** The initial runs produced an empty graph ("0 files") or failed due to incorrect assumptions about the Rust core's API (e.g., calling a non-existent `build()` method).
3.  **API Correction:** After inspecting the PyO3 bindings in `rust_core/src/lib.rs`, the script was corrected to use the proper `scan_repository` and `build_complete` methods.
4.  **Second Failure (The Root Cause):** The corrected script successfully scanned all files and built a graph containing over 4,000 nodes (files). However, the generated `repo_map.log` showed that every single file had a PageRank of `0.000` and an import count of `0`. This indicates a complete failure to create any dependency edges.
5.  **Code Investigation:** A direct review of the Rust source code, specifically `rust_core/src/import_resolver.rs`, confirmed the root cause.

## 3. Root Cause Analysis

The `RepoGraph`'s failure to build edges for JavaScript/TypeScript is not a bug in the graph logic itself, but a limitation of its language-specific dependencies. The entire import resolution and symbol harvesting pipeline is hardcoded for Python.

Key evidence from `rust_core/src/import_resolver.rs`:

*   **Python-Only File Indexing:** The `index_modules` function explicitly filters for `.py` and `.pyi` files, ignoring all other extensions.
*   **Python-Specific Tree-sitter Query:** The query used to find imports is written for Python's `import` and `from ... import` syntax. It cannot parse JavaScript's `import`/`export` or CommonJS `require()`.
*   **Python-Centric Logic:** The resolver's logic for converting file paths to module names is based on Python's packaging rules (e.g., handling `__init__.py`). It also contains large, hardcoded sets of Python standard libraries to exclude from the graph, which are irrelevant for JavaScript.

Because the `ImportResolver` finds no modules and resolves no imports, no `EdgeKind::Import` edges are created. It is highly likely the `SymbolHarvester` is similarly Python-specific, leading to no `EdgeKind::SymbolUsage` edges being created either. The result is a graph of thousands of disconnected nodes.
