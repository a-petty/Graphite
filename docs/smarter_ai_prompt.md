# Prompt for AI-Assisted Debugging: `SymbolUsage` Edge Failure

## High-Level Goal

Fix the persistently failing Rust test `test_symbol_usage_edge_creation` in the `semantic_engine` crate. The test is failing because a graph edge that should be `EdgeKind::SymbolUsage` is incorrectly remaining as `EdgeKind::Import`.

## Instructions

You are an expert Rust developer specializing in static analysis and `tree-sitter`. Your task is to analyze the provided context and files to create a definitive plan to resolve the failing test.

### Step 1: Understand the Problem Context

First, read and fully understand the detailed problem description, my debugging process, and the core mystery outlined in this file:
- `docs/smarter_ai_prompt_context.md`

### Step 2: Review the Key Files

Next, thoroughly review the following key files. They are essential to understanding the interaction between parsing, symbol harvesting, and graph construction.

- **`rust_core/src/parser.rs`**: **(Primary Focus)** Contains the `SymbolHarvester::harvest` method, which is the likely source of the bug.
- **`rust_core/queries/python/symbols.scm`**: **(Primary Focus)** The `tree-sitter` query file for Python symbols. Note the `@usage.*` capture names.
- **`rust_core/src/graph.rs`**: Contains the core graph logic, including `build_complete`, `build_semantic_edges`, and the `ensure_edge` function.
- **`rust_core/tests/test_graph_construction.rs`**: Contains the exact test case that is failing.

### Step 3: Analyze and Formulate a Plan

A preliminary investigation has uncovered a critical finding:

**Hypothesis:** The root cause is a mismatch in the `SymbolHarvester::harvest` method in `parser.rs`. The method is hardcoded to only recognize symbol usages from captures named `@reference.*`, but the Python query file (`symbols.scm`) now uses a more specific `@usage.*` naming convention. As a result, no usages are ever found for Python files, the `build_semantic_edges` function does no work, and the initial `Import` edge is never upgraded.

Your task is to:
1.  **Confirm this hypothesis.** Verify that this mismatch is indeed the root cause of the failure.
2.  **Propose a specific, surgical fix.** Provide a detailed, step-by-step plan to modify the `SymbolHarvester::harvest` method in `rust_core/src/parser.rs`. The fix should be robust enough to handle **both** `@reference.*` and `@usage.*` captures, ensuring that you don't break the working functionality for JavaScript and TypeScript.
3.  **Provide the exact code modification** needed for `rust_core/src/parser.rs` to implement your proposed plan. There should be no other file changes.