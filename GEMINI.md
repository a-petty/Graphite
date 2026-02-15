## CRITICAL OPERATIONAL DIRECTIVES

1.  **ZERO-ASSUMPTION PROTOCOL:**
    *   You are strictly forbidden from assuming the existence of files, function signatures, or variable states.
    *   **Requirement:** You must verify the current state of any file (read it first) before proposing edits.
    *   **Requirement:** If a path or dependency is uncertain, you must stop and ask for clarification rather than hallucinating a path.

2.  **EXTREME CAUTION & VERIFICATION:**
    *   Operate with a "Do No Harm" implementation strategy. Your primary goal is to preserve existing functionality while implementing changes.
    *   **Requirement:** specific checks for side effects must be performed before any delete or overwrite operation.
    *   **Requirement:** When refactoring, prioritize small, reversible atomic changes over sweeping architectural updates unless explicitly requested.

3.  **MANDATORY CHANGE EXPLANATION:**
    *   Every single response involving code modification must end with a structured summary section.
    *   **Format:**
        ### Change Summary
        * **[File Name]**: [Specific change made] - [Reason for change]
        * **[File Name]**: [Specific change made] - [Reason for change]

### Project Overview

This project is a sophisticated, local-first autonomous coding agent named **Atlas** (Sentient Repository). It combines a high-performance Rust core with a Python orchestration layer to achieve a deep, symbol-level understanding of a codebase.

The agent is designed to run locally on a developer's machine. Its goal is to act as an intelligent assistant that can reason about code structure, dependencies, and semantics, rather than just performing simple text-based operations.

-   **Purpose:** To create a local, autonomous coding agent that understands the structure of a repository.
-   **Core Technologies:**
    -   **Rust Core:** For performance-critical tasks like file scanning, code parsing (`tree-sitter`), and building a semantic graph of the codebase (`petgraph`). It is exposed to Python using `PyO3`.
    -   **Python Shell:** For orchestrating the agent, interacting with local Large Language Models (`ollama`), managing vector embeddings for semantic search (`fastembed`), and providing file system tools to the agent.
-   **Architecture:** Atlas uses a hybrid model:
    1.  The **Rust core** first scans the repository, parses all source files into Abstract Syntax Trees (ASTs), and extracts symbols (functions, classes, variables). It uses this information to build a "Repository Graph" that maps not just file imports, but actual symbol usages between files.
    2.  The **Python shell** then uses this graph to provide rich context to the LLM. It employs a unique "Anchor & Expand" strategy: using vector search to find code relevant to a user's query (the "anchor") and then using the graph to pull in all related dependencies (the "expansion"). This gives the agent a deep understanding of the code's architecture.
    3.  A **"Reflexive Sensory Loop"** ensures that any code the agent writes is first validated for syntactic correctness by the fast Rust parser before it's saved to disk, allowing the agent to self-correct.

### Building and Running

The project is managed as a hybrid Rust/Python project using `maturin`.

**1. Building the Rust Core:**

The Rust library must be compiled into a Python module. This is handled by `maturin`.

```bash
# To install maturin (if not already installed in your venv)
pip install maturin

# To build the Rust core and install it in the current Python environment
# Note: This must be run from within the virtual environment (.venv)
source .venv/bin/activate
cd rust_core
maturin develop
```

**2. Running Tests:**

The project contains tests for both the Rust core and the Python shell.

-   **Rust:**
    ```bash
    cd rust_core
    cargo test
    ```
-   **Python:** (Using `pytest`)
    ```bash
    cd python_shell
    pytest
    ```

### Development Conventions

-   **Hybrid Structure:** All performance-sensitive code (parsing, graph logic) should be in the `rust_core` library. All agent orchestration, LLM interaction, and tool implementation should be in the `python_shell`.
-   **Data Structures:** The core data structures (like `RepoGraph` and `SymbolIndex`) are defined in Rust for efficiency and are exposed to Python.
-   **Parsing:** Code parsing is done exclusively with `tree-sitter` via the Rust core to ensure consistency and performance. `tree-sitter` queries for various languages are stored in the `rust_core/queries` directory.
-   **Dependencies:** Rust dependencies are managed in `rust_core/Cargo.toml`. Python dependencies are likely managed in a `pyproject.toml` or `requirements.txt` file in the root or `python_shell` directory.
-   **Contribution:** Changes should be accompanied by tests in the respective `tests/` directory. Benchmarks are encouraged for performance-critical Rust code.