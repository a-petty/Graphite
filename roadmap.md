Here is the improved **Project Atlas: Implementation Roadmap**, upgraded from a naive architectural scanner to a symbol-aware, reflexive intelligence engine.

I have integrated the "Stack Graph Lite" architecture, the "Anchor & Expand" context strategy, and the "Reflexive Sensory Loop" into the existing design structure.

---

# PROJECT ATLAS: COMPLETE IMPLEMENTATION ROADMAP

## Sentient Repository - Autonomous Coding Agent

**Target:** Local, autonomous coding agent for MacBook Pro M4 Max (36GB RAM)

**Core Model:** DeepSeek-R1-Distill-Qwen-32B (4-bit quantized)

**Architecture:** Hybrid Rust-Core (Performance) / Python-Shell (Orchestration)

---

## DESIGN PHILOSOPHY

### The Three Constraints

1. **Intelligence-Per-Token:** Maximize semantic density in context window
2. **Sub-Second Response:** Agent must feel "alive," not batch-processing
3. **Graceful Degradation:** Must work with broken/incomplete code

### What We're Building

A **symbol-level structural reasoning engine** that converts raw code into queryable semantic graphs. Unlike simple file scanners, this enables the agent to trace execution flow across boundaries (e.g., knowing *who* calls `auth.login()`, not just which file imports it).

### What We're NOT Building (Scope Boundaries)

* ❌ Full Compiler-Level CFG - Too heavy for real-time
* ❌ Cross-language type inference - Compiler-level complexity
* ❌ Real-time collaborative editing - Not the core use case
* ❌ Cloud deployment - Local-first by design

---

## TECHNICAL ARCHITECTURE

### The Hybrid Stack

#### Rust Core ("The Sensory System")

**Purpose:** High-performance parsing, symbol extraction, and graph construction

**Dependencies:**

```toml
[dependencies]
tree-sitter = "0.20"
tree-sitter-python = "0.20"
tree-sitter-rust = "0.20"
petgraph = "0.6"           # Graph algorithms
ignore = "0.4"             # Fast .gitignore-aware file walking
rayon = "1.7"              # Parallel iteration
serde = { version = "1.0", features = ["derive"] }
pyo3 = { version = "0.20", features = ["extension-module"] }

```

**Core Data Structures:**

```rust
// The normalized representation of a code file
pub struct CodeSkeleton {
    pub path: PathBuf,
    pub language: Language,
    pub definitions: Vec<Definition>,
    pub token_count: usize,
}

// Symbol-Level Indexing (The "Stack Graph Lite")
pub struct SymbolIndex {
    // Map "User" -> ["src/models/user.py"]
    // Handles collisions by tracking all potential definitions
    pub definitions: HashMap<String, Vec<PathBuf>>, 
    
    // Map "src/auth.py" -> ["User", "hash_password"]
    // Symbols explicitly USED in this file (extracted via Tree-sitter)
    pub usages: HashMap<PathBuf, Vec<String>>,
}

pub struct RepoGraph {
    pub graph: DiGraph<FileNode, EdgeKind>,
    pub index: SymbolIndex,
    pub last_updated: SystemTime,
}

```

#### Python Orchestration ("The Agent Brain")

**Purpose:** LLM integration, vector embeddings, tool execution

**Dependencies:**

```python
# Core
pydantic >= 2.0        # Structured data validation
rich >= 13.0           # Terminal UI
tiktoken >= 0.5        # Token counting

# Intelligence & Vector Support
ollama >= 0.1          # Local model serving
fastembed >= 0.2       # Local vector embeddings (NPU optimized)
numpy >= 1.24          # Math ops for embeddings

# Graph Queries
networkx >= 3.0        # Optional: additional graph algorithms

```

### The Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PERCEPTION (Rust)                                        │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│    │  Scan    │───▶│  Parse   │───▶│ Harvest  │           │
│    │ (.git-   │    │ (Tree-   │    │ Symbols  │           │
│    │  ignore) │    │ sitter)  │    │ (Defs/Use│           │
│    └──────────┘    └──────────┘    └──────────┘           │
│                                          │                  │
│                                          ▼                  │
│                                    ┌──────────┐            │
│                                    │ SymGraph │            │
│                                    │ (Usage   │            │
│                                    │  Edges)  │            │
│                                    └──────────┘            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (PyO3 Bridge)
┌─────────────────────────────────────────────────────────────┐
│ 2. CONTEXT SYNTHESIS (Python)                               │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│    │ Vector   │───▶│ Anchor & │───▶│ Assemble │           │
│    │ Search   │    │ Expand   │    │  Prompt  │           │
│    │          │    │          │    │          │           │
│    └──────────┘    └──────────┘    └──────────┘           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. REASONING (DeepSeek R1)                                  │
│    ┌──────────────────────────────────────────┐            │
│    │  <think>                                 │            │
│    │    1. Analyze architecture from map      │            │
│    │    2. Identify relevant files            │            │
│    │    3. Plan modification strategy         │            │
│    │  </think>                                │            │
│    │                                          │            │
│    │  <action>                                │            │
│    │    write_file("src/auth.py", code)       │            │
│    │  </action>                               │            │
│    └──────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. REFLEXIVE ACTION (Rust + Python)                         │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│    │ Syntax   │───▶│ Execute  │───▶│  Update  │           │
│    │ Check    │    │  Tool    │    │  Graph   │──┐        │
│    └──────────┘    └──────────┘    └──────────┘  │        │
│                                          ▲         │        │
│                                          └─────────┘        │
│                                           Loop Back         │
└─────────────────────────────────────────────────────────────┘

```

---

## PHASE 1: FOUNDATION & SCAFFOLDING

**Duration:** 1-2 weeks

**Goal:** Prove the hybrid architecture works and establish performance baselines

### 1.1 Project Scaffolding

**Deliverable:** Working Rust/Python hybrid project structure

**Directory Structure:**

```
atlas/
├── rust_core/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs              # PyO3 entry point
│   │   ├── symbol_table.rs     # NEW: Symbol storage
│   │   ├── parser.rs           # Tree-sitter integration
│   │   └── graph.rs            # Dependency graph
│   └── tests/
├── python_shell/
│   ├── atlas/
│   │   ├── __init__.py
│   │   ├── agent.py            # Main agent loop
│   │   ├── embeddings.py       # NEW: FastEmbed wrapper
│   │   └── tools.py            # File operations
├── pyproject.toml
└── maturin.toml

```

**Implementation Steps:**

1. Initialize Rust library: `cargo init --lib rust_core`
2. Add PyO3 and `ort` (optional, if doing embeddings in Rust later)
3. Set up Python package with `fastembed` dependency
4. Configure development environment

**Success Criteria:**

* [x] `maturin develop` builds Rust module successfully
* [x] Python can import `semantic_engine`
* [x] Tests pass on both Rust and Python sides

### 1.2 Performance Baseline: Fast File Scanner

**Deliverable:** Rust-powered file scanner that's 50-100x faster than Python's `os.walk`

**Rust Implementation:**
```rust
use ignore::WalkBuilder;
use pyo3::prelude::*;

#[pyfunction]
fn scan_repository(path: &str) -> PyResult<Vec<String>> {
    let mut files = Vec::new();
    
    for result in WalkBuilder::new(path)
        .hidden(false)           // Don't skip hidden files
        .git_ignore(true)        // Respect .gitignore
        .build()
    {
        match result {
            Ok(entry) if entry.path().is_file() => {
                if let Some(path_str) = entry.path().to_str() {
                    files.push(path_str.to_string());
                }
            }
            _ => {}
        }
    }
    
    Ok(files)
}
```

**Benchmark Test:**
```python
# Test on a real repository (e.g., clone a popular GitHub repo)
import time
import os
from semantic_engine import scan_repository

# Python baseline
start = time.time()
py_files = []
for root, dirs, files in os.walk("./test_repo"):
    for file in files:
        py_files.append(os.path.join(root, file))
py_time = time.time() - start

# Rust implementation
start = time.time()
rust_files = scan_repository("./test_repo")
rust_time = time.time() - start

print(f"Python: {py_time:.3f}s for {len(py_files)} files")
print(f"Rust: {rust_time:.3f}s for {len(rust_files)} files")
print(f"Speedup: {py_time/rust_time:.1f}x")
```

**Success Criteria:**
- [x] Scans 10,000 file repository in < 100ms
- [x] Respects .gitignore correctly (no node_modules, .git, etc.)
- [x] At least 5x faster than Python's os.walk
- [x] Memory usage < 100MB for 10k files

### 1.3 Language Detection & Parser Loading

**Deliverable:** Automatic language detection and Tree-sitter parser initialization

**Rust Implementation:**
```rust
use std::collections::HashMap;
use tree_sitter::{Language, Parser};

pub enum SupportedLanguage {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    Go,
    Unknown,
}

impl SupportedLanguage {
    pub fn from_extension(ext: &str) -> Self {
        match ext {
            "py" => Self::Python,
            "rs" => Self::Rust,
            "js" | "jsx" => Self::JavaScript,
            "ts" | "tsx" => Self::TypeScript,
            "go" => Self::Go,
            _ => Self::Unknown,
        }
    }
    
    pub fn get_parser(&self) -> Option<Language> {
        match self {
            Self::Python => Some(tree_sitter_python::language()),
            Self::Rust => Some(tree_sitter_rust::language()),
            Self::JavaScript => Some(tree_sitter_javascript::language()),
            Self::TypeScript => Some(tree_sitter_typescript::language_typescript()),
            Self::Go => Some(tree_sitter_go::language()),
            Self::Unknown => None,
        }
    }
}

pub struct ParserPool {
    parsers: HashMap<SupportedLanguage, Parser>,
}

impl ParserPool {
    pub fn new() -> Self {
        let mut pool = Self {
            parsers: HashMap::new(),
        };
        
        // Pre-initialize parsers for each language
        for lang in [
            SupportedLanguage::Python,
            SupportedLanguage::Rust,
            // ... etc
        ] {
            if let Some(ts_lang) = lang.get_parser() {
                let mut parser = Parser::new();
                parser.set_language(ts_lang).unwrap();
                pool.parsers.insert(lang, parser);
            }
        }
        
        pool
    }
    
    pub fn get(&mut self, lang: SupportedLanguage) -> Option<&mut Parser> {
        self.parsers.get_mut(&lang)
    }
}
```

**Success Criteria:**
- [x] Correctly identifies language for 100% of common extensions
- [x] Parser initialization time < 1ms per language
- [x] Handles binary/unknown files gracefully (skips them)

---

## PHASE 2: THE NORMALIZATION LAYER

**Duration:** 2-3 weeks

**Goal:** Convert verbose CSTs into compact, semantic skeletons and harvest symbols

### 2.1 CST Pruning (The Noise Filter)

**Deliverable:** Reduce raw Tree-sitter output by 50-70% while preserving semantics

**Concept:** Tree-sitter produces nodes like:
```
module
  expression_statement
    assignment
      identifier "x"
      "="
      number "5"
```

We want:
```
assignment
  identifier "x"
  number "5"
```

**Implementation:**
```rust
use tree_sitter::{Node, TreeCursor};

// Define which node types are "trivial" per language. This is crucial
// for the normalization process to work across all supported languages.
const TRIVIAL_PYTHON_NODES: &[&str] = &[
    "module", "expression_statement", "block",
    "(", ")", "[", "]", "{", "}", ":", ",",
];
const TRIVIAL_RUST_NODES: &[&str] = &[
    "token_tree", "source_file", ";", "{", "}",
];
const TRIVIAL_JS_TS_NODES: &[&str] = &[
    "program", "expression_statement", ";", "{", "}", "(", ")",
];
const TRIVIAL_GO_NODES: &[&str] = &[
    "source_file", "expression_statement",
];

// NOTE: The `normalize_node` function will need to be updated to select
// the correct list based on the language being parsed.

pub struct NormalizedNode {
    pub kind: String,
    pub text: Option<String>,
    pub children: Vec<NormalizedNode>,
    pub byte_range: (usize, usize),
}

pub fn normalize_tree(root: Node, source: &[u8]) -> NormalizedNode {
    let mut cursor = root.walk();
    normalize_node(&mut cursor, source, 0)
}

fn normalize_node(
    cursor: &mut TreeCursor,
    source: &[u8],
    depth: usize
) -> NormalizedNode {
    let node = cursor.node();
    let kind = node.kind();
    
    // Skip trivial nodes by promoting their children
    if TRIVIAL_PYTHON_NODES.contains(&kind) && node.child_count() == 1 {
        cursor.goto_first_child();
        let child = normalize_node(cursor, source, depth);
        cursor.goto_parent();
        return child;
    }
    
    // For significant nodes, process normally
    let text = if node.child_count() == 0 {
        Some(node.utf8_text(source).unwrap().to_string())
    } else {
        None
    };
    
    let mut children = Vec::new();
    if cursor.goto_first_child() {
        loop {
            children.push(normalize_node(cursor, source, depth + 1));
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
    
    NormalizedNode {
        kind: kind.to_string(),
        text,
        children,
        byte_range: (node.start_byte(), node.end_byte()),
    }
}
```

**Test Cases:**
```rust
#[test]
fn test_normalization_reduces_nodes() {
    let source = "x = 5";
    let tree = parse_python(source);
    let normalized = normalize_tree(tree.root_node(), source.as_bytes());
    
    // Raw tree has ~7 nodes, normalized should have ~3
    assert!(count_nodes(&normalized) < 5);
}
```

**Success Criteria:**
- [ ] Reduces node count by 50-70% for Python
- [ ] Preserves all identifier names and literals
- [ ] Processing time < 5ms per 1000 lines of code
- [ ] No loss of semantic information (validated by manual inspection)

### 2.2 Signature Extraction (The Skeletonizer)

**Deliverable:** Extract function/class signatures while eliding implementation bodies

**Tree-sitter Query Definitions:**

Create `queries/python/tags.scm`:
```scheme
; Function definitions
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (type)? @function.return
  body: (block) @function.body
  (#set! "role" "definition"))

; Capture docstrings (string immediately after def)
(function_definition
  body: (block
    (expression_statement
      (string) @function.doc)))

; Class definitions
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block)
  ; By not capturing `@class.body`, the linear skeletonizer will recurse
  ; into the class, find the methods, and elide their bodies individually.
  (#set! "role" "definition"))

; Import statements
(import_statement
  name: (dotted_name) @import.module)

(import_from_statement
  module_name: (dotted_name) @import.module
  name: (dotted_name) @import.symbol)
```

**Rust Skeletonization Logic:**
```rust
use tree_sitter::{Query, QueryCursor};

pub struct FunctionSignature {
    pub name: String,
    pub params: String,
    pub return_type: Option<String>,
    pub docstring: Option<String>,
    pub body_byte_range: (usize, usize),
}

pub fn extract_signatures(
    tree: &Tree,
    source: &str,
    language: &Language
) -> Vec<FunctionSignature> {
    let query_source = include_str!("../queries/python/tags.scm");
    let query = Query::new(language, query_source).unwrap();
    
    let mut cursor = QueryCursor::new();
    let matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    
    let mut signatures = Vec::new();
    
    for m in matches {
        let mut sig = FunctionSignature::default();
        
        for capture in m.captures {
            let text = capture.node.utf8_text(source.as_bytes()).unwrap();
            
            match query.capture_names()[capture.index as usize].as_str() {
                "function.name" => sig.name = text.to_string(),
                "function.params" => sig.params = text.to_string(),
                "function.return" => sig.return_type = Some(text.to_string()),
                "function.doc" => sig.docstring = Some(text.to_string()),
                "function.body" => {
                    sig.body_byte_range = (
                        capture.node.start_byte(),
                        capture.node.end_byte()
                    );
                }
                _ => {}
            }
        }
        
        signatures.push(sig);
    }
    
    signatures
}

pub fn create_skeleton(source: &str, signatures: &[FunctionSignature]) -> String {
    let mut skeleton = String::new();
    let mut last_end = 0;
    
    for sig in signatures {
        // Keep everything before the function body
        skeleton.push_str(&source[last_end..sig.body_byte_range.0]);
        
        // Replace body with ellipsis
        skeleton.push_str("\n    ...\n");
        
        last_end = sig.body_byte_range.1;
    }
    
    // Append remaining source
    skeleton.push_str(&source[last_end..]);
    
    skeleton
}
```

**Example Input/Output:**
```python
# INPUT (72 tokens)
def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticates a user against the database.
    Returns User object if valid, None otherwise.
    """
    hashed = hash_password(password)
    user = db.query(User).filter_by(username=username).first()
    if user and user.password_hash == hashed:
        return user
    return None

# OUTPUT SKELETON (23 tokens - 68% reduction)
def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticates a user against the database.
    Returns User object if valid, None otherwise.
    """
    ...
```

**Success Criteria:**
- [ ] Reduces token count by 70-85% for typical Python files
- [ ] Preserves ALL type signatures
- [ ] Preserves ALL docstrings
- [ ] Preserves imports
- [ ] Processing time < 10ms per file
- [ ] Skeleton is still syntactically valid Python (can be parsed)

### 2.3 Symbol Harvesting (The "Stack Graph Lite")

**Deliverable:** Extract Defined Symbols vs. Used Symbols

**Tree-sitter Query Definitions (Updated):**
Create `queries/python/symbols.scm`:

```scheme
; Definitions
(function_definition name: (identifier) @def.function)
(class_definition name: (identifier) @def.class)
(assignment left: (identifier) @def.variable)
(annotated_assignment left: (identifier) @def.variable)
; For tuple unpacking
(assignment left: (pattern_list (identifier) @def.variable))
(assignment left: (tuple_pattern (identifier) @def.variable))

; Usages (Call sites)
(call function: (identifier) @usage.call)
(call function: (attribute attribute: (identifier) @usage.method))

```

**Rust Implementation:**

```rust
pub struct SymbolHarvester;

impl SymbolHarvester {
    pub fn harvest(tree: &Tree, source: &str) -> (Vec<String>, Vec<String>) {
        let mut definitions = Vec::new();
        let mut usages = Vec::new();
        
        // Execute queries to split "What I make" vs "What I need"
        // ... implementation using QueryCursor ...
        
        (definitions, usages)
    }
}

```

---

## PHASE 3: THE REPOSITORY GRAPH (REVISED)

**Duration:** 2-3 weeks

**Goal:** Build a "Symbol-Aware" map, not just a file dependency tree.

### 3.1 Import Resolution (Building the Edges)

**Deliverable:** Accurate file-to-file dependency graph

**Challenge:** Resolve import statements to actual files

**Python Example:**
```python
# In file: src/app.py
from .auth import authenticate_user        # Relative import
from utils.logging import setup_logger     # Absolute import
import os                                  # Standard library (ignore)
```

Must resolve to:
- `src/auth.py` ✓
- `src/utils/logging.py` ✓
- (skip stdlib)

**Implementation:**
```rust
use std::path::{Path, PathBuf};

pub struct ImportResolver {
    project_root: PathBuf,
    // Map of module names to file paths
    module_index: HashMap<String, PathBuf>,
}

impl ImportResolver {
    pub fn new(project_root: PathBuf) -> Self {
        let mut resolver = Self {
            project_root: project_root.clone(),
            module_index: HashMap::new(),
        };
        
        // Build index of all Python files
        resolver.index_modules();
        resolver
    }
    
    fn index_modules(&mut self) {
        // Walk all .py files and create module -> path mapping
        // e.g., "src.auth" -> "src/auth.py"
    }
    
    pub fn resolve_import(
        &self,
        current_file: &Path,
        import_statement: &str
    ) -> Option<PathBuf> {
        if import_statement.starts_with('.') {
            // Relative import
            self.resolve_relative(current_file, import_statement)
        } else {
            // Absolute import
            self.resolve_absolute(import_statement)
        }
    }
    
    fn resolve_relative(&self, current_file: &Path, import: &str) -> Option<PathBuf> {
        let dots = import.chars().take_while(|&c| c == '.').count();
        let module_path = &import[dots..];
        
        // Go up 'dots' directories from current_file
        let mut base = current_file.parent()?.to_path_buf();
        for _ in 1..dots {
            base = base.parent()?.to_path_buf();
        }
        
        // Append module path
        let full_path = base.join(module_path.replace('.', "/")).with_extension("py");
        
        if full_path.exists() {
            Some(full_path)
        } else {
            None
        }
    }
    
    fn resolve_absolute(&self, import: &str) -> Option<PathBuf> {
        // Check if it's in our module index
        self.module_index.get(import).cloned()
    }
}
```

**Edge Cases to Handle:**
- [ ] Relative imports (., .., ...)
- [ ] Package imports (`from package import module`)
- [ ] Star imports (`from package import *`) - mark as "weak edge"
- [ ] Import aliases (`import numpy as np`)
- [ ] Standard library (skip)
- [ ] Third-party packages (optionally skip or mark differently)

**Success Criteria:**
- [X] Resolves 95%+ of imports in real repositories
- [X] False positive rate < 5% (doesn't hallucinate dependencies)
- [X] Handles all Python import styles
- [X] Processing time < 100ms for 1000 files

### 3.2 Symbol-Based Graph Construction

**Deliverable:** Weighted directed graph based on *symbol usage*.

**Concept:** Instead of trusting `import` statements (which can be unused or vague), we link files based on actual code usage. If `main.py` calls `User()`, it depends on `user.py`, regardless of how it imported it.

**Implementation:**

```rust
use std::collections::HashMap;
use petgraph::prelude::*;

pub enum EdgeKind {
    Import,         // Weak link (explicit import)
    SymbolUsage,    // Strong link (function call/class instantiation)
}

pub struct RepoGraph {
    graph: DiGraph<PathBuf, EdgeKind>,
    symbol_index: SymbolIndex,
}

impl RepoGraph {
    pub fn build_semantic_edges(&mut self) {
        // Iterate over every file's usages
        for (file_path, used_symbols) in &self.symbol_index.usages {
            let source_node = self.get_node_index(file_path);
            
            for symbol in used_symbols {
                // Find where this symbol is defined
                if let Some(def_paths) = self.symbol_index.definitions.get(symbol) {
                    for def_path in def_paths {
                        // Don't link file to itself
                        if def_path != file_path {
                            let target_node = self.get_node_index(def_path);
                            
                            // Create a STRONG semantic edge (Weight 2.0)
                            // This indicates actual logical dependency
                            self.graph.add_edge(
                                source_node, 
                                target_node, 
                                EdgeKind::SymbolUsage
                            );
                        }
                    }
                }
            }
        }
    }
}

```

**Success Criteria:**

* [ ] Distinguishes between "Imported but unused" (Weak) and "Called" (Strong)
* [ ] Handles multiple definitions (collisions) via heuristic weighting
* [ ] Graph construction time < 2 seconds for 10k files

### 3.3 PageRank Implementation

**Deliverable:** Rank files by architectural importance

**The Algorithm:**
PageRank assigns scores based on:
- **Incoming edges** (how many files import this file) → High rank
- **Quality of incoming edges** (being imported by a high-rank file) → Higher rank

**Implementation:**
```rust
use petgraph::visit::EdgeRef;

impl RepoGraph {
    pub fn calculate_pagerank(&mut self, iterations: usize, damping: f64) {
        let node_count = self.graph.node_count();
        if node_count == 0 {
            return;
        }
        
        // Initialize all ranks to 1/N
        let initial_rank = 1.0 / node_count as f64;
        for node in self.graph.node_weights_mut() {
            node.rank = initial_rank;
        }
        
        // Iterative PageRank
        for _ in 0..iterations {
            let mut new_ranks = vec![0.0; node_count];
            
            for idx in self.graph.node_indices() {
                let mut rank_sum = 0.0;
                
                // Sum contributions from incoming edges
                for edge in self.graph.edges_directed(idx, Direction::Incoming) {
                    let source_idx = edge.source();
                    let source_rank = self.graph[source_idx].rank;
                    let out_degree = self.graph
                        .edges_directed(source_idx, Direction::Outgoing)
                        .count() as f64;
                    
                    let edge_weight = edge.weight().strength as f64;
                    rank_sum += (source_rank / out_degree) * edge_weight;
                }
                
                // Apply damping factor
                new_ranks[idx.index()] = 
                    (1.0 - damping) / node_count as f64 + damping * rank_sum;
            }
            
            // Update ranks
            for (idx, &new_rank) in new_ranks.iter().enumerate() {
                self.graph[NodeIndex::new(idx)].rank = new_rank;
            }
        }
    }
    
    pub fn get_top_ranked_files(&self, limit: usize) -> Vec<(PathBuf, f64)> {
        let mut ranked: Vec<_> = self.graph
            .node_weights()
            .map(|n| (n.path.clone(), n.rank))
            .collect();
        
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(limit);
        ranked
    }
}
```

**Validation:**
```rust
#[test]
fn test_pagerank_identifies_core_files() {
    // Create a graph where 'core.py' is imported by many files
    let mut graph = RepoGraph::new();
    
    let core = graph.add_file("src/core.py", SupportedLanguage::Python);
    let util1 = graph.add_file("src/util1.py", SupportedLanguage::Python);
    let util2 = graph.add_file("src/util2.py", SupportedLanguage::Python);
    let app = graph.add_file("src/app.py", SupportedLanguage::Python);
    
    // Everyone imports core
    graph.add_edge(util1, core);
    graph.add_edge(util2, core);
    graph.add_edge(app, core);
    
    graph.calculate_pagerank(20, 0.85);
    
    let top = graph.get_top_ranked_files(1);
    assert_eq!(top[0].0, PathBuf::from("src/core.py"));
}
```

**Success Criteria:**
- [ ] Core utility files rank higher than leaf files
- [ ] Entry points (main.py) rank high due to exports
- [ ] Convergence in < 20 iterations
- [ ] Calculation time < 100ms for 10k files

### 3.4 Repository Map Serialization

**Deliverable:** Compact text representation for LLM context

**Format Design:**
```
Repository: my_project (486 files, 52k LOC)
Architecture Score: 0.847 (well-structured)

TOP RANKED FILES (by architectural importance):
  1. src/core/database.py         [rank: 0.156, imported by: 43 files]
  2. src/auth/session.py          [rank: 0.134, imported by: 28 files]
  3. src/models/user.py           [rank: 0.098, imported by: 31 files]
  4. src/api/routes.py            [rank: 0.087, imported by: 12 files]
  5. src/utils/logging.py         [rank: 0.076, imported by: 38 files]

DIRECTORY STRUCTURE:
my_project/
├── src/
│   ├── core/
│   │   ├── database.py         ★★★★★ (rank: 0.156)
│   │   └── config.py           ★★ (rank: 0.034)
│   ├── auth/
│   │   ├── session.py          ★★★★★ (rank: 0.134)
│   │   └── permissions.py      ★★★ (rank: 0.045)
│   ├── models/
│   │   ├── user.py             ★★★★ (rank: 0.098)
│   │   └── product.py          ★★ (rank: 0.032)
│   └── api/
│       └── routes.py           ★★★★ (rank: 0.087)
└── tests/

IMPORT CLUSTERS (detected communities):
  Cluster 1: Authentication System
    - auth/session.py
    - auth/permissions.py
    - models/user.py
    
  Cluster 2: Data Layer
    - core/database.py
    - models/user.py
    - models/product.py
```

**Token Efficiency:**
- Full file listing: ~5000 tokens
- This map: ~400 tokens
- **Compression: 12.5x**

**Implementation:**
```rust
impl RepoGraph {
    pub fn generate_map(&self, max_files: usize) -> String {
        let mut output = String::new();
        
        // Header stats
        let total_files = self.graph.node_count();
        output.push_str(&format!("Repository Map ({} files)\n\n", total_files));
        
        // Top ranked files
        output.push_str("TOP RANKED FILES:\n");
        for (i, (path, rank)) in self.get_top_ranked_files(max_files).iter().enumerate() {
            let dependents = self.get_dependents(path).len();
            output.push_str(&format!(
                "  {}. {} [rank: {:.3}, imported by: {} files]\n",
                i + 1,
                path.display(),
                rank,
                dependents
            ));
        }
        
        output
    }
}
```

**Success Criteria:**
- [ ] Map generation time < 10ms
- [ ] Token count < 500 for typical repositories
- [ ] Contains all architecturally important files
- [ ] Human-readable and LLM-parseable

---

## PHASE 4: INCREMENTAL UPDATES
**Duration:** 2 weeks  
**Goal:** Keep graph synchronized with code changes in real-time

### 4.1 File Change Detection

**Deliverable:** Detect when files are modified and trigger updates

**Challenge:** We need to know:
1. Which files changed
2. What changed in those files (for incremental parse)
3. What downstream files might be affected

**Implementation:**
```rust
use std::time::SystemTime;

pub struct FileWatcher {
    // Map of file paths to their last modification time
    file_mtimes: HashMap<PathBuf, SystemTime>,
}

impl FileWatcher {
    pub fn scan_for_changes(&mut self, files: &[PathBuf]) -> Vec<FileChange> {
        let mut changes = Vec::new();
        
        for path in files {
            if let Ok(metadata) = std::fs::metadata(path) {
                if let Ok(mtime) = metadata.modified() {
                    match self.file_mtimes.get(path) {
                        Some(&old_mtime) if mtime > old_mtime => {
                            changes.push(FileChange::Modified(path.clone()));
                            self.file_mtimes.insert(path.clone(), mtime);
                        }
                        None => {
                            changes.push(FileChange::Added(path.clone()));
                            self.file_mtimes.insert(path.clone(), mtime);
                        }
                        _ => {}
                    }
                }
            } else if self.file_mtimes.contains_key(path) {
                changes.push(FileChange::Deleted(path.clone()));
                self.file_mtimes.remove(path);
            }
        }
        
        changes
    }
}

pub enum FileChange {
    Added(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
}
```

### 4.2 Tree-sitter Incremental Parsing

**Deliverable:** Update syntax trees in microseconds instead of milliseconds

**The Tree-sitter API:**
```rust
use tree_sitter::{Parser, Tree, InputEdit};

pub struct IncrementalParser {
    parser: Parser,
    // Cache of parsed trees
    trees: HashMap<PathBuf, Tree>,
    // Original source code
    sources: HashMap<PathBuf, String>,
}

impl IncrementalParser {
    pub fn update_file(
        &mut self,
        path: &Path,
        new_source: String,
        edit: TextEdit
    ) -> Result<Tree, ParseError> {
        // Get existing tree or parse from scratch
        let old_tree = self.trees.get(path);
        let old_source = self.sources.get(path).ok_or(ParseError::NoCache)?;
        
        if let Some(tree) = old_tree {
            // Apply edit to the tree
            let input_edit = self.text_edit_to_input_edit(&edit, old_source);
            let mut new_tree = tree.clone();
            new_tree.edit(&input_edit);
            
            // Re-parse with the edit applied
            let parsed = self.parser
                .parse(&new_source, Some(&new_tree))
                .ok_or(ParseError::Failed)?;
            
            // Cache the new tree and source
            self.trees.insert(path.to_path_buf(), parsed.clone());
            self.sources.insert(path.to_path_buf(), new_source);
            
            Ok(parsed)
        } else {
            // No cached tree, parse from scratch
            self.parse_from_scratch(path, new_source)
        }
    }
    
    fn text_edit_to_input_edit(&self, edit: &TextEdit, source: &str) -> InputEdit {
        // Convert line/column edit to byte offsets
        let start_byte = self.position_to_byte(source, edit.start_line, edit.start_col);
        let old_end_byte = start_byte + edit.old_text.len();
        let new_end_byte = start_byte + edit.new_text.len();
        
        InputEdit {
            start_byte,
            old_end_byte,
            new_end_byte,
            start_position: Point::new(edit.start_line, edit.start_col),
            old_end_position: Point::new(edit.end_line, edit.end_col),
            new_end_position: self.calculate_new_end_position(edit),
        }
    }
}

pub struct TextEdit {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub old_text: String,
    pub new_text: String,
}
```

**Performance Benchmark:**
```rust
#[bench]
fn bench_incremental_vs_full_parse(b: &mut Bencher) {
    let large_file = include_str!("../test_data/large_file.py"); // 5000 lines
    
    b.iter(|| {
        // Full parse
        let start = Instant::now();
        parser.parse(large_file, None);
        let full_time = start.elapsed();
        
        // Incremental parse (change one line)
        let edit = TextEdit { /* change line 100 */ };
        let start = Instant::now();
        parser.update_file(&path, new_source, edit);
        let inc_time = start.elapsed();
        
        assert!(inc_time < full_time / 10); // Should be 10x faster
    });
}
```

**Success Criteria:**
- [ ] Incremental parse 10-100x faster than full parse
- [ ] Incremental parse time < 5ms for typical edits
- [ ] Correctly handles multi-line edits
- [ ] Memory usage doesn't grow unbounded (cache eviction)

### 4.3 Graph Update Propagation

**Deliverable:** Update dependency graph when imports change

**The Update Strategy:**

When a file is modified:
1. **Re-parse the file** (incremental)
2. **Re-extract imports** from the new tree
3. **Compare old vs new imports**
4. **Update graph edges** (add new, remove deleted)
5. **Recalculate PageRank** (but only if structure changed significantly)

**Implementation:**
```rust
impl RepoGraph {
    pub fn update_file_imports(
        &mut self,
        path: &Path,
        new_imports: Vec<Import>
    ) -> UpdateResult {
        let idx = self.path_to_idx.get(path).ok_or(GraphError::NodeNotFound)?;
        
        // Get old imports (current outgoing edges)
        let old_edges: Vec<_> = self.graph
            .edges_directed(*idx, Direction::Outgoing)
            .map(|e| (e.target(), e.weight().clone()))
            .collect();
        
        // Remove all old outgoing edges
        self.graph.retain_edges(|g, e| {
            let (source, _) = g.edge_endpoints(e).unwrap();
            source != *idx
        });
        
        // Add new edges
        for import in new_imports {
            if let Some(target_path) = import.resolved_path {
                if let Some(&target_idx) = self.path_to_idx.get(&target_path) {
                    self.graph.add_edge(*idx, target_idx, ImportEdge {
                        kind: EdgeKind::DirectImport,
                        strength: 1.0,
                    });
                }
            }
        }
        
        // Determine if significant change occurred
        let new_edge_count = self.graph
            .edges_directed(*idx, Direction::Outgoing)
            .count();
        
        let significant_change = new_edge_count != old_edges.len();
        
        UpdateResult {
            edges_added: new_edge_count.saturating_sub(old_edges.len()),
            edges_removed: old_edges.len().saturating_sub(new_edge_count),
            needs_pagerank_recalc: significant_change,
        }
    }
}

pub struct UpdateResult {
    pub edges_added: usize,
    pub edges_removed: usize,
    pub needs_pagerank_recalc: bool,
}
```

**Optimization: Lazy PageRank:**
```rust
impl RepoGraph {
    // Don't recalculate immediately, mark as dirty
    pub fn mark_dirty(&mut self) {
        self.pagerank_dirty = true;
    }
    
    // Only recalculate when actually needed (query time)
    pub fn get_top_ranked_files(&mut self, limit: usize) -> Vec<(PathBuf, f64)> {
        if self.pagerank_dirty {
            self.calculate_pagerank(20, 0.85);
            self.pagerank_dirty = false;
        }
        
        // ... rest of implementation
    }
}
```

**Success Criteria:**
- [ ] Graph update time < 10ms per file change
- [ ] PageRank recalculation only when necessary
- [ ] Correctly handles import additions/deletions
- [ ] No memory leaks over 1000s of updates

### 4.4 Multi-Tier Update Strategy

**Deliverable:** Optimize update performance based on change magnitude

**The Tier System:**

```rust
pub enum UpdateTier {
    // Single variable renamed within one function
    Tier1_Local,        // Update time: < 1ms
    
    // Function signature changed
    Tier2_FileScope,    // Update time: < 10ms
    
    // Import added/removed
    Tier3_GraphScope,   // Update time: < 100ms
    
    // Major refactor (many files)
    Tier4_FullRebuild,  // Update time: < 5s
}

impl RepoGraph {
    pub fn classify_update(&self, change: &FileChange) -> UpdateTier {
        match change {
            FileChange::Modified(path) => {
                let old_imports = self.get_file_imports(path);
                let new_imports = extract_imports(path);
                
                if old_imports == new_imports {
                    // Imports unchanged, local edit
                    UpdateTier::Tier1_Local
                } else if new_imports.len().abs_diff(old_imports.len()) < 3 {
                    // Minor import changes
                    UpdateTier::Tier2_FileScope
                } else {
                    // Major import changes
                    UpdateTier::Tier3_GraphScope
                }
            }
            FileChange::Added(_) => UpdateTier::Tier3_GraphScope,
            FileChange::Deleted(_) => UpdateTier::Tier3_GraphScope,
        }
    }
    
    pub fn perform_update(&mut self, change: FileChange) {
        let tier = self.classify_update(&change);
        
        match tier {
            UpdateTier::Tier1_Local => {
                // Just update the skeleton, graph unchanged
                self.update_skeleton_only(&change);
            }
            UpdateTier::Tier2_FileScope => {
                // Update skeleton + check for broken references
                self.update_file_scope(&change);
            }
            UpdateTier::Tier3_GraphScope => {
                // Update graph edges + recalc PageRank
                self.update_graph_scope(&change);
            }
            UpdateTier::Tier4_FullRebuild => {
                // Rebuild entire graph
                self.rebuild_from_scratch();
            }
        }
    }
}
```

**Success Criteria:**
- [ ] 90% of edits classified as Tier1 or Tier2
- [ ] Tier1 updates complete in < 1ms
- [ ] Tier2 updates complete in < 10ms
- [ ] Tier3 updates complete in < 100ms
- [ ] User never waits more than 100ms for agent to "see" changes

---

## PHASE 5: CONTEXT ASSEMBLY & AGENT INTEGRATION (REVISED)

**Duration:** 2-3 weeks

**Goal:** Connect Intelligence (Vectors) with Structure (Graph)

### 5.1 Token Budget Manager

**Deliverable:** Intelligent context window allocation

**The Budget Equation:**
```
Total Context (128k) = System Prompt + User Query + Context Files + Reserved Response
                128k = ~2k          + variable     + variable      + 4k

Available for Context = 128k - 2k - query_tokens - 4k
```

**Implementation:**
```python
from dataclasses import dataclass
from typing import List
import tiktoken

@dataclass
class ContextBudget:
    total_tokens: int = 128_000
    system_prompt: str = ""
    user_query: str = ""
    reserved_for_response: int = 4096
    
    def __post_init__(self):
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.system_tokens = len(self.encoder.encode(self.system_prompt))
        self.query_tokens = len(self.encoder.encode(self.user_query))
    
    @property
    def available(self) -> int:
        return (
            self.total_tokens 
            - self.system_tokens 
            - self.query_tokens 
            - self.reserved_for_response
        )
    
    def can_fit(self, text: str) -> bool:
        tokens = len(self.encoder.encode(text))
        return tokens <= self.available
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
```

### 5.2 Hybrid "Anchor & Expand" Context Strategy

**Deliverable:** Intelligent context selection that uses Vectors to find the *topic* and Graph to find the *dependencies*.

**The Flaw in PageRank:** Static ranking ignores user intent. "Fix login" needs `auth.py` even if `utils.py` has a higher PageRank.

**The Solution:**

1. **Anchor:** Use Vector Search to find the entry point.
2. **Expand:** Use Symbol Graph to traverse 1-2 hops out.

**Implementation:**

```python
from semantic_engine import RepoGraph
from fastembed import TextEmbedding
from typing import Set

class ContextAssembler:
    def __init__(self, graph: RepoGraph, budget: ContextBudget):
        self.graph = graph
        self.budget = budget
        # Fast, local embedding model (runs on CPU/NPU)
        self.embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.vector_db = VectorDB() # Simple in-memory FAISS or similar

    def assemble_smart_context(self, query: str) -> ContextAssembly:
        # 1. SEMANTIC ANCHOR (Vector Search)
        # Use embeddings to find the "concept" the user is talking about.
        # e.g., "Fix login" -> hits `src/auth/auth_controller.py`
        anchors = self.vector_db.search(query, top_k=3)
        
        # 2. STRUCTURAL EXPANSION (Graph Search)
        # Don't just grab the file. Grab the "Neighborhood".
        neighborhood = set(anchors)
        for anchor in anchors:
            # Incoming: Who relies on this? (Risk of breaking changes)
            dependents = self.graph.get_dependents(anchor)
            neighborhood.update(dependents)
            
            # Outgoing: What does this rely on? (Required context to understand code)
            dependencies = self.graph.get_dependencies(anchor)
            neighborhood.update(dependencies)
            
        # 3. FILL BUDGET
        # Prioritize: Neighborhood (Full Code) > Global PageRank (Skeletons)
        return self.fill_budget(neighborhood, self.graph.get_top_ranked_files())

    def fill_budget(self, neighborhood: Set[str], global_ranks: List[str]):
        # Logic to fill 'focus_files' and 'skeleton_files' until tokens exhausted
        pass

```

### 5.3 Prompt Construction

**Deliverable:** The final prompt that goes to DeepSeek

**System Prompt Design:**
```python
SYSTEM_PROMPT = """You are an autonomous coding agent with deep structural understanding of the codebase.

CAPABILITIES:
- You can see the complete architecture via the Repository Map
- You have function signatures (not implementations) for architectural context
- You can request full file content when needed
- You understand dependencies and data flow

TOOLS AVAILABLE:
1. read_file(path: str) -> str
   Read the complete content of a file
   
2. write_file(path: str, content: str) -> None
   Write or overwrite a file
   
3. grep_search(pattern: str, path: Optional[str]) -> List[Match]
   Search for regex pattern in files
   
4. list_directory(path: str) -> List[str]
   List contents of a directory

THINKING PROTOCOL:
Before taking action, you MUST output your reasoning in <think> tags:
<think>
1. What is the user asking for?
2. What files are involved?
3. What dependencies exist?
4. What could go wrong?
5. What's my plan?
</think>

Then output your action in <action> tags:
<action>
read_file("src/auth.py")
</action>

IMPORTANT:
- Always think before acting
- Check the Repository Map to understand architecture
- Use skeletons to understand interfaces before reading full files
- Verify changes don't break dependencies
"""
```

**Prompt Assembly:**
```python
def construct_prompt(
    system_prompt: str,
    user_query: str,
    context: ContextAssembly
) -> str:
    prompt_parts = [system_prompt, "\n\n"]
    
    # Repository Map (for orientation)
    prompt_parts.append("=== REPOSITORY ARCHITECTURE ===\n")
    for part in context.parts:
        if part['type'] == 'repo_map':
            prompt_parts.append(part['content'])
    prompt_parts.append("\n\n")
    
    # Code Context
    prompt_parts.append("=== CODE CONTEXT ===\n")
    for part in context.parts:
        if part['type'] == 'full_file':
            prompt_parts.append(f"\n--- {part['path']} (FULL CONTENT) ---\n")
            prompt_parts.append(part['content'])
        elif part['type'] == 'skeleton':
            prompt_parts.append(f"\n--- {part['path']} (SKELETON) ---\n")
            prompt_parts.append(part['content'])
    
    # User Query
    prompt_parts.append("\n\n=== USER REQUEST ===\n")
    prompt_parts.append(user_query)
    
    return "".join(prompt_parts)
```

### 5.4 Tool Execution Layer

**Deliverable:** Safe, instrumented file operations for the agent

**Implementation:**
```python
from pathlib import Path
from typing import List, Dict, Any
import re
import subprocess

class ToolExecutor:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.execution_log = []
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """Read file with safety checks"""
        full_path = self.project_root / path
        
        # Security: prevent path traversal
        if not full_path.resolve().is_relative_to(self.project_root):
            return {
                'success': False,
                'error': 'Path traversal detected'
            }
        
        try:
            content = full_path.read_text()
            self.log_action('read_file', path, success=True)
            return {
                'success': True,
                'content': content,
                'lines': len(content.split('\n'))
            }
        except Exception as e:
            self.log_action('read_file', path, success=False, error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write file with backup"""
        full_path = self.project_root / path
        
        # Security checks
        if not full_path.resolve().is_relative_to(self.project_root):
            return {'success': False, 'error': 'Path traversal detected'}
        
        # Create backup if file exists
        if full_path.exists():
            backup_path = full_path.with_suffix(full_path.suffix + '.backup')
            backup_path.write_text(full_path.read_text())
        
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self.log_action('write_file', path, success=True)
            
            # Trigger incremental update in graph
            self.update_graph(path, content)
            
            return {
                'success': True,
                'bytes_written': len(content.encode())
            }
        except Exception as e:
            self.log_action('write_file', path, success=False, error=str(e))
            return {'success': False, 'error': str(e)}
    
    def grep_search(
        self,
        pattern: str,
        path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search with ripgrep for performance"""
        try:
            search_path = self.project_root / path if path else self.project_root
            
            result = subprocess.run(
                ['rg', '--json', pattern, str(search_path)],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            matches = self.parse_rg_json(result.stdout)
            self.log_action('grep_search', pattern, success=True)
            
            return {
                'success': True,
                'matches': matches,
                'count': len(matches)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def log_action(self, tool: str, target: str, success: bool, **kwargs):
        """Log all tool executions for debugging"""
        self.execution_log.append({
            'tool': tool,
            'target': target,
            'success': success,
            'timestamp': time.time(),
            **kwargs
        })
```

**Success Criteria:**
- [ ] All file operations logged
- [ ] Path traversal prevented
- [ ] File backups created before overwrites
- [ ] Graph updates triggered automatically
- [ ] Tool execution time < 50ms (except grep)

### 5.5 The Agent Loop

**Deliverable:** The main REPL that ties everything together

**Implementation:**
```python
from semantic_engine import RepoGraph
from rich.console import Console
from rich.markdown import Markdown
import ollama

class AtlasAgent:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.console = Console()
        
        # Initialize Rust core
        self.console.print("[yellow]Initializing repository graph...[/yellow]")
        self.graph = RepoGraph(str(project_root))
        self.graph.build()  # Initial indexing
        
        # Initialize Python components
        self.tools = ToolExecutor(project_root)
        self.budget = ContextBudget(system_prompt=SYSTEM_PROMPT)
        self.assembler = ContextAssembler(self.graph, self.budget)
        
        # Model configuration
        self.model = "deepseek-r1:32b-qwen-distill-q4_K_M"
        
        self.console.print("[green]✓ Agent ready![/green]\n")
    
    def run(self):
        """Main REPL loop"""
        while True:
            try:
                # Get user input
                user_query = self.console.input("[bold blue]You:[/bold blue] ")
                if user_query.lower() in ['exit', 'quit']:
                    break
                
                # Assemble context
                self.budget.user_query = user_query
                context = self.assembler.assemble_context(user_query)
                
                self.console.print(f"[dim]Context: {context.total_tokens} tokens[/dim]")
                
                # Construct prompt
                prompt = construct_prompt(
                    SYSTEM_PROMPT,
                    user_query,
                    context
                )
                
                # Call model
                self.console.print("[yellow]Thinking...[/yellow]")
                response = self.query_model(prompt)
                
                # Parse and execute actions
                thoughts, actions = self.parse_response(response)
                
                # Display thoughts
                if thoughts:
                    self.console.print("\n[bold cyan]Agent Reasoning:[/bold cyan]")
                    self.console.print(Markdown(thoughts))
                
                # Execute actions
                if actions:
                    self.console.print("\n[bold green]Actions:[/bold green]")
                    for action in actions:
                        result = self.execute_action(action)
                        self.console.print(f"  {action['tool']}: {result}")
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def query_model(self, prompt: str) -> str:
        """Send prompt to DeepSeek via Ollama"""
        response = ollama.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.7,
                'num_predict': 4096,
            }
        )
        return response['message']['content']
    
    def parse_response(self, response: str) -> tuple[str, List[Dict]]:
        """Extract <think> and <action> tags"""
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        thoughts = think_match.group(1).strip() if think_match else ""
        
        action_matches = re.findall(
            r'<action>(.*?)</action>',
            response,
            re.DOTALL
        )
        
        actions = []
        for action_text in action_matches:
            # Parse tool calls (simple regex, could use AST)
            # Example: read_file("src/auth.py")
            match = re.match(r'(\w+)\((.*?)\)', action_text.strip())
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                actions.append({
                    'tool': tool_name,
                    'args': eval(f'[{args_str}]')  # UNSAFE: Should use proper parser
                })
        
        return thoughts, actions
    
    def execute_action(self, action: Dict) -> Any:
        """Execute a tool call"""
        tool_name = action['tool']
        args = action['args']
        
        if hasattr(self.tools, tool_name):
            return getattr(self.tools, tool_name)(*args)
        else:
            return {'error': f'Unknown tool: {tool_name}'}
```

**Success Criteria:**
- [ ] Agent starts up in < 5 seconds
- [ ] Responds to queries in < 10 seconds
- [ ] Correctly parses and executes tool calls
- [ ] Maintains conversation context
- [ ] Graceful error handling

### 5.6 The Reflexive Sensory Loop (NEW)

**Deliverable:** "Proprioception" for the agent. The agent should verify its own actions before committing them.

**Concept:** This is "System 2" checking for the AI. Instead of blindly saving code, the agent attempts to parse it first. If the Rust parser (which is extremely fast) detects a syntax error, the action is rejected immediately, and the agent is asked to self-correct.

**Implementation (in `tools.py`):**

```python
    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write file with IMMEDIATE syntax verification"""
        
        # 1. Validation Step (The "System 2" Check)
        # Pass content to Rust parser *before* saving. 
        # This takes <5ms on your M4 Max.
        try:
            # Call into Rust core to parse without building full tree
            syntax_check = self.rust_core.check_syntax(content, language_from_path(path))
            
            if not syntax_check.is_valid:
                # REJECT the action. The agent must try again.
                return {
                    'success': False,
                    'error': f"Syntax Error on line {syntax_check.line}: {syntax_check.message}. Please fix before saving.",
                    'status': 'refused_by_parser'
                }
        except Exception as e:
            # Fallback if parser fails (should not happen)
            pass
            
        # 2. Execution Step (Original Logic)
        full_path = self.project_root / path
        
        # Create backup
        if full_path.exists():
            backup_path = full_path.with_suffix(full_path.suffix + '.backup')
            backup_path.write_text(full_path.read_text())
            
        full_path.write_text(content)
        self.log_action('write_file', path, success=True)
        
        # 3. Update Graph
        self.update_graph(path, content)
        
        return {'success': True}

```

**Success Criteria:**

* [ ] Prevents 100% of syntactically invalid Python/Rust code from being saved
* [ ] Verification adds < 10ms latency
* [ ] Agent receives clear error messages enabling self-correction

---

## PHASE 6: OPTIMIZATION & POLISH

**Duration:** 1-2 weeks

**Goal:** Performance tuning, error handling, and UX improvements

### 6.1 Memory Optimization

**Target:** Run comfortably within 36GB M4 Max RAM

**Optimizations:**

1. **Lazy Skeleton Loading:**
```rust
// Don't load all skeletons into memory
pub struct LazyRepoGraph {
    graph: DiGraph<FileNodeRef, ImportEdge>,
    skeleton_cache: LruCache<PathBuf, CodeSkeleton>,
}

// Skeletons loaded on-demand
impl LazyRepoGraph {
    pub fn get_skeleton(&mut self, path: &Path) -> &CodeSkeleton {
        if !self.skeleton_cache.contains(path) {
            let skeleton = self.load_skeleton(path);
            self.skeleton_cache.put(path.to_path_buf(), skeleton);
        }
        self.skeleton_cache.get(path).unwrap()
    }
}
```

2. **Tree Eviction:**
```rust
// Don't keep all trees in memory, only recent ones
pub struct TreeCache {
    cache: LruCache<PathBuf, Tree>,
    max_size: usize,
}

impl TreeCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: LruCache::new(max_size),
            max_size,
        }
    }
}
```

**Success Criteria:**
- [ ] Peak memory usage < 4GB for 10k file repo
- [ ] Memory stable over long sessions (no leaks)
- [ ] Cache hit rate > 80% for typical workflows

### 6.2 Error Handling & Graceful Degradation

**Principle:** Never crash, always provide partial results

**Error Scenarios:**

1. **Parser Failure:**
```rust
// If Tree-sitter fails, fall back to regex-based extraction
pub fn parse_with_fallback(path: &Path) -> Result<ParsedFile, ParseError> {
    match parse_with_treesitter(path) {
        Ok(result) => Ok(result),
        Err(e) => {
            warn!("Tree-sitter failed for {}: {}, using fallback", path, e);
            parse_with_regex(path)  // Crude but works
        }
    }
}
```

2. **Import Resolution Failure:**
```rust
// If import can't be resolved, mark as "external" instead of failing
pub fn resolve_import_safe(import: &str) -> ImportResolution {
    match self.resolve_import(import) {
        Some(path) => ImportResolution::Resolved(path),
        None => {
            if is_stdlib(import) {
                ImportResolution::StandardLibrary
            } else {
                ImportResolution::External(import.to_string())
            }
        }
    }
}
```

3. **Corrupted Graph State:**
```python
# If graph is inconsistent, offer to rebuild
def check_graph_health(self) -> HealthStatus:
    issues = []
    
    # Check for orphaned nodes
    orphans = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
    if len(orphans) > self.graph.node_count() * 0.5:
        issues.append("Too many orphaned nodes")
    
    # Check for broken edges
    for edge in self.graph.edges():
        if not self.file_exists(edge.source) or not self.file_exists(edge.target):
            issues.append(f"Broken edge: {edge}")
    
    if issues:
        return HealthStatus.Degraded(issues)
    else:
        return HealthStatus.Healthy
```

**Success Criteria:**
- [ ] No crashes on malformed code
- [ ] Helpful error messages for users
- [ ] Automatic recovery when possible
- [ ] Clear indication when operating in degraded mode

### 6.3 Progress Indicators & UX

**Deliverable:** Rich terminal UI with progress feedback

**Implementation:**
```python
from rich.progress import Progress, SpinnerColumn, TextColumn

def build_graph_with_progress(self):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        
        task1 = progress.add_task("[cyan]Scanning files...", total=None)
        files = self.scan_repository()
        progress.update(task1, completed=True)
        
        task2 = progress.add_task(
            f"[yellow]Parsing {len(files)} files...",
            total=len(files)
        )
        for file in files:
            self.parse_file(file)
            progress.advance(task2)
        
        task3 = progress.add_task("[magenta]Building graph...", total=None)
        self.construct_graph()
        progress.update(task3, completed=True)
        
        task4 = progress.add_task("[green]Calculating PageRank...", total=None)
        self.calculate_pagerank()
        progress.update(task4, completed=True)
```

**Success Criteria:**
- [ ] Clear feedback for all long operations
- [ ] ETA displayed for indexing
- [ ] Beautiful, colorful output
- [ ] Accessibility for screen readers

---

## TESTING STRATEGY

### Unit Tests (Rust)
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_python_parser() { /* ... */ }
    
    #[test]
    fn test_import_resolution() { /* ... */ }
    
    #[test]
    fn test_pagerank_convergence() { /* ... */ }
}
```

### Integration Tests (Python)
```python
def test_end_to_end_workflow():
    # Create test repository
    with tempfile.TemporaryDirectory() as tmpdir:
        create_test_repo(tmpdir)
        
        # Initialize agent
        agent = AtlasAgent(Path(tmpdir))
        
        # Verify graph built correctly
        assert agent.graph.node_count() > 0
        
        # Test query
        result = agent.process_query("Explain the architecture")
        assert len(result) > 0
```

### Benchmark Suite
```rust
#[bench]
fn bench_parse_10k_files(b: &mut Bencher) { /* ... */ }

#[bench]
fn bench_pagerank_large_graph(b: &mut Bencher) { /* ... */ }
```

### Real-World Validation
Test on popular open-source repositories:
- [ ] FastAPI (Python, ~100 files)
- [ ] Tokio (Rust, ~200 files)
- [ ] React (TypeScript, ~1000 files)
- [ ] Kubernetes (Go, ~5000 files)

**Success Metrics:**
- [ ] All repos index in < 10 seconds
- [ ] All repos query in < 1 second
- [ ] PageRank identifies actual core files (manual validation)
- [ ] No crashes on any test repo

---

## DEPLOYMENT CHECKLIST

### Pre-Launch
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Benchmarks meet targets
- [ ] Memory profiling shows no leaks
- [ ] Documentation complete
- [ ] Example repositories tested

### Installation
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/yourusername/atlas.git
cd atlas

# Build Rust core
cd rust_core
maturin develop --release
cd ..

# Install Python dependencies
pip install -e ".[dev]"

# Install Ollama and pull model
curl https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:32b-qwen-distill-q4_K_M

# Run agent
atlas /path/to/your/project
```

### First Run Experience
```
$ atlas ~/my_project

  ____             ____       
 / ___|  ___ _ __ |  _ \ ___  
 \___ \ / _ \ '_ \| |_) / _ \ 
  ___) |  __/ | | |  __/ (_) |
 |____/ \___|_| |_|_|   \___/ 
                               
 Sentient Repository v0.1.0

[1/4] Scanning repository...
      Found 2,847 files (523 Python, 189 Rust, 1,203 JavaScript)

[2/4] Parsing files...
      ████████████████████████ 100% (2,847/2,847) [00:03<00:00]

[3/4] Building dependency graph...
      Resolved 1,234 imports
      Created 2,847 nodes, 3,456 edges

[4/4] Calculating PageRank...
      Converged in 18 iterations

✓ Repository indexed! Top architectural files:
  1. src/core/database.py (rank: 0.156)
  2. src/auth/session.py (rank: 0.134)
  3. src/models/user.py (rank: 0.098)

Ready! Ask me anything about this codebase.

You: 
```

---

## SUCCESS CRITERIA (FINAL)

### Performance

* [ ] Index 10k files in < 10 seconds
* [ ] Query response in < 5 seconds
* [ ] Incremental updates in < 100ms

### Quality

* [ ] **Symbol-level accuracy:** Links `call()` to `def()` correctly
* [ ] **Context Relevance:** Anchor & Expand consistently finds related files for "vague" queries
* [ ] **Safety:** Reflexive loop catches syntax errors before write

### Agent Capability

* [ ] Can explain repository architecture
* [ ] Can navigate dependencies correctly
* [ ] Can make safe code modifications with self-correction

---

## CONCLUSION

This roadmap prioritizes **structural intelligence** over raw speed. By upgrading from a simple file graph to a **Symbol-Level Graph** and implementing **Reflexive Tooling**, we ensure the agent doesn't just "grep" the codebase, but actually understands the flow of execution.

The addition of **Vector Anchoring** ensures that even when the agent's architectural knowledge is partial, it can locate the correct entry point based on semantic intent.

Let's build Atlas.