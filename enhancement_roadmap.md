# Atlas Enhancement Roadmap: Code Property Graph (CPG) Evolution

## Executive Summary

Atlas currently operates at **file-level granularity**: nodes are files, edges are `Import` or `SymbolUsage` relationships between files, and ASTs are parsed transiently during graph construction but discarded immediately. This roadmap describes the evolution from a file-level dependency graph to a full **Code Property Graph (CPG)** — a unified data structure that layers AST, Control Flow Graph (CFG), and Program Dependence Graph (PDG) into a single queryable graph.

A CPG enables capabilities that file-level analysis cannot provide: precise variable tracking, data flow tracing across function boundaries, taint analysis, dead code detection, and vulnerability pattern matching.

---

## 1. Current State Audit

### 1.1 Graph Model (`rust_core/src/graph.rs`)

| Component | Current State |
|-----------|--------------|
| **Node type** | `FileNode` — one node per source file |
| **Edge types** | `EdgeKind::Import` (weight 1.0) and `EdgeKind::SymbolUsage` (weight 2.0) |
| **Graph library** | `petgraph::DiGraph<FileNode, EdgeKind>` |
| **Ranking** | PageRank with edge-weight-aware damping |
| **Incremental updates** | `classify_change()` with 4 tiers: `Local`, `FileScope`, `GraphScope`, `FullRebuild` |
| **Consistency** | `validate_consistency()` checks path↔index integrity and symbol edge correctness |

**`FileNode` fields:**
```rust
pub struct FileNode {
    pub path: PathBuf,
    pub definitions: Vec<String>,   // Symbol names only — no position, no kind
    pub usages: Vec<String>,        // Symbol names only
    pub rank: f64,
    pub imports_hash: u64,
    pub definitions_hash: u64,
    pub usages_hash: u64,
    pub content_hash: u64,
}
```

**Key limitation:** `FileNode` stores symbol *names* as flat strings. There is no record of where in the file each symbol lives, what kind it is (function vs. class vs. variable), or what its scope is.

### 1.2 Parsing Infrastructure (`rust_core/src/parser.rs`)

| Component | Current State |
|-----------|--------------|
| **Languages** | Python, Rust, JavaScript, JSX, TypeScript, TSX, Go (7 variants, 6 grammars) |
| **Parser management** | `ParserPool` — one `tree_sitter::Parser` per language |
| **Symbol extraction** | `SymbolHarvester` using `.scm` query files per language |
| **Symbol output** | `Symbol { name, kind, start_byte, end_byte, is_definition }` |
| **Skeleton generation** | `create_skeleton()` — strips function bodies, keeps signatures/docstrings |
| **Tree normalization** | `normalize_tree()` / `NormalizedNode` — prunes trivial AST nodes |

**Key insight:** The `Symbol` struct already carries `SymbolKind` (Function, Method, Class, Interface, Type, Enum, Variable) and byte positions. This rich data is **harvested but then flattened** to a plain `String` name before storage in `FileNode`. The CPG can preserve this full structure.

### 1.3 Symbol Table (`rust_core/src/symbol_table.rs`)

```rust
pub struct SymbolIndex {
    pub definitions: HashMap<String, Vec<PathBuf>>,    // symbol_name → [defining files]
    pub usages: HashMap<PathBuf, Vec<String>>,          // file → [used symbol names]
    pub users: HashMap<String, Vec<PathBuf>>,           // symbol_name → [using files]
}
```

**Limitation:** Cross-references are file-level only. "File A uses symbol X defined in file B" — but not *which* usage site in A connects to *which* definition site in B.

### 1.4 Import Resolution (`rust_core/src/import_resolver.rs`)

- `PythonImportResolver`: Module indexing with stdlib/third-party filtering, relative import support
- `JsTsImportResolver`: tsconfig.json path alias resolution, multi-extension probing
- Both use tree-sitter queries to extract import statements from ASTs

### 1.5 Incremental Parsing (`rust_core/src/incremental_parser.rs`)

- `IncrementalParser` with LRU caches for trees (100 entries) and source text
- `TextEdit` → `InputEdit` conversion for tree-sitter incremental reparsing
- Trees are retained in the cache but **not connected to the graph**

### 1.6 Query Files (`rust_core/queries/`)

| Language | File | Captures |
|----------|------|----------|
| Python | `queries/python/symbols.scm` | `definition.function`, `definition.method`, `definition.class`, `definition.variable`, `reference.call` |
| Rust | `queries/rust/symbols.scm` | `definition.function`, `definition.method`, `definition.class`, `definition.type`, `definition.enum`, `reference.call` |
| JavaScript | `queries/javascript/symbols.scm` | `definition.function`, `definition.class`, `definition.variable`, `definition.method`, `reference.call` |
| TypeScript | `queries/typescript/symbols.scm` | Same as JavaScript + `definition.interface`, `definition.type`, `definition.enum` |

### 1.7 PyO3 Boundary (`rust_core/src/lib.rs`)

- `PyRepoGraph` wraps `graph::RepoGraph`
- Exposed methods: `build_complete`, `add_file`, `remove_file`, `update_file`, `generate_map`, `get_dependencies`, `get_dependents`, `get_top_ranked_files`, `get_skeleton`
- Custom Python exceptions: `GraphError`, `ParseError`, `NodeNotFoundError`

---

## 2. Gap Analysis: Current State → CPG

### 2.1 AST Persistence (Gap: Critical)

**Current:** Trees are parsed during `build()`, symbols are extracted, then trees are dropped. The `IncrementalParser` caches trees separately but they are not integrated with `RepoGraph`.

**Required:** Every file's AST must be retained and accessible for CFG/PDG construction and queries. The existing `IncrementalParser.trees: LruCache<PathBuf, Tree>` is a starting point but needs to be either integrated into `RepoGraph` or shared via `Arc`.

### 2.2 Fine-Grained Node Model (Gap: Critical)

**Current:** `DiGraph<FileNode, EdgeKind>` — one node per file.

**Required:** Nodes must represent individual code elements (functions, classes, statements, variables, expressions). A single file should expand into dozens or hundreds of graph nodes.

### 2.3 Rich Edge Types (Gap: Critical)

**Current:** Only `Import` and `SymbolUsage` edge kinds.

**Required:** At minimum 8-10 edge types covering AST parentage, control flow, data flow, and call relationships.

### 2.4 Control Flow Graph (Gap: Complete)

No CFG construction exists. Need to build per-function CFGs by walking ASTs and connecting statements in execution order, with branch edges for conditionals/loops/exceptions.

### 2.5 Data Flow / Program Dependence Graph (Gap: Complete)

No data flow analysis exists. Need reaching-definitions analysis, def-use chain construction, and inter-procedural call graph for cross-function tracking.

### 2.6 Query Interface (Gap: Complete)

No structured query API exists beyond `get_dependencies()`/`get_dependents()`. Need traversal primitives, pattern matching, and reachability queries over the CPG.

---

## 3. Implementation Plan

### Phase 1: AST Persistence & Fine-Grained Node Model

**Goal:** Retain parsed ASTs and introduce sub-file graph nodes while keeping the existing file-level graph operational.

**Duration estimate:** 2-3 weeks

#### 3.1.1 New Core Types (`rust_core/src/cpg.rs` — new file)

```rust
use petgraph::graph::{DiGraph, NodeIndex};
use std::path::PathBuf;
use tree_sitter::Tree;
use std::sync::Arc;
use parking_lot::RwLock;

/// A node in the Code Property Graph.
/// Each variant represents a different granularity of code element.
#[derive(Debug, Clone)]
pub enum CpgNode {
    /// A source file (preserves current FileNode role)
    File {
        path: PathBuf,
        content_hash: u64,
        rank: f64,
    },
    /// A function or method definition
    Function {
        name: String,
        file: PathBuf,
        byte_range: (usize, usize),
        line_range: (usize, usize),
        is_method: bool,
        params: Vec<Parameter>,
        return_type: Option<String>,
    },
    /// A class or struct definition
    Class {
        name: String,
        file: PathBuf,
        byte_range: (usize, usize),
        line_range: (usize, usize),
        bases: Vec<String>,
    },
    /// A variable binding (local, parameter, global)
    Variable {
        name: String,
        scope: ScopeKind,
        byte_range: (usize, usize),
        line_range: (usize, usize),
    },
    /// A single statement (for CFG nodes)
    Statement {
        kind: StatementKind,
        byte_range: (usize, usize),
        line_range: (usize, usize),
    },
    /// An expression (for data flow tracking)
    Expression {
        kind: ExpressionKind,
        byte_range: (usize, usize),
    },
    /// CFG entry/exit sentinel nodes
    CfgEntry { function: NodeIndex },
    CfgExit { function: NodeIndex },
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<String>,
    pub has_default: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeKind {
    Global,
    Local,
    Parameter,
    ClassField,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementKind {
    Assignment,
    Return,
    If,
    For,
    While,
    Try,
    With,
    Assert,
    Raise,
    Break,
    Continue,
    Pass,
    ExpressionStatement,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionKind {
    Call,
    Attribute,
    BinaryOp,
    UnaryOp,
    Literal,
    Name,
    Subscript,
    Other(String),
}

/// Edge types in the Code Property Graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpgEdge {
    // === File-level (preserves current behavior) ===
    /// File A imports file B
    Import,
    /// File A uses a symbol defined in file B
    SymbolUsage,

    // === AST relationships ===
    /// Parent → child in the AST (File contains Function, Function contains Statement, etc.)
    AstChild { index: usize },
    /// Node → its enclosing scope/container
    EnclosedBy,

    // === Control Flow ===
    /// Sequential control flow: statement A executes, then statement B
    ControlFlowNext,
    /// Conditional branch: true branch
    ControlFlowTrue,
    /// Conditional branch: false branch
    ControlFlowFalse,
    /// Exception flow: try → except handler
    ControlFlowException,
    /// Loop back-edge
    ControlFlowBack,

    // === Data Flow ===
    /// Variable definition reaches this usage (def-use chain)
    DataFlowReach,
    /// Value flows from expression A into variable B (assignment)
    DataFlowAssign,
    /// Argument value flows into parameter
    DataFlowArgument { position: usize },
    /// Return value flows to call site
    DataFlowReturn,

    // === Call Graph ===
    /// Function A calls function B
    Calls,
    /// Function B is called by function A (reverse of Calls)
    CalledBy,

    // === Type/Inheritance ===
    /// Class A extends/inherits from class B
    Inherits,
    /// Class A implements interface B
    Implements,
}
```

#### 3.1.2 Dual-Graph Architecture

Rather than replacing `RepoGraph` immediately, introduce `CpgLayer` as an overlay:

```rust
/// The CPG layer sits alongside the existing RepoGraph.
/// Phase 1 builds the fine-grained nodes; later phases add CFG/PDG edges.
pub struct CpgLayer {
    /// The fine-grained graph
    pub graph: DiGraph<CpgNode, CpgEdge>,
    /// File path → File node index in the CPG
    pub file_nodes: HashMap<PathBuf, NodeIndex>,
    /// File path → list of all CPG node indices belonging to that file
    pub file_members: HashMap<PathBuf, Vec<NodeIndex>>,
    /// Retained ASTs for each file
    pub trees: HashMap<PathBuf, Arc<Tree>>,
    /// Source code cache (needed for AST text extraction)
    pub sources: HashMap<PathBuf, Arc<String>>,
    /// Function name → defining CPG node indices
    pub function_index: HashMap<String, Vec<NodeIndex>>,
    /// Class name → defining CPG node index
    pub class_index: HashMap<String, Vec<NodeIndex>>,
}
```

#### 3.1.3 AST → CPG Node Extraction

New tree-sitter query files per language to extract structured nodes:

- `queries/python/cpg_nodes.scm` — capture functions, classes, assignments, returns, calls
- `queries/rust/cpg_nodes.scm` — capture fn items, impl blocks, struct/enum, let bindings
- `queries/javascript/cpg_nodes.scm` — capture function declarations, arrow functions, class declarations
- `queries/typescript/cpg_nodes.scm` — same as JS + interface/type declarations

**AST child edges** are built by walking the tree-sitter CST top-down:
1. File node → Function/Class nodes (top-level definitions)
2. Class node → Method nodes (class body)
3. Function node → Statement nodes (function body)
4. Statement node → Expression nodes (as needed for data flow)

#### 3.1.4 Integration with Existing `RepoGraph`

The file-level `RepoGraph` continues to handle:
- File watching and incremental updates
- PageRank computation
- Import resolution
- Repository map generation

The `CpgLayer` is rebuilt for files that change (using the same `UpdateTier` classification). When `classify_change()` returns `Local`, only the affected function's sub-tree is rebuilt. When it returns `FileScope` or higher, the entire file's CPG nodes are rebuilt.

#### 3.1.5 Reusable Infrastructure

| Existing Component | Reuse Strategy |
|---|---|
| `SymbolHarvester` + `.scm` queries | Extend queries to capture positions, parameters, return types |
| `IncrementalParser` tree/source caches | Merge into `CpgLayer.trees` / `CpgLayer.sources` |
| `ParserPool` | Reuse directly for all parsing |
| `normalize_tree()` / `NormalizedNode` | Use as intermediate representation before CPG node creation |
| `classify_change()` + `UpdateTier` | Drive selective CPG rebuilds |
| `skeleton_cache` | Keep for context assembly; independent of CPG |

---

### Phase 2: Intra-Procedural Control Flow Graph

**Goal:** For every function/method, build a CFG connecting statements in execution order.

**Duration estimate:** 2-3 weeks

#### 3.2.1 CFG Construction Algorithm

For each function node in the CPG:

1. Create `CfgEntry` and `CfgExit` sentinel nodes
2. Walk the function body's statements in AST order
3. Connect sequential statements with `ControlFlowNext` edges
4. For branching constructs, insert appropriate edges:

```
if condition:        →  ControlFlowTrue  → if_body_first_stmt
                     →  ControlFlowFalse → else_body_first_stmt (or next_after_if)

for x in iter:       →  ControlFlowTrue  → loop_body_first_stmt
                     →  ControlFlowFalse → next_after_loop
  loop_body_last     →  ControlFlowBack  → for_header

while condition:     →  ControlFlowTrue  → loop_body_first_stmt
                     →  ControlFlowFalse → next_after_while
  loop_body_last     →  ControlFlowBack  → while_header

try:                 →  ControlFlowNext  → try_body_first_stmt
  (any stmt in try)  →  ControlFlowException → except_handler

return expr          →  ControlFlowNext  → CfgExit

break                →  ControlFlowNext  → next_after_enclosing_loop
continue             →  ControlFlowBack  → enclosing_loop_header
```

#### 3.2.2 Language-Specific CFG Rules

| Construct | Python | Rust | JavaScript/TypeScript | Go |
|-----------|--------|------|----------------------|-----|
| Conditionals | `if/elif/else` | `if/else`, `match` | `if/else`, `switch` | `if/else`, `switch` |
| Loops | `for`, `while`, `async for` | `for`, `while`, `loop` | `for`, `while`, `do-while`, `for-of`, `for-in` | `for`, `range` |
| Exception flow | `try/except/finally` | `Result` + `?` (no exceptions) | `try/catch/finally` | `defer`, `panic/recover` |
| Early return | `return`, `yield` | `return`, `?` operator | `return`, `yield` | `return` |
| Generators | `yield`, `yield from` | N/A (iterators are different) | `yield`, `yield*` | N/A (goroutines are different) |

**Python and JavaScript share the most CFG patterns.** Rust's `match` and `?` operator require custom handling. Go's `defer` creates implicit control flow edges to function exit.

#### 3.2.3 New Module: `rust_core/src/cfg.rs`

```rust
pub struct CfgBuilder {
    language: SupportedLanguage,
}

impl CfgBuilder {
    pub fn new(language: SupportedLanguage) -> Self;

    /// Build CFG edges for a single function.
    /// Mutates the CPG graph in-place, adding ControlFlow* edges
    /// between the function's statement nodes.
    pub fn build_function_cfg(
        &self,
        cpg: &mut CpgLayer,
        function_node: NodeIndex,
        tree: &Tree,
        source: &str,
    );

    /// Walk a block of statements and return (entry_node, exit_nodes).
    /// exit_nodes is a Vec because branches produce multiple exit points.
    fn build_block(
        &self,
        cpg: &mut CpgLayer,
        stmts: &[NodeIndex],
    ) -> (NodeIndex, Vec<NodeIndex>);

    fn build_if(&self, cpg: &mut CpgLayer, node: tree_sitter::Node, source: &str) -> (NodeIndex, Vec<NodeIndex>);
    fn build_for(&self, cpg: &mut CpgLayer, node: tree_sitter::Node, source: &str) -> (NodeIndex, Vec<NodeIndex>);
    fn build_while(&self, cpg: &mut CpgLayer, node: tree_sitter::Node, source: &str) -> (NodeIndex, Vec<NodeIndex>);
    fn build_try(&self, cpg: &mut CpgLayer, node: tree_sitter::Node, source: &str) -> (NodeIndex, Vec<NodeIndex>);
    fn build_match(&self, cpg: &mut CpgLayer, node: tree_sitter::Node, source: &str) -> (NodeIndex, Vec<NodeIndex>);
}
```

#### 3.2.4 Testing Strategy

- **Per-language golden tests:** Parse known code snippets, build CFG, assert specific edge sets
- **Reachability tests:** "Can statement X reach statement Y?" for various control flow patterns
- **Break/continue correctness:** Verify these jump to the correct loop boundary
- **Exception flow:** Verify try/except creates edges to handlers
- **Benchmark:** CFG construction time per 1000 LOC (target: <10ms)

---

### Phase 3: Intra-Procedural Data Flow (Program Dependence Graph)

**Goal:** Build def-use chains within functions — track where variables are defined and where those definitions are consumed.

**Duration estimate:** 3-4 weeks

#### 3.3.1 Reaching Definitions Analysis

Classic worklist algorithm operating over the CFG:

```
For each statement S in the CFG:
    GEN(S)  = set of (variable, S) pairs for variables defined at S
    KILL(S) = set of (variable, S') pairs for all other definitions of the same variable

    IN(S)  = union of OUT(P) for all predecessors P of S in the CFG
    OUT(S) = GEN(S) ∪ (IN(S) - KILL(S))

Iterate until fixed point.
```

For each variable usage at statement U, the reaching definitions `IN(U)` tell us which assignment statements could have produced the value being read. Each such pair generates a `DataFlowReach` edge from the defining statement to the usage.

#### 3.3.2 Assignment Tracking

For each assignment `x = expr`:
- Create `DataFlowAssign` edge from `expr` node to `x` variable node
- Register `x` as a new definition (for reaching definitions)

For augmented assignments (`x += expr`, `x.append(y)`):
- Both a use and a definition of `x`

#### 3.3.3 Scope Analysis

Before data flow, build a scope tree:

```
Module scope
  └── Function scope (my_function)
       ├── Parameter scope (a, b)
       └── Local scope
            └── For-loop scope (x)
```

Python's scoping rules (LEGB: Local, Enclosing, Global, Built-in) must be modeled. Rust's ownership/borrowing adds complexity. JavaScript's `var` vs `let`/`const` hoisting needs handling.

#### 3.3.4 New Module: `rust_core/src/dataflow.rs`

```rust
pub struct DataFlowAnalyzer {
    language: SupportedLanguage,
}

impl DataFlowAnalyzer {
    pub fn new(language: SupportedLanguage) -> Self;

    /// Run reaching definitions on a single function's CFG.
    /// Adds DataFlow* edges to the CPG.
    pub fn analyze_function(
        &self,
        cpg: &mut CpgLayer,
        function_node: NodeIndex,
    );

    /// Extract GEN/KILL sets for a statement node.
    fn compute_gen_kill(
        &self,
        cpg: &CpgLayer,
        stmt: NodeIndex,
    ) -> (HashSet<(String, NodeIndex)>, HashSet<(String, NodeIndex)>);

    /// Worklist iteration to fixed point.
    fn reaching_definitions(
        &self,
        cpg: &CpgLayer,
        cfg_nodes: &[NodeIndex],
    ) -> HashMap<NodeIndex, HashSet<(String, NodeIndex)>>;
}
```

#### 3.3.5 Testing Strategy

- **Simple assignment chains:** `x = 1; y = x; z = y` → verify DataFlowReach from first assignment to second, second to third
- **Branch merging:** `if cond: x = 1 else: x = 2; print(x)` → two reaching definitions for `x` at print
- **Loop data flow:** `for i in range(10): x = x + i` → `x` has reaching definitions from both the initializer and the loop body
- **Shadow variables:** Inner scope `x` should not interfere with outer `x`

---

### Phase 4: Inter-Procedural Analysis & Call Graph

**Goal:** Connect data flow across function boundaries via call sites and return values.

**Duration estimate:** 2-3 weeks

#### 3.4.1 Call Graph Construction

For each `Call` expression in the CPG:
1. Resolve the callee to a function definition (using the symbol index)
2. Add `Calls` edge from the calling function to the callee function
3. Add `CalledBy` reverse edge
4. For each argument, add `DataFlowArgument { position }` edge from the argument expression to the corresponding parameter

For return statements:
1. Add `DataFlowReturn` edge from the return expression to the call site that receives the value

#### 3.4.2 Challenges

| Challenge | Approach |
|-----------|----------|
| **Dynamic dispatch** (Python, JS) | Resolve statically where possible; mark unresolved calls |
| **Higher-order functions** | Track function references as data flow; resolve at call sites |
| **Closures** | Model captured variables with `DataFlowReach` edges from enclosing scope |
| **Method resolution** | Use class hierarchy (`Inherits` edges) plus type inference heuristics |
| **Recursive calls** | Handled naturally by the call graph; data flow uses fixed-point iteration |

#### 3.4.3 Conservative vs. Optimistic Resolution

Start with **conservative** analysis:
- Only resolve calls where the callee is unambiguously identified
- Mark ambiguous calls (e.g., `obj.method()` without type info) as unresolved
- Unresolved calls produce no inter-procedural edges but are flagged for the query API

Future enhancement: integrate lightweight type inference to resolve more calls.

---

### Phase 5: Query Interface & PyO3 Exposure

**Goal:** Make the CPG queryable from both Rust and Python.

**Duration estimate:** 2-3 weeks

#### 3.5.1 Rust Query API (`rust_core/src/cpg_query.rs`)

```rust
pub struct CpgQuery<'a> {
    cpg: &'a CpgLayer,
}

impl<'a> CpgQuery<'a> {
    // === Structural Queries ===

    /// Get all functions defined in a file
    pub fn functions_in_file(&self, path: &Path) -> Vec<NodeIndex>;

    /// Get all methods of a class
    pub fn methods_of_class(&self, class_node: NodeIndex) -> Vec<NodeIndex>;

    /// Get the class hierarchy (superclasses/subclasses)
    pub fn class_hierarchy(&self, class_node: NodeIndex) -> Vec<(NodeIndex, CpgEdge)>;

    // === Control Flow Queries ===

    /// Can statement A reach statement B via control flow?
    pub fn is_reachable(&self, from: NodeIndex, to: NodeIndex) -> bool;

    /// Get all possible execution paths from A to B (up to max_depth)
    pub fn paths_between(
        &self,
        from: NodeIndex,
        to: NodeIndex,
        max_depth: usize,
    ) -> Vec<Vec<NodeIndex>>;

    /// Get all statements reachable from a given statement
    pub fn reachable_statements(&self, from: NodeIndex) -> HashSet<NodeIndex>;

    // === Data Flow Queries ===

    /// Where does the value of variable `var` at statement `stmt` come from?
    pub fn reaching_definitions(
        &self,
        var: &str,
        stmt: NodeIndex,
    ) -> Vec<NodeIndex>;

    /// Where does the value defined at `def_stmt` flow to?
    pub fn definition_uses(&self, def_stmt: NodeIndex) -> Vec<NodeIndex>;

    /// Taint tracking: can data from `source` reach `sink`?
    pub fn taint_reachable(
        &self,
        source: NodeIndex,
        sink: NodeIndex,
    ) -> bool;

    /// Taint tracking with path: return the data flow path from source to sink
    pub fn taint_path(
        &self,
        source: NodeIndex,
        sink: NodeIndex,
    ) -> Option<Vec<NodeIndex>>;

    // === Call Graph Queries ===

    /// What functions does this function call?
    pub fn callees(&self, function: NodeIndex) -> Vec<NodeIndex>;

    /// What functions call this function?
    pub fn callers(&self, function: NodeIndex) -> Vec<NodeIndex>;

    /// Transitive call graph (up to max_depth)
    pub fn call_graph(
        &self,
        function: NodeIndex,
        max_depth: usize,
    ) -> DiGraph<NodeIndex, ()>;

    // === Pattern Matching ===

    /// Find all call sites matching a pattern (e.g., "os.system(*)")
    pub fn find_calls(&self, pattern: &str) -> Vec<NodeIndex>;

    /// Find all assignments where the RHS matches a pattern
    pub fn find_assignments(&self, lhs_pattern: &str) -> Vec<NodeIndex>;

    /// Find dead code: functions never called
    pub fn dead_functions(&self) -> Vec<NodeIndex>;

    /// Find unused variables: defined but never used
    pub fn unused_variables(&self) -> Vec<NodeIndex>;
}
```

#### 3.5.2 PyO3 Wrappers (`rust_core/src/lib.rs` additions)

```rust
#[pyclass(name = "CpgQuery")]
pub struct PyCpgQuery { /* ... */ }

#[pymethods]
impl PyCpgQuery {
    /// Get all functions in a file.
    /// Returns: List[Dict] with keys: name, line_start, line_end, params
    fn functions_in_file(&self, path: &str) -> PyResult<Vec<PyDict>>;

    /// Check if data can flow from source to sink.
    /// Both source and sink specified as (file_path, line_number).
    fn taint_reachable(
        &self,
        source_file: &str,
        source_line: usize,
        sink_file: &str,
        sink_line: usize,
    ) -> PyResult<bool>;

    /// Find dead functions across the project.
    /// Returns: List[Dict] with keys: name, file, line_start
    fn dead_functions(&self) -> PyResult<Vec<PyDict>>;

    /// Get the call graph for a function.
    /// Returns: Dict with keys: nodes (List[str]), edges (List[Tuple[str, str]])
    fn call_graph(&self, file: &str, function_name: &str, max_depth: usize) -> PyResult<PyDict>;
}
```

#### 3.5.3 Integration with Agent Context Assembly

Update `python_shell/atlas/context.py` to leverage CPG queries:

1. **Smarter neighborhood expansion:** Instead of BFS over file-level edges, traverse the call graph to find *functions* that are semantically related, then include only those functions' code (not entire files)
2. **Focused context:** When the user asks about a specific function, include its callers, callees, and data flow dependencies — not just the files they happen to live in
3. **Vulnerability detection:** Before generating code modifications, run taint analysis to ensure the proposed change doesn't introduce data flow from untrusted sources to sensitive sinks

---

## 4. Per-Language Considerations

### 4.1 Python (Primary Target)

Python should be the first language with full CPG support due to:
- Existing mature tree-sitter queries
- Simplest scoping rules among the supported languages
- Most used language in Atlas's target audience
- `create_skeleton()` already handles Python's AST structure

**Python-specific challenges:**
- Dynamic typing makes call resolution harder (no static type to look up methods)
- `*args`/`**kwargs` complicate argument-to-parameter mapping
- Decorators may wrap/replace functions
- List comprehensions and generator expressions create implicit scopes

### 4.2 Rust

**Rust-specific challenges:**
- Ownership/borrowing: Data flow must model moves, borrows, and lifetimes
- Pattern matching: `match` arms create complex control flow
- Macros: Macro-expanded code is not visible to tree-sitter (tree-sitter parses pre-expansion syntax)
- Trait method resolution: Requires type information not available from syntax alone
- `?` operator: Creates implicit early-return control flow

**Recommendation:** Defer full Rust CPG to after Python is complete. Start with CFG only (no data flow) since Rust's type system already catches most data flow bugs at compile time.

### 4.3 JavaScript / TypeScript

**JS/TS-specific challenges:**
- `var` hoisting vs. `let`/`const` block scoping
- Prototype-based inheritance (JS) vs. class syntax
- `this` binding rules (arrow functions vs. regular functions)
- Async/await and Promise chains create complex control flow
- TypeScript types are erased at runtime but useful for call resolution

**Recommendation:** Implement alongside Python since JS/TS share similar CFG patterns. TypeScript's type annotations can significantly improve call resolution accuracy compared to plain JavaScript.

### 4.4 Go

**Go-specific challenges:**
- Goroutines: `go func()` creates concurrent control flow
- Channels: Data flow through channels is implicit
- `defer` statements: Execute in LIFO order at function exit
- Multiple return values: Must model as tuples in data flow
- Interface satisfaction is implicit (structural typing)

**Recommendation:** Defer to after Python + JS/TS. Go's `defer` and goroutine semantics require specialized modeling.

---

## 5. Incremental CPG Updates

The CPG must support incremental updates to avoid rebuilding the entire graph on every file change. Leverage the existing `UpdateTier` classification:

| Tier | CPG Action |
|------|-----------|
| `Local` | Rebuild only the modified function's CPG nodes and re-run CFG + data flow for that function |
| `FileScope` | Rebuild all CPG nodes for the file; re-run CFG + data flow for all functions in the file; update call graph edges |
| `GraphScope` | Same as `FileScope` plus re-resolve imports; update cross-file call graph edges |
| `FullRebuild` | Rebuild entire CPG (should be rare) |

**Invalidation strategy:**
1. When a file changes, identify affected CPG nodes via `file_members` map
2. Remove those nodes and all their edges from the CPG graph
3. Re-extract CPG nodes from the new AST
4. Re-build CFG for affected functions
5. Re-run data flow for affected functions
6. Re-connect inter-procedural edges (call graph, data flow arguments/returns)

**Performance target:** Incremental CPG update for a single function change should complete in <50ms for a file with <500 LOC.

---

## 6. Testing Strategy

### 6.1 Unit Tests (per module)

| Module | Test Focus |
|--------|-----------|
| `cpg.rs` | Node/edge creation, file-to-CPG extraction correctness |
| `cfg.rs` | CFG edge correctness for each control flow construct, per language |
| `dataflow.rs` | Reaching definitions accuracy, def-use chain correctness |
| `cpg_query.rs` | Query result correctness, reachability, taint tracking |

### 6.2 Integration Tests

- **Round-trip:** Parse → build CPG → modify source → incremental update → verify CPG consistency
- **Cross-file:** Build CPG for multi-file project → verify call graph crosses file boundaries
- **Consistency validation:** Extend `validate_consistency()` to check CPG invariants (every CpgNode has a parent File node, every CFG has entry/exit, etc.)

### 6.3 Golden Tests

For each supported language, maintain a set of known source files with expected:
- Number of CPG nodes by type
- Specific CFG edges
- Specific data flow edges
- Query results (dead functions, unused variables)

### 6.4 Benchmarks

Extend `rust_core/benches/incremental_benchmark.rs`:
- CPG construction time per 1000 LOC
- Incremental CPG update time (Local vs. FileScope)
- Query response time for reachability, taint tracking
- Memory usage per file in the CPG

---

## 7. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **Per-language effort explosion** | High | High | Start with Python only; defer others until patterns stabilize |
| **Memory growth** | Medium | Medium | Use `Arc<Tree>` sharing; consider on-demand CFG construction (build CFG only when queried) |
| **tree-sitter grammar limitations** | Medium | Low | Tree-sitter grammars for all 6 languages are mature; macro expansion (Rust) is the main gap |
| **Data flow precision** | Medium | Medium | Start with intra-procedural only; inter-procedural is optional enhancement |
| **Breaking existing functionality** | High | Low | Dual-graph architecture keeps `RepoGraph` unchanged; CPG is additive |
| **Dynamic dispatch unresolvable** | Low | High | Accept imprecision; mark unresolved calls rather than guessing |

---

## 8. Execution Order & Dependencies

```
Phase 1: AST Persistence + Fine-Grained Nodes
    │
    ├──→ Phase 2: Control Flow Graph (requires Phase 1 nodes)
    │       │
    │       └──→ Phase 3: Data Flow / PDG (requires Phase 2 CFG)
    │               │
    │               └──→ Phase 4: Inter-Procedural Analysis (requires Phase 3)
    │
    └──→ Phase 5: Query Interface (can start after Phase 1, iteratively extended)
```

Phases 1 and 5 can be developed in partial parallel. Phase 5 grows as each analysis phase adds new queryable information.

---

## 9. Success Criteria

| Metric | Target |
|--------|--------|
| CPG construction for 10K LOC Python project | < 2 seconds |
| Incremental update (single function change) | < 50ms |
| Memory overhead vs. file-only graph | < 3x |
| Taint query (source → sink, same file) | < 5ms |
| Taint query (cross-file, 5-hop) | < 50ms |
| Dead function detection accuracy | > 90% precision, > 80% recall |
| CFG correctness (golden tests) | 100% pass rate |
| No regressions in existing `RepoGraph` tests | 66/66 pass |
| No regressions in Python shell tests | All pass |

---

## 10. Future Extensions (Post-CPG)

These capabilities become possible once the CPG is operational:

1. **Security scanning:** Pattern-based vulnerability detection (SQL injection, XSS, command injection) using taint analysis
2. **Automated refactoring:** Safe rename, extract function, inline variable — guided by data flow and call graph
3. **Impact analysis:** "If I change this function's signature, what breaks?" — answered by the call graph + data flow
4. **Code review assistance:** Highlight data flow paths through changed code to help reviewers understand impact
5. **Test coverage mapping:** Map test functions to production code via call graph to identify untested code paths
6. **Dependency visualization:** Interactive graph exploration via the Python/CLI layer
