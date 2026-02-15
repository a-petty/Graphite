# Implementation Plan: Phase 4 - Dynamic Graph Mutation (Stage 5)

**Project:** Atlas - Semantic Repository Analysis Engine  
**Phase:** 4 (Additional Graph Operations)  
**Status:** Final Implementation Plan  
**Version:** 2.1 (Revised with Clarifications)  
**Last Updated:** 2024

---

## Revision History

**Version 2.1 (Current)** - Added Critical Clarifications
- Added comprehensive "Method Distinction" section explaining `add_file()` vs `update_file()` use cases
- Expanded "Error Type System & Exception Hierarchy" with specific Python exception types
- Added detailed "Symbol Index Updates" section with future-proofing strategy for NodeIndex caching
- Updated Python bindings to use automatic error mapping via `From<GraphError> for PyErr`
- Enhanced agent event handlers with granular exception handling examples
- Added error handling philosophy and decision matrix
- Expanded error handling reference with common scenarios

**Version 2.0** - Optimizations and Safety
- Added Unresolved Imports Index for O(1) edge resolution
- Implemented Prepare-then-Commit pattern for transactional safety
- Added bidirectional edge downgrading for self-healing graph
- Enforced path canonicalization requirements

**Version 1.0** - Initial Plan
- Basic add_file and remove_file implementation
- Swap-remove handling strategy
- Initial testing framework


## Executive Summary

This document provides the comprehensive implementation plan for adding dynamic mutation capabilities to the `RepoGraph`. The core challenge is maintaining consistency between `petgraph`'s internal node indices and our `path_to_idx` mapping when the underlying graph structure changes, particularly during node removal where `petgraph` uses a "swap-remove" strategy that shifts indices.

**Key Innovations:**
- **Unresolved Imports Index**: O(1) edge resolution instead of O(N) graph scanning
- **Prepare-then-Commit Pattern**: Transactional safety for mutation operations
- **Bidirectional Edge Downgrading**: Maintains graph integrity when dependencies are removed
- **Symbol Index Remapping**: Ensures symbol lookups remain valid after index shifts

---

## Context & Objectives

### Background

The current `RepoGraph` implementation supports:
- Initial graph construction via `build_complete()`
- Incremental updates via `update_file()` (preserves node, updates edges)

**Gap:** No support for:
- Adding new files discovered after initialization
- Removing deleted files from the graph
- Re-establishing edges when deleted files are restored

### Critical Constraint: The Swap-Remove Problem

`petgraph::DiGraph` uses swap-remove for O(1) node deletion:
```rust
// When removing node at index 5 from a graph with 10 nodes:
graph.remove_node(NodeIndex::new(5));
// Result: Node at index 9 is MOVED to index 5
// Our path_to_idx map now points to the wrong node!
```

**Impact:** Without proper handling, our `HashMap<PathBuf, NodeIndex>` becomes corrupt, causing:
- Lookup failures for files that should exist
- Edge creation between wrong nodes
- Symbol index pointing to incorrect file locations

---

## Technical Design Decisions

### 1. Optimization: Unresolved Imports Index

**Problem:** Original plan scanned entire graph (O(N)) when adding files to find reverse dependencies.

**Solution:** Maintain a forward-looking registry:
```rust
pub unresolved_imports: HashMap<String, HashSet<NodeIndex>>
// Example: {"utils.py" -> {idx_3, idx_7, idx_12}}
// Meaning: Files at indices 3, 7, and 12 are waiting for "utils.py"
```

**Benefits:**
- O(1) lookup when adding a file that resolves pending imports
- Automatic reconnection when deleted files are restored
- Memory overhead is minimal (only stores pending links)

### 2. Safety: Prepare-then-Commit Pattern

**Problem:** Mutations that partially fail leave the graph in an inconsistent state.

**Solution:** Two-phase operations:
1. **Prepare (Failable)**: Validate inputs, calculate consequences, return errors
2. **Commit (Infallible)**: Execute mutations that cannot fail (barring panics)

**Example:**
```rust
pub fn remove_file(&mut self, path: &str) -> Result<(), GraphError> {
    // PREPARE: All operations that can fail
    let target_idx = self.path_to_idx.get(path)
        .ok_or(GraphError::NodeNotFound(path.into()))?;
    let last_idx = NodeIndex::new(self.graph.node_count() - 1);
    let will_swap = target_idx != last_idx;
    let moved_path = if will_swap {
        Some(self.graph[last_idx].path.clone())
    } else {
        None
    };
    
    // COMMIT: Infallible mutations
    self.execute_removal(target_idx, moved_path);
    Ok(())
}
```

### 3. Consistency: Bidirectional Edge Management

**Problem:** When file B is deleted, files importing B have dangling edges.

**Solution:** Downgrade edges to unresolved state:
1. On removal of B: Move importers of B into `unresolved_imports["B"]`
2. If B is re-added: Automatically reconnect waiting nodes
3. Result: Graph self-heals on file restoration (e.g., git checkout)

### 4. Path Normalization: Single Source of Truth

**Requirement:** All paths MUST be canonical and project-root-relative before reaching Rust.

**Enforcement:**
- Python: `Path.resolve()` → canonicalize symlinks
- Python: Convert to relative path from project root
- Rust: Accept only `String` (not `&Path`) to force explicit conversion
- Validation: Reject paths containing `..` (outside project root)

### 5. Method Distinction: `add_file()` vs `update_file()`

**Critical Design Choice:** These methods serve different purposes and should not be used interchangeably.

#### `update_file()` - Incremental, In-Place Update

**Purpose:** Handle file modifications during normal development workflow

**Behavior:**
- Preserves existing NodeIndex (node stays in same position)
- Reparses content to extract new definitions/usages
- Calculates differential: symbols added vs removed
- Updates edges incrementally (only touches changed edges)
- Uses change tier classification (Local/Import/Semantic/Structural)
- Skips PageRank recalculation for Local-tier changes

**Use Cases:**
- File watcher detects modification (user saves in editor)
- Continuous development: edit → save → update graph
- Accounts for 95% of graph mutations during development
- When file identity is preserved (same path, modified content)

**Performance:** Optimized via change classification - minimal graph churn

#### `add_file()` - Destructive Upsert

**Purpose:** Add genuinely new files OR perform clean rebuild of existing files

**Behavior:**
- If path already exists: calls `remove_file()` first (nuclear option)
- Creates brand new NodeIndex (node gets new position)
- Parses from scratch with no historical context
- Builds all edges fresh (no incremental logic)
- No differential calculation
- Always marks PageRank as dirty

**Use Cases:**
- File watcher detects file creation (new file appears)
- Git operations: `checkout`, `merge` (files appear/disappear/change drastically)
- Agent restart: rebuilding graph from disk
- Forcing clean state after suspected corruption
- Files that were deleted from graph and need re-addition

**Performance:** Slower than `update_file` - full teardown and rebuild

#### Decision Matrix

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| User saves file in editor | `update_file()` | Incremental, preserves history |
| New file created by user | `add_file()` | Ensures clean initial state |
| File restored from git | `add_file()` | Content may have changed significantly |
| File renamed (detected as delete+create) | `add_file()` | Treat as genuinely new file |
| Forcing graph consistency | `add_file()` | Nuclear option guarantees correctness |
| Batch import during initialization | `build_complete()` | Optimized for bulk operations |
| File modified by external tool | `update_file()` | Likely incremental changes |

#### Agent Implementation Pattern

```python
def _handle_file_modified(self, file_path: Path):
    """Existing file was modified - use incremental update"""
    try:
        result = self.repo_graph.update_file(str(canonical_path), content)
        # Handle based on result.needs_pagerank_recalc
    except GraphError as e:
        if "NodeNotFound" in str(e):
            # File wasn't in graph - treat as creation instead
            log.warning(f"File {file_path.name} not found, adding as new")
            self._handle_file_created(file_path)
        else:
            raise

def _handle_file_created(self, file_path: Path):
    """New file appeared - use add_file with automatic upsert"""
    # add_file internally handles the case where file already exists
    self.repo_graph.add_file(normalized_path, content)
```

**Key Insight:** The upsert behavior in `add_file()` (calling `remove_file()` first if path exists) prevents edge case bugs where:
- File was deleted but graph wasn't notified
- File is now being recreated
- Without upsert, we'd have duplicate/stale state

This design prioritizes **correctness** (`add_file`) vs **efficiency** (`update_file`).

---

## Error Type System & Exception Hierarchy

### Rust Error Types (`graph.rs`)

**Design Goal:** Provide specific, actionable error types that Python can handle granularly.

```rust
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("File not found in graph: {0}")]
    NodeNotFound(PathBuf),
    
    #[error("Failed to parse {0}: syntax error or unsupported language")]
    ParseError(PathBuf),
    
    #[error("I/O error: {0}")]
    IoError(String),
}

impl From<std::io::Error> for GraphError {
    fn from(err: std::io::Error) -> Self {
        GraphError::IoError(err.to_string())
    }
}
```

### Python Exception Hierarchy (`lib.rs`)

**Design Goal:** Create a type hierarchy that allows granular exception handling in Python.

```rust
use pyo3::create_exception;
use pyo3::exceptions::PyException;

// Base exception
create_exception!(semantic_engine, GraphError, PyException);

// Specific exceptions inherit from GraphError
create_exception!(semantic_engine, ParseError, GraphError);
create_exception!(semantic_engine, NodeNotFoundError, GraphError);
```

**Exception Tree:**
```
PyException (Python built-in)
    └── GraphError (our base - catch-all for graph operations)
            ├── ParseError (syntax errors in source files)
            ├── NodeNotFoundError (file path not in graph)
            └── (future extensions: ImportResolutionError, etc.)
```

### Rust-to-Python Error Mapping

```rust
impl From<graph::GraphError> for PyErr {
    fn from(err: graph::GraphError) -> PyErr {
        match err {
            graph::GraphError::ParseError(path) => {
                ParseError::new_err(format!(
                    "Syntax error in {}: unable to parse file", 
                    path.display()
                ))
            }
            graph::GraphError::NodeNotFound(path) => {
                NodeNotFoundError::new_err(format!(
                    "File not found in graph: {}", 
                    path.display()
                ))
            }
            graph::GraphError::IoError(msg) => {
                GraphError::new_err(format!("I/O error: {}", msg))
            }
        }
    }
}
```

### Exception Usage in Python

```python
from semantic_engine import GraphError, ParseError, NodeNotFoundError

try:
    graph.add_file("broken.py", "def invalid syntax")
except ParseError as e:
    # File has syntax errors - non-fatal, user needs to fix
    log.warning(f"Syntax error, file skipped: {e}")
except NodeNotFoundError as e:
    # Shouldn't happen in add_file, but defensive
    log.error(f"Unexpected missing node: {e}")
except GraphError as e:
    # Catch-all for other graph errors
    log.error(f"Graph operation failed: {e}")
```

### Benefits of This Hierarchy

1. **Specific Error Handling**: Agent can respond differently to each error type
2. **User Experience**: Clear error messages indicate what went wrong
3. **Type Safety**: Python type checkers (mypy) can verify exception handling
4. **Debugging**: Stack traces immediately identify error category
5. **Monitoring**: Can track ParseError rate separately (code quality metric)
6. **Testing**: Tests can assert on specific exception types

### Error Handling Decision Matrix

| Error Type | Severity | Agent Response | User Notification |
|------------|----------|----------------|-------------------|
| `ParseError` | Warning | Skip file, continue | "Syntax error in X, fix to include in graph" |
| `NodeNotFoundError` | Info | Log debug message | None (expected for ignored files) |
| `GraphError` (generic) | Error | Log error, continue | "Internal error processing X" |
| Uncaught Exception | Critical | Log with traceback, may crash | "Critical agent error, check logs" |

---

## 1. Rust Core Implementation

### 1.1. Data Structure Updates (`graph.rs`)

```rust
pub struct RepoGraph {
    graph: DiGraph<FileNode, EdgeKind>,
    path_to_idx: HashMap<PathBuf, NodeIndex>,
    symbol_index: SymbolIndex,
    import_resolver: Option<ImportResolver>,
    pagerank_dirty: bool,
    
    /// NEW: Maps import target paths to nodes waiting for them
    /// Example: {"utils/logger.py" -> {node_5, node_8}}
    pub unresolved_imports: HashMap<PathBuf, HashSet<NodeIndex>>,
}

impl RepoGraph {
    pub fn new_with_import_resolution(project_root: &Path) -> Self {
        Self {
            // ... existing fields ...
            unresolved_imports: HashMap::new(),
        }
    }
}
```

### 1.2. Method: `add_file` (Full Implementation)

```rust
pub fn add_file(&mut self, path: PathBuf, content: &str) -> Result<(), GraphError> {
    // 1. UPSERT PROTECTION
    // If file exists, remove it first to ensure clean state
    if self.path_to_idx.contains_key(&path) {
        self.remove_file(&path)?;
    }
    
    // 2. PARSING
    let lang = SupportedLanguage::from_path(&path);
    if lang == SupportedLanguage::Unknown {
        return Ok(()); // Silently skip unsupported files
    }
    
    let mut parser = TreeSitterParser::new();
    parser.set_language(lang.get_parser()
        .ok_or_else(|| GraphError::ParseError(path.clone()))?)?;
    
    let tree = parser.parse(content, None)
        .ok_or_else(|| GraphError::ParseError(path.clone()))?;
    
    let (defs, uses) = SymbolHarvester::harvest(&tree, content, lang);
    let imports = parse_imports_from_tree(
        &tree, 
        content, 
        self.import_resolver.as_ref(), 
        &path, 
        lang
    );
    
    // 3. NODE CREATION
    let file_node = FileNode::new(
        path.clone(),
        defs.clone(),
        uses.clone(),
        &imports,
        content,
    );
    
    let new_idx = self.graph.add_node(file_node);
    self.path_to_idx.insert(path.clone(), new_idx);
    
    // 4. SYMBOL INDEX REGISTRATION
    for def in &defs {
        self.symbol_index
            .definitions
            .entry(def.clone())
            .or_default()
            .push(path.clone());
    }
    
    for usage in &uses {
        self.symbol_index
            .users
            .entry(usage.clone())
            .or_default()
            .push(path.clone());
    }
    
    self.symbol_index.usages.insert(path.clone(), uses.clone());
    
    // 5. OUTGOING EDGES (Dependencies)
    for target_path in &imports {
        if let Some(&target_idx) = self.path_to_idx.get(target_path) {
            // Target exists - create edge immediately
            if new_idx != target_idx {
                self.ensure_edge(new_idx, target_idx, EdgeKind::Import);
            }
        } else {
            // Target doesn't exist yet - register as unresolved
            self.unresolved_imports
                .entry(target_path.clone())
                .or_default()
                .insert(new_idx);
        }
    }
    
    // 6. INCOMING EDGES (Resolution - OPTIMIZED)
    // Check if any nodes were waiting for THIS file
    if let Some(waiting_nodes) = self.unresolved_imports.remove(&path) {
        for waiting_idx in waiting_nodes {
            if waiting_idx != new_idx {
                self.ensure_edge(waiting_idx, new_idx, EdgeKind::Import);
            }
        }
    }
    
    // 7. SYMBOL USAGE EDGES
    // Create edges from this file to files defining symbols we use
    for usage in &uses {
        if let Some(defining_paths) = self.symbol_index.definitions.get(usage) {
            for def_path in defining_paths {
                if let Some(&def_idx) = self.path_to_idx.get(def_path) {
                    if new_idx != def_idx {
                        self.ensure_edge(new_idx, def_idx, EdgeKind::SymbolUsage);
                    }
                }
            }
        }
    }
    
    // 8. REVERSE SYMBOL USAGE EDGES
    // Create edges FROM files that use symbols WE define
    for def in &defs {
        if let Some(using_paths) = self.symbol_index.users.get(def) {
            for user_path in using_paths {
                if let Some(&user_idx) = self.path_to_idx.get(user_path) {
                    if user_idx != new_idx && user_path != &path {
                        self.ensure_edge(user_idx, new_idx, EdgeKind::SymbolUsage);
                    }
                }
            }
        }
    }
    
    // 9. FLAG UPDATE
    self.pagerank_dirty = true;
    
    Ok(())
}
```

### 1.3. Method: `remove_file` (Critical Path with Swap Handling)

```rust
pub fn remove_file(&mut self, path: &Path) -> Result<(), GraphError> {
    // ===== PHASE 1: PREPARE (Read-Only, Failable) =====
    
    // 1.1. Resolve path to index
    let target_idx = *self.path_to_idx.get(path)
        .ok_or_else(|| GraphError::NodeNotFound(path.to_path_buf()))?;
    
    // 1.2. Identify swap consequences
    let last_idx = NodeIndex::new(self.graph.node_count() - 1);
    let will_swap = target_idx != last_idx;
    
    // 1.3. Capture the path that will be moved (if swap occurs)
    let moved_path = if will_swap {
        Some(self.graph[last_idx].path.clone())
    } else {
        None
    };
    
    // 1.4. Identify incoming import edges (for downgrading)
    let incoming_importers: Vec<PathBuf> = self.graph
        .edges_directed(target_idx, petgraph::Direction::Incoming)
        .filter(|e| *e.weight() == EdgeKind::Import)
        .map(|e| self.graph[e.source()].path.clone())
        .collect();
    
    // ===== PHASE 2: COMMIT (Infallible Mutations) =====
    
    // 2.1. DOWNGRADE EDGES TO UNRESOLVED
    // Files that imported this file go back to "waiting" state
    for importer_path in incoming_importers {
        self.unresolved_imports
            .entry(path.to_path_buf())
            .or_default()
            .insert(self.path_to_idx[&importer_path]);
    }
    
    // 2.2. SYMBOL CLEANUP
    // Remove all definitions this file provided
    let node_defs = self.graph[target_idx].definitions.clone();
    for def in &node_defs {
        if let Some(paths) = self.symbol_index.definitions.get_mut(def) {
            paths.retain(|p| p != path);
            if paths.is_empty() {
                self.symbol_index.definitions.remove(def);
            }
        }
        // Also clean up users list
        if let Some(users) = self.symbol_index.users.get_mut(def) {
            users.retain(|p| p != path);
            if users.is_empty() {
                self.symbol_index.users.remove(def);
            }
        }
    }
    
    // Remove this file's usages
    self.symbol_index.usages.remove(path);
    
    // 2.3. GRAPH MUTATION
    // This is where the swap happens!
    self.graph.remove_node(target_idx);
    
    // 2.4. PATH MAP UPDATES
    // Remove the deleted file
    self.path_to_idx.remove(path);
    
    // If a swap occurred, update the map for the moved node
    if let Some(moved_path) = moved_path {
        // The node that WAS at last_idx is NOW at target_idx
        self.path_to_idx.insert(moved_path.clone(), target_idx);
        
        // 2.5. SYMBOL INDEX REMAP
        // Update all symbol index entries pointing to last_idx to point to target_idx
        self.remap_symbol_index(last_idx, target_idx);
        
        // 2.6. UNRESOLVED IMPORTS REMAP
        // If the moved node was waiting for imports, update its index
        for waiting_set in self.unresolved_imports.values_mut() {
            if waiting_set.remove(&last_idx) {
                waiting_set.insert(target_idx);
            }
        }
    }
    
    // 2.7. FLAG UPDATE
    self.pagerank_dirty = true;
    
    Ok(())
}

/// Helper: Remap all symbol index entries from old_idx to new_idx
fn remap_symbol_index(&mut self, old_idx: NodeIndex, new_idx: NodeIndex) {
    // CURRENT IMPLEMENTATION: The existing SymbolIndex uses PathBuf as keys exclusively,
    // so no NodeIndex-based remapping is required. The path stays the same when a node
    // moves positions, and the path_to_idx update above handles all lookups correctly.
    
    // FUTURE-PROOFING: If we add NodeIndex-based caching for performance (O(1) lookups
    // without path resolution), this function will need to remap those cached entries.
    // The interface is reserved now to ensure API stability.
    
    // Example future implementation (if we add node_to_symbols cache):
    // if let Some(symbols) = self.symbol_index.node_to_definitions.remove(&old_idx) {
    //     self.symbol_index.node_to_definitions.insert(new_idx, symbols);
    // }
    // if let Some(symbols) = self.symbol_index.node_to_usages.remove(&old_idx) {
    //     self.symbol_index.node_to_usages.insert(new_idx, symbols);
    // }
}
```

### 1.4. Symbol Index Updates (`symbol_index.rs`)

**Current Structure:**
```rust
pub struct SymbolIndex {
    /// Maps symbol names to the files that define them
    pub definitions: HashMap<String, Vec<PathBuf>>,
    
    /// Maps symbol names to the files that use them
    pub users: HashMap<String, Vec<PathBuf>>,
    
    /// Maps file paths to the symbols they use
    pub usages: HashMap<PathBuf, Vec<String>>,
}
```

**Analysis:** The current implementation uses PathBuf exclusively, meaning NodeIndex changes during swap-remove don't affect it. However, we need cleanup methods.

**Required New Methods:**

```rust
impl SymbolIndex {
    /// Remove all references to a specific file
    /// Called when a file is deleted from the graph
    pub fn remove_node_references(&mut self, path: &Path) {
        // Remove from definitions (this file no longer provides symbols)
        for symbol_paths in self.definitions.values_mut() {
            symbol_paths.retain(|p| p != path);
        }
        
        // Remove from users (this file no longer uses symbols)
        for user_paths in self.users.values_mut() {
            user_paths.retain(|p| p != path);
        }
        
        // Remove from usages (this file's usage list is gone)
        self.usages.remove(path);
        
        // Clean up empty entries to prevent memory leaks
        self.definitions.retain(|_, paths| !paths.is_empty());
        self.users.retain(|_, paths| !paths.is_empty());
    }
    
    /// Remap a node index (currently no-op, reserved for future optimization)
    /// 
    /// FUTURE USE: If we add NodeIndex-based caching for O(1) symbol lookups,
    /// this method will update those cached mappings after a swap-remove.
    /// 
    /// Example future field:
    ///   node_to_definitions: HashMap<NodeIndex, Vec<String>>
    ///   node_to_usages: HashMap<NodeIndex, Vec<String>>
    /// 
    /// This would allow: graph.symbol_index.get_definitions(node_idx)
    /// Instead of:       graph.symbol_index.get_definitions(graph[node_idx].path)
    pub fn remap_node_index(&mut self, _old_idx: NodeIndex, _new_idx: NodeIndex) {
        // Currently a no-op because we use PathBuf keys
        // Paths don't change when nodes swap positions
        
        // FUTURE IMPLEMENTATION (when adding NodeIndex caching):
        // if let Some(defs) = self.node_to_definitions.remove(&old_idx) {
        //     self.node_to_definitions.insert(new_idx, defs);
        // }
        // if let Some(uses) = self.node_to_usages.remove(&old_idx) {
        //     self.node_to_usages.insert(new_idx, uses);
        // }
    }
}
```

**Future-Proofing Strategy:**

The interface methods (`remove_node_references` and `remap_node_index`) are designed to:
1. **Work correctly now** with the PathBuf-based implementation
2. **Remain stable** when we add NodeIndex caching for performance
3. **Prevent breaking changes** in downstream code

**Hypothetical Future Optimization:**
```rust
// If profiling shows symbol lookups are a bottleneck, we could add:
pub struct SymbolIndex {
    // ... existing fields ...
    
    // NEW: Direct node → symbols mapping (bypasses path resolution)
    node_to_definitions: HashMap<NodeIndex, Vec<String>>,
    node_to_usages: HashMap<NodeIndex, Vec<String>>,
}
```

Then the `remap_node_index` implementation becomes critical for maintaining correctness during swap-remove operations.

**Design Principle:** Reserve the interface now, implement as needed. This prevents API churn and ensures the swap-remove logic is complete even if the optimization is never added.

### 1.5. Helper Methods

```rust
impl RepoGraph {
    /// Ensures an edge exists between two nodes
    fn ensure_edge(&mut self, from: NodeIndex, to: NodeIndex, kind: EdgeKind) {
        // Check if edge already exists
        let edge_exists = self.graph
            .edges_connecting(from, to)
            .any(|e| *e.weight() == kind);
        
        if !edge_exists {
            self.graph.add_edge(from, to, kind);
        }
    }
    
    /// Validation helper for testing
    #[cfg(test)]
    pub fn validate_consistency(&self) -> Result<(), String> {
        // 1. Size check
        if self.path_to_idx.len() != self.graph.node_count() {
            return Err(format!(
                "Size mismatch: path_to_idx has {} entries, graph has {} nodes",
                self.path_to_idx.len(),
                self.graph.node_count()
            ));
        }
        
        // 2. Mapping integrity
        for (path, &idx) in &self.path_to_idx {
            let node = self.graph.node_weight(idx)
                .ok_or_else(|| format!(
                    "path_to_idx points to invalid index {:?} for path '{}'",
                    idx, path.display()
                ))?;
            
            if node.path != *path {
                return Err(format!(
                    "Mismatch at index {:?}: path_to_idx expects '{}', but node contains '{}'",
                    idx, path.display(), node.path.display()
                ));
            }
        }
        
        // 3. Reverse check - all graph nodes have map entries
        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            if !self.path_to_idx.contains_key(&node.path) {
                return Err(format!(
                    "Node at index {:?} with path '{}' has no path_to_idx entry",
                    idx, node.path.display()
                ));
            }
        }
        
        Ok(())
    }
}
```

---

## 2. Python Bindings (`lib.rs`)

### 2.1. Exception Module Setup

First, ensure the exception types are properly exposed:

```rust
#[pymodule]
fn semantic_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_repository, m)?)?;
    m.add_class::<PyRepoGraph>()?;
    m.add_class::<PyFileWatcher>()?;
    m.add_class::<PyGraphUpdateResult>()?;
    m.add_class::<PyFileChangeEvent>()?;
    m.add_class::<PyWatcherStats>()?;
    
    // Add exception types (defined in Error Type System section above)
    m.add("GraphError", _py.get_type::<GraphError>())?;
    m.add("ParseError", _py.get_type::<ParseError>())?;
    m.add("NodeNotFoundError", _py.get_type::<NodeNotFoundError>())?;
    
    Ok(())
}
```

### 2.2. PyO3 Method Implementations

```rust
#[pymethods]
impl PyRepoGraph {
    /// Add or update a file in the graph.
    /// 
    /// If the file already exists in the graph, it is first removed and then re-added
    /// with the new content (destructive upsert). This ensures a clean state.
    /// 
    /// Args:
    ///     path: Project-root-relative path (must be canonical)
    ///     content: File content as string
    /// 
    /// Raises:
    ///     ParseError: If file has syntax errors
    ///     GraphError: On other graph operation failures
    fn add_file(&mut self, path: String, content: String) -> PyResult<()> {
        let path_buf = PathBuf::from(path);
        self.graph.add_file(path_buf, &content)
            .map_err(|e| e.into()) // Uses From<GraphError> for PyErr (automatic mapping)
    }
    
    /// Remove a file from the graph.
    /// 
    /// This removes the file node and all associated edges. If other files were
    /// importing this file, they will be downgraded to "unresolved import" status
    /// and will automatically reconnect if this file is re-added later.
    /// 
    /// Args:
    ///     path: Project-root-relative path (must match stored path exactly)
    /// 
    /// Raises:
    ///     NodeNotFoundError: If file is not in the graph
    ///     GraphError: On other graph operation failures
    fn remove_file(&mut self, path: String) -> PyResult<()> {
        let path_buf = PathBuf::from(path);
        self.graph.remove_file(&path_buf)
            .map_err(|e| e.into()) // Uses From<GraphError> for PyErr (automatic mapping)
    }
}
```

**Key Design Point:** By using `.map_err(|e| e.into())`, we rely on the `From<graph::GraphError> for PyErr` implementation defined in the Error Type System section. This provides:
- Automatic, type-safe error conversion
- No need to match on error variants in bindings layer
- Centralized error mapping logic
- Easy to extend with new error types

---

## 3. Python Agent Integration (`agent.py`)

### 3.1. Path Normalization Helper

```python
class AtlasAgent:
    def __init__(self, project_root: Path, use_real_llm: bool = False):
        self.config = AgentConfig(project_root=project_root)
        # Store canonical project root
        self.project_root_canonical = project_root.resolve()
        # ... rest of initialization ...
    
    def _normalize_path(self, file_path: Path) -> str:
        """
        Convert a file path to canonical, project-root-relative format.
        
        This ensures consistency with how paths are stored in the graph:
        1. Resolve symlinks (e.g., /var -> /private/var)
        2. Make absolute
        3. Convert to relative path from project root
        4. Validate it's within project bounds
        
        Args:
            file_path: Any path (absolute, relative, symlinked)
        
        Returns:
            Project-root-relative path as string
        
        Raises:
            ValueError: If path is outside project root
        """
        # Step 1: Canonicalize (resolve symlinks, make absolute)
        canonical = file_path.resolve()
        
        # Step 2: Convert to relative path
        try:
            relative = canonical.relative_to(self.project_root_canonical)
        except ValueError:
            raise ValueError(
                f"Path {canonical} is outside project root {self.project_root_canonical}"
            )
        
        # Step 3: Convert to string with forward slashes (cross-platform)
        return str(relative).replace('\\', '/')
```

### 3.2. Updated Event Handlers with Granular Error Handling

```python
from semantic_engine import GraphError, ParseError, NodeNotFoundError

def _handle_file_created(self, file_path: Path):
    """
    Handle file creation by adding it to the graph.
    
    Uses granular exception handling to respond appropriately to different error types:
    - ParseError: File has syntax errors (non-fatal, skip for now)
    - ValueError: Path outside project (ignore)
    - Other errors: Log and continue
    """
    content = self._safe_read_file(file_path)
    if content is None:
        return
    
    try:
        normalized_path = self._normalize_path(file_path)
        self.repo_graph.add_file(normalized_path, content)
        console.print(f"  ↳ [green]Added to graph[/green]: {file_path.name}")
        
    except ParseError as e:
        # Non-fatal: File has syntax errors, can't be parsed
        # This is expected during development when files have temporary errors
        log.warning(f"  ↳ [yellow]Syntax error in {file_path.name}[/yellow]")
        log.warning(f"    File will be skipped until errors are fixed")
        log.debug(f"    Parse error details: {e}")
        # Future enhancement: Could track these files and retry when they're modified
        
    except ValueError as e:
        # Path is outside project root - silently ignore
        # This can happen if the watcher picks up system files or parent directories
        log.debug(f"  ↳ Skipped file outside project: {e}")
        
    except NodeNotFoundError as e:
        # Shouldn't happen in add_file (it does upsert), but be defensive
        log.error(f"  ✗ Unexpected NodeNotFoundError in add_file: {e}")
        
    except GraphError as e:
        # Catch-all for other graph errors (future error types)
        log.error(f"  ✗ Graph error adding {file_path.name}: {e}")
        
    except Exception as e:
        # Non-graph errors (file system issues, bugs, etc.)
        log.error(f"  ✗ Unexpected error adding {file_path.name}: {e}", exc_info=True)

def _handle_file_modified(self, file_path: Path):
    """
    Handle file modification using incremental update.
    
    Falls back to add_file if the file isn't in the graph yet (edge case).
    """
    content = self._safe_read_file(file_path)
    if content is None:
        return
    
    try:
        canonical_path = file_path.resolve()
        result = self.repo_graph.update_file(str(canonical_path), content)
        
        if result.needs_pagerank_recalc:
            console.print(
                f"  ↳ [green]Graph Updated[/green] "
                f"(+{result.edges_added} / -{result.edges_removed} edges)"
            )
        else:
            console.print("  ↳ [dim]Content updated (structure unchanged)[/dim]")
            
    except ParseError as e:
        # Syntax error introduced by modification
        log.warning(f"  ↳ [yellow]Syntax error after modification: {file_path.name}[/yellow]")
        log.warning(f"    Graph state for this file will be stale until fixed")
        log.debug(f"    Parse error details: {e}")
        
    except NodeNotFoundError as e:
        # File not in graph - this is unusual for a modification event
        # Treat it as a creation instead
        log.warning(f"  ↳ File {file_path.name} not in graph, adding as new")
        self._handle_file_created(file_path)
        
    except GraphError as e:
        log.error(f"  ✗ Graph error updating {file_path.name}: {e}")
        
    except Exception as e:
        log.error(f"  ✗ Unexpected error updating {file_path.name}: {e}", exc_info=True)

def _handle_file_deleted(self, file_path: Path):
    """
    Handle file deletion by removing it from the graph.
    
    NodeNotFoundError is expected and normal - many files are never in the graph
    (ignored files, binaries, etc.).
    """
    try:
        normalized_path = self._normalize_path(file_path)
        self.repo_graph.remove_file(normalized_path)
        console.print(f"  ↳ [red]Removed from graph[/red]: {file_path.name}")
        
    except NodeNotFoundError:
        # Expected: File was never tracked (gitignored, binary, outside project, etc.)
        # This is normal and not an error - just log at debug level
        log.debug(f"  ↳ File not in graph: {file_path.name}")
        
    except ValueError as e:
        # Path outside project root - ignore
        log.debug(f"  ↳ Ignored deletion outside project: {e}")
        
    except GraphError as e:
        # Other graph errors - shouldn't happen in remove_file, but log if they do
        log.error(f"  ✗ Graph error removing {file_path.name}: {e}")
        
    except Exception as e:
        # Non-graph errors
        log.error(f"  ✗ Unexpected error removing {file_path.name}: {e}", exc_info=True)
```

**Error Handling Philosophy:**

1. **ParseError**: Graceful degradation
   - Log warning (not error)
   - Continue watching
   - File will be automatically added when syntax is fixed

2. **NodeNotFoundError**: Context-dependent
   - In `add_file`: Unexpected, log error
   - In `update_file`: Fall back to `add_file`
   - In `remove_file`: Normal, log debug

3. **GraphError**: Catch-all
   - Log error
   - Continue operation
   - Allows for future error types

4. **ValueError**: Path validation
   - Silent ignore (debug log only)
   - Prevents pollution from system files

5. **Exception**: Unknown errors
   - Full stack trace
   - May indicate bugs


---

## 4. Testing Strategy

### 4.1. Test File Structure

**Location:** `rust_core/tests/test_graph_mutation.rs`

```rust
#[cfg(test)]
mod graph_mutation_tests {
    use super::*;
    use std::path::PathBuf;
    
    fn create_test_graph() -> RepoGraph {
        RepoGraph::new_with_import_resolution(Path::new("/test"))
    }
    
    // ... tests follow ...
}
```

### 4.2. Test Case 1: Basic Add/Remove

```rust
#[test]
fn test_add_remove_basic() {
    let mut graph = create_test_graph();
    
    // Add a file
    let path = PathBuf::from("test.py");
    let content = "def hello(): pass";
    graph.add_file(path.clone(), content).unwrap();
    
    // Verify it exists
    assert!(graph.path_to_idx.contains_key(&path));
    assert_eq!(graph.graph.node_count(), 1);
    graph.validate_consistency().unwrap();
    
    // Remove it
    graph.remove_file(&path).unwrap();
    
    // Verify it's gone
    assert!(!graph.path_to_idx.contains_key(&path));
    assert_eq!(graph.graph.node_count(), 0);
    graph.validate_consistency().unwrap();
}
```

### 4.3. Test Case 2: Swap-Remove Index Integrity

```rust
#[test]
fn test_swap_remove_index_integrity() {
    let mut graph = create_test_graph();
    
    // Add 3 files (indices: 0, 1, 2)
    let a = PathBuf::from("A.py");
    let b = PathBuf::from("B.py");
    let c = PathBuf::from("C.py");
    
    graph.add_file(a.clone(), "# A").unwrap();
    graph.add_file(b.clone(), "# B").unwrap();
    graph.add_file(c.clone(), "# C").unwrap();
    
    assert_eq!(graph.graph.node_count(), 3);
    graph.validate_consistency().unwrap();
    
    // Remove middle file (B at index 1)
    // This causes C (index 2) to swap into B's slot (index 1)
    graph.remove_file(&b).unwrap();
    
    // Verify B is gone
    assert!(!graph.path_to_idx.contains_key(&b));
    assert_eq!(graph.graph.node_count(), 2);
    
    // Verify C's index was updated
    let c_idx = graph.path_to_idx[&c];
    assert_eq!(c_idx.index(), 1, "C should have moved to index 1");
    
    // Verify the node at index 1 is actually C
    assert_eq!(graph.graph[c_idx].path, c);
    
    // Verify A is still at index 0
    let a_idx = graph.path_to_idx[&a];
    assert_eq!(a_idx.index(), 0);
    assert_eq!(graph.graph[a_idx].path, a);
    
    // Final consistency check
    graph.validate_consistency().unwrap();
}
```

### 4.4. Test Case 3: Unresolved Imports Connect Later

```rust
#[test]
fn test_unresolved_imports_resolution() {
    let mut graph = create_test_graph();
    
    // Add main.py which imports utils.py (doesn't exist yet)
    let main = PathBuf::from("main.py");
    let utils = PathBuf::from("utils.py");
    
    graph.add_file(main.clone(), "import utils").unwrap();
    
    // Verify main was added
    assert!(graph.path_to_idx.contains_key(&main));
    
    // Verify utils is in unresolved imports
    assert!(graph.unresolved_imports.contains_key(&utils));
    let waiting = &graph.unresolved_imports[&utils];
    assert!(waiting.contains(&graph.path_to_idx[&main]));
    
    // No edge should exist yet
    let main_idx = graph.path_to_idx[&main];
    assert_eq!(
        graph.graph.edges_directed(main_idx, petgraph::Direction::Outgoing).count(),
        0
    );
    
    // Now add utils.py
    graph.add_file(utils.clone(), "def helper(): pass").unwrap();
    
    // Verify edge was created
    let utils_idx = graph.path_to_idx[&utils];
    let has_edge = graph.graph
        .edges_connecting(main_idx, utils_idx)
        .any(|e| *e.weight() == EdgeKind::Import);
    assert!(has_edge, "Import edge should exist after resolution");
    
    // Verify utils is no longer in unresolved imports
    assert!(!graph.unresolved_imports.contains_key(&utils));
    
    graph.validate_consistency().unwrap();
}
```

### 4.5. Test Case 4: Edge Downgrade on Removal

```rust
#[test]
fn test_remove_downgrades_to_unresolved() {
    let mut graph = create_test_graph();
    
    // Setup: main.py -> utils.py
    let main = PathBuf::from("main.py");
    let utils = PathBuf::from("utils.py");
    
    graph.add_file(utils.clone(), "def helper(): pass").unwrap();
    graph.add_file(main.clone(), "import utils").unwrap();
    
    let main_idx = graph.path_to_idx[&main];
    let utils_idx = graph.path_to_idx[&utils];
    
    // Verify edge exists
    assert!(graph.graph.edges_connecting(main_idx, utils_idx).count() > 0);
    
    // Remove utils.py
    graph.remove_file(&utils).unwrap();
    
    // Verify utils is gone
    assert!(!graph.path_to_idx.contains_key(&utils));
    
    // Verify main is now in unresolved imports for utils
    assert!(graph.unresolved_imports.contains_key(&utils));
    let new_main_idx = graph.path_to_idx[&main]; // May have shifted
    assert!(graph.unresolved_imports[&utils].contains(&new_main_idx));
    
    // Re-add utils.py
    graph.add_file(utils.clone(), "def helper(): pass").unwrap();
    
    // Verify edge was re-established
    let final_main_idx = graph.path_to_idx[&main];
    let final_utils_idx = graph.path_to_idx[&utils];
    let has_edge = graph.graph
        .edges_connecting(final_main_idx, final_utils_idx)
        .any(|e| *e.weight() == EdgeKind::Import);
    assert!(has_edge, "Edge should be restored after re-adding file");
    
    // Verify unresolved imports is clean
    assert!(!graph.unresolved_imports.contains_key(&utils));
    
    graph.validate_consistency().unwrap();
}
```

### 4.6. Test Case 5: Symbol Usage Edges

```rust
#[test]
fn test_symbol_usage_edges() {
    let mut graph = create_test_graph();
    
    // Add a file with a definition
    let lib = PathBuf::from("lib.py");
    graph.add_file(lib.clone(), "def my_function(): pass").unwrap();
    
    // Add a file that uses the symbol
    let main = PathBuf::from("main.py");
    graph.add_file(main.clone(), "from lib import my_function\nmy_function()").unwrap();
    
    let main_idx = graph.path_to_idx[&main];
    let lib_idx = graph.path_to_idx[&lib];
    
    // Verify import edge exists
    let has_import = graph.graph
        .edges_connecting(main_idx, lib_idx)
        .any(|e| *e.weight() == EdgeKind::Import);
    assert!(has_import);
    
    // Verify symbol usage edge exists
    let has_usage = graph.graph
        .edges_connecting(main_idx, lib_idx)
        .any(|e| *e.weight() == EdgeKind::SymbolUsage);
    assert!(has_usage);
    
    graph.validate_consistency().unwrap();
}
```

### 4.7. Test Case 6: Upsert Behavior

```rust
#[test]
fn test_add_file_upsert() {
    let mut graph = create_test_graph();
    
    let path = PathBuf::from("test.py");
    
    // Add initial version
    graph.add_file(path.clone(), "def old(): pass").unwrap();
    assert_eq!(graph.graph.node_count(), 1);
    let initial_idx = graph.path_to_idx[&path];
    
    // Add again with new content (should update)
    graph.add_file(path.clone(), "def new(): pass").unwrap();
    
    // Should still have 1 node
    assert_eq!(graph.graph.node_count(), 1);
    
    // May have new index due to remove + add
    assert!(graph.path_to_idx.contains_key(&path));
    
    // Verify definitions updated
    let new_idx = graph.path_to_idx[&path];
    let node = &graph.graph[new_idx];
    assert!(node.definitions.contains(&"new".to_string()));
    assert!(!node.definitions.contains(&"old".to_string()));
    
    graph.validate_consistency().unwrap();
}
```

---

## 5. Integration Checklist

### 5.1. Pre-Implementation

- [ ] Review current `SymbolIndex` structure for compatibility
- [ ] Confirm `parse_imports_from_tree` returns `HashSet<PathBuf>`
- [ ] Verify `EdgeKind` enum includes `Import` and `SymbolUsage`
- [ ] Check if `FileNode` has `definitions` and `usages` fields

### 5.2. Implementation Order

1. [ ] Add `unresolved_imports` field to `RepoGraph`
2. [ ] Implement `ensure_edge` helper
3. [ ] Implement `add_file` method
4. [ ] Implement `remove_file` method with swap handling
5. [ ] Implement `validate_consistency` test helper
6. [ ] Add Python bindings in `lib.rs`
7. [ ] Add `_normalize_path` to Python agent
8. [ ] Update `_handle_file_created` handler
9. [ ] Update `_handle_file_deleted` handler

### 5.3. Testing Order

1. [ ] Run `test_add_remove_basic`
2. [ ] Run `test_swap_remove_index_integrity`
3. [ ] Run `test_unresolved_imports_resolution`
4. [ ] Run `test_remove_downgrades_to_unresolved`
5. [ ] Run `test_symbol_usage_edges`
6. [ ] Run `test_add_file_upsert`
7. [ ] Run full test suite: `cargo test`
8. [ ] Manual testing with agent watching a directory
9. [ ] Test file creation/deletion/modification cycle

### 5.4. Validation

- [ ] No panics during add/remove operations
- [ ] `validate_consistency()` passes after every operation
- [ ] Memory usage stable (no leaks in unresolved_imports)
- [ ] PageRank recalculation triggers correctly
- [ ] Agent logs show correct file operations
- [ ] Performance: add/remove should be O(E) where E = edges touched

---

## 6. Error Handling Reference

**Note:** This section provides a quick reference. For comprehensive error type documentation, see the "Error Type System & Exception Hierarchy" section at the beginning of this plan.

### Rust Error Types (`graph.rs`)

```rust
pub enum GraphError {
    NodeNotFound(PathBuf),      // File not in graph
    ParseError(PathBuf),         // Syntax error in file  
    IoError(String),             // File I/O failure
}
```

### Python Exception Hierarchy

```
PyException (Python built-in)
    └── GraphError (base exception for all graph operations)
            ├── ParseError (file has syntax errors)
            ├── NodeNotFoundError (file path not in graph)
            └── (future extensions)
```

### Exception Mapping

| Rust Error | Python Exception | When It Occurs | Severity |
|------------|------------------|----------------|----------|
| `GraphError::NodeNotFound` | `NodeNotFoundError` | File not found in graph | Info/Warning |
| `GraphError::ParseError` | `ParseError` | Invalid syntax in source file | Warning |
| `GraphError::IoError` | `GraphError` (base) | File I/O failure | Error |

### Agent Error Handling Patterns

#### Pattern 1: File Creation (Graceful Degradation)

```python
try:
    self.repo_graph.add_file(path, content)
except ParseError as e:
    # Expected during development - file has syntax errors
    log.warning(f"Syntax error in {path}, skipping for now")
except NodeNotFoundError as e:
    # Shouldn't happen in add_file, but defensive
    log.error(f"Unexpected: {e}")
except GraphError as e:
    # Catch-all for future error types
    log.error(f"Graph operation failed: {e}")
```

#### Pattern 2: File Deletion (Expected Failures)

```python
try:
    self.repo_graph.remove_file(path)
except NodeNotFoundError:
    # Normal - file was never in graph (ignored, binary, etc.)
    log.debug(f"File not tracked: {path}")
except GraphError as e:
    # Unexpected in remove_file
    log.error(f"Graph error: {e}")
```

#### Pattern 3: File Modification (Fallback to Creation)

```python
try:
    result = self.repo_graph.update_file(path, content)
except ParseError as e:
    # Syntax error introduced - log but continue
    log.warning(f"Syntax error after modification: {path}")
except NodeNotFoundError:
    # File not in graph - treat as creation
    log.warning(f"File not found, adding as new: {path}")
    self._handle_file_created(path)
except GraphError as e:
    log.error(f"Update failed: {e}")
```

### Error Severity Guidelines

| Exception | Log Level | User Notification | Continue Operation? |
|-----------|-----------|-------------------|---------------------|
| `ParseError` | WARNING | Optional (syntax errors) | Yes |
| `NodeNotFoundError` (in remove) | DEBUG | None | Yes |
| `NodeNotFoundError` (in add/update) | WARNING | None | Yes |
| `GraphError` (base) | ERROR | "Internal error" | Yes |
| Uncaught Exception | CRITICAL | "Critical failure" | Maybe not |

### Common Error Scenarios

**Scenario 1: User saves file with syntax error**
- Exception: `ParseError`
- Action: Log warning, skip file, continue watching
- Resolution: Auto-recovers when user fixes syntax

**Scenario 2: User deletes tracked file**
- Exception: None (normal operation)
- Action: Remove from graph, log success

**Scenario 3: User deletes ignored file**
- Exception: `NodeNotFoundError`
- Action: Log at debug level (expected)

**Scenario 4: File watcher reports modification of new file**
- Exception: `NodeNotFoundError` in `update_file`
- Action: Fall back to `add_file`, log warning

**Scenario 5: Network drive disconnects during read**
- Exception: Python `IOError` (before reaching Rust)
- Action: Retry with backoff, log error


---

## 7. Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `add_file` | O(E_in + E_out) | E_in/out = edges created |
| `remove_file` | O(E_total) | Worst case: many symbol edges |
| Unresolved lookup | O(1) | HashMap access |
| Swap remap | O(U) | U = unresolved imports count |

### Space Complexity

| Structure | Size | Notes |
|-----------|------|-------|
| `unresolved_imports` | O(P) | P = pending import count |
| `path_to_idx` | O(N) | N = file count |
| `graph` | O(N + E) | Standard graph |

### Optimization Notes

- `unresolved_imports` should remain small in practice (most imports resolve quickly)
- If memory becomes an issue, consider periodic cleanup of empty HashSet entries
- Symbol index cleanup could be parallelized for large files

---

## 8. Future Enhancements

### Potential Improvements (Out of Scope for Phase 4)

1. **Batch Operations**: `add_files(Vec<(PathBuf, String)>)` to amortize overhead
2. **Transaction Support**: Rollback capability if batch operation partially fails
3. **Change Notifications**: Return detailed diff of what changed for UI updates
4. **Incremental PageRank**: Update ranks without full recalculation
5. **Symbol Versioning**: Track when symbol definitions change for cache invalidation

### Known Limitations

1. **No Undo**: Once a file is removed, historical state is lost
2. **No Concurrency**: Operations are not thread-safe (requires `Arc<RwLock<>>`)
3. **No Rename Handling**: Rename is treated as delete + add (loses history)
4. **No Partial Updates**: Changing one symbol requires full file reparse

---

## Appendix A: Visual Decision Tree

```
File Event Received
│
├─ Is this a CREATION/MODIFICATION?
│  │
│  ├─ Read file content
│  │  │
│  │  ├─ SUCCESS
│  │  │  │
│  │  │  ├─ Normalize path
│  │  │  │  │
│  │  │  │  ├─ Path valid (within project)
│  │  │  │  │  │
│  │  │  │  │  └─ Call add_file(path, content)
│  │  │  │  │     │
│  │  │  │  │     ├─ SUCCESS → Log "Added to graph"
│  │  │  │  │     └─ PARSE ERROR → Log "Syntax error, skipped"
│  │  │  │  │
│  │  │  │  └─ Path invalid (outside project)
│  │  │  │     └─ Log "Ignored (outside project)"
│  │  │  │
│  │  │  └─ UNICODE ERROR
│  │  │     └─ Log "Skipped binary file"
│  │  │
│  │  └─ IO ERROR (file locked, deleted)
│  │     └─ Retry with backoff
│  │
│  └─ Continue monitoring
│
└─ Is this a DELETION?
   │
   ├─ Normalize path
   │  │
   │  ├─ Path valid
   │  │  │
   │  │  └─ Call remove_file(path)
   │  │     │
   │  │     ├─ SUCCESS → Log "Removed from graph"
   │  │     └─ NODE NOT FOUND → Log "Not tracked (ignored)"
   │  │
   │  └─ Path invalid
   │     └─ Ignore silently
   │
   └─ Continue monitoring
```

---

## Appendix B: Example Scenarios

### Scenario 1: File Creation

```
User creates: src/new_module.py
├─ Agent detects creation event
├─ Normalizes: "src/new_module.py" (canonical, relative)
├─ Reads content: "import utils\ndef process(): ..."
├─ Calls: graph.add_file("src/new_module.py", content)
│  ├─ Parses file
│  ├─ Finds import: "utils"
│  ├─ Resolves: utils → src/utils.py (exists in graph)
│  ├─ Creates edge: new_module → utils (Import)
│  ├─ Finds definition: "process"
│  ├─ Registers symbol: "process" → new_module
│  └─ Checks unresolved: No one was waiting for new_module
└─ Result: File added, edges created, PageRank dirty
```

### Scenario 2: Dependency Deletion & Restoration

```
Initial state: main.py → utils.py (import edge exists)

User deletes: utils.py
├─ Agent detects deletion
├─ Calls: graph.remove_file("utils.py")
│  ├─ Identifies utils at index 5
│  ├─ Finds incoming edge from main.py
│  ├─ Downgrades: main.py added to unresolved["utils.py"]
│  ├─ Removes utils node (index 5)
│  ├─ Last node (index 8) swaps into slot 5
│  └─ Updates: path_to_idx[last_node_path] = index 5
└─ Result: utils gone, main waiting for resolution

User restores: utils.py (git checkout)
├─ Agent detects creation
├─ Calls: graph.add_file("utils.py", content)
│  ├─ Parses file
│  ├─ Checks unresolved["utils.py"]
│  ├─ Finds: main.py is waiting
│  ├─ Creates edge: main.py → utils.py (Import)
│  └─ Clears: unresolved["utils.py"]
└─ Result: Dependency automatically restored!
```

### Scenario 3: Swap-Remove Detail

```
Graph state:
  Index 0: A.py
  Index 1: B.py
  Index 2: C.py

Remove: B.py (index 1)

Petgraph action:
  1. Remove node at index 1
  2. Move node at index 2 to index 1
  3. Decrement node count

Our response:
  1. Capture: C.py is at index 2 (before removal)
  2. Remove: B.py from path_to_idx
  3. Update: path_to_idx[C.py] = index 1 (after swap)
  4. Remap: unresolved_imports indices (2 → 1)

Final state:
  Index 0: A.py (unchanged)
  Index 1: C.py (moved from 2)
  path_to_idx: {A.py → 0, C.py → 1}
```

---

**End of Implementation Plan**

This plan provides a complete roadmap for implementing dynamic graph mutation with proper handling of petgraph's swap-remove behavior, optimized edge resolution, and transactional safety.