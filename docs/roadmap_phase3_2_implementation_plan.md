# Phase 3.2 Implementation Plan: Symbol-Based Graph Construction

## Executive Summary

**Status**: Phase 3.1 ✅ Complete | Phase 3.2 🔄 Ready to Implement

**Current State Analysis:**
- ✅ `SymbolHarvester` exists and works (parser.rs lines 399-480)
- ✅ `SymbolIndex` exists (symbol_table.rs)
- ✅ `RepoGraph` with `EdgeKind` enum exists (graph.rs)
- ✅ Basic graph building infrastructure exists
- ✅ `build_semantic_edges()` method exists but needs refinement
- ❌ Import edges not yet implemented
- ❌ Import resolution not integrated into graph
- ❌ No comprehensive tests for Phase 3.2

**Objective**: Integrate import resolution into graph construction and create weighted edges based on both imports and symbol usage.

**Estimated Time**: 3-4 hours of focused work

---

## Prerequisites Checklist

Before starting, verify:

- [ ] Phase 3.1 tests all pass: `cargo test` in `rust_core/`
- [ ] Files exist:
  - [ ] `rust_core/src/graph.rs`
  - [ ] `rust_core/src/import_resolver.rs`
  - [ ] `rust_core/src/parser.rs`
  - [ ] `rust_core/src/symbol_table.rs`
- [ ] ImportResolver successfully resolves imports (all Phase 3.1 tests pass)
- [ ] No compilation errors: `cargo build`

---

## Phase 3.2 Implementation: Step-by-Step

### STEP 1: Add ImportResolver to RepoGraph

**File**: `rust_core/src/graph.rs`
**Time**: 10 minutes

#### 1.1: Add Import at Top of File

Find the imports section (lines 1-8) and add:

```rust
use crate::import_resolver::ImportResolver;
```

**Verification**: File compiles without errors.

#### 1.2: Add ImportResolver Field to RepoGraph

Find the `RepoGraph` struct (around line 30) and modify it:

**BEFORE:**
```rust
pub struct RepoGraph {
    pub graph: DiGraph<FileNode, EdgeKind>,
    pub path_to_idx: HashMap<PathBuf, NodeIndex>,
    pub symbol_index: SymbolIndex,
}
```

**AFTER:**
```rust
pub struct RepoGraph {
    pub graph: DiGraph<FileNode, EdgeKind>,
    pub path_to_idx: HashMap<PathBuf, NodeIndex>,
    pub symbol_index: SymbolIndex,
    pub import_resolver: Option<ImportResolver>,  // NEW: Optional to allow creation without project root
}
```

**Verification**: Run `cargo build` - should compile.

#### 1.3: Update RepoGraph::new()

Find the `new()` method (around line 39) and modify:

**BEFORE:**
```rust
pub fn new() -> Self {
    Self {
        graph: DiGraph::new(),
        path_to_idx: HashMap::new(),
        symbol_index: SymbolIndex::new(),
    }
}
```

**AFTER:**
```rust
pub fn new() -> Self {
    Self {
        graph: DiGraph::new(),
        path_to_idx: HashMap::new(),
        symbol_index: SymbolIndex::new(),
        import_resolver: None,
    }
}

/// Creates a new RepoGraph with import resolution enabled.
pub fn new_with_import_resolution(project_root: &Path) -> Self {
    Self {
        graph: DiGraph::new(),
        path_to_idx: HashMap::new(),
        symbol_index: SymbolIndex::new(),
        import_resolver: Some(ImportResolver::new(project_root)),
    }
}
```

**Verification**: Run `cargo build` - should compile.

---

### STEP 2: Integrate Import Edge Creation

**File**: `rust_core/src/graph.rs`
**Time**: 30 minutes

#### 2.1: Create build_import_edges() Method

Add this new method to the `impl RepoGraph` block (after the `build()` method):

```rust
/// Builds edges based on import statements.
/// This must be called AFTER build() has populated all nodes.
pub fn build_import_edges(&mut self) {
    if self.import_resolver.is_none() {
        eprintln!("[WARN] ImportResolver not initialized. Skipping import edge construction.");
        return;
    }
    
    let resolver = self.import_resolver.as_ref().unwrap();
    
    // Clone the paths to avoid borrowing issues
    let all_paths: Vec<PathBuf> = self.path_to_idx.keys().cloned().collect();
    
    for source_path in &all_paths {
        // Read the file
        let source_code = match fs::read_to_string(source_path) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("[ERROR] Failed to read {:?}: {}", source_path, e);
                continue;
            }
        };
        
        // Parse the file to get import statements
        let lang = SupportedLanguage::from_extension(
            source_path.extension().and_then(|s| s.to_str()).unwrap_or("")
        );
        
        if lang == SupportedLanguage::Unknown {
            continue;
        }
        
        let mut parser = TreeSitterParser::new();
        if parser.set_language(lang.get_parser().unwrap()).is_err() {
            continue;
        }
        
        let tree = match parser.parse(&source_code, None) {
            Some(t) => t,
            None => continue,
        };
        
        // Find all import statements using a query
        let import_query_str = match lang {
            SupportedLanguage::Python => {
                "(import_statement) @import\n(import_from_statement) @import"
            },
            _ => continue, // Only Python for now
        };
        
        let import_query = match tree_sitter::Query::new(lang.get_parser().unwrap(), import_query_str) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("[ERROR] Failed to create import query: {}", e);
                continue;
            }
        };
        
        let mut cursor = tree_sitter::QueryCursor::new();
        let matches = cursor.matches(&import_query, tree.root_node(), source_code.as_bytes());
        
        for m in matches {
            for capture in m.captures {
                let import_node = capture.node;
                
                // Resolve the import
                if let Some(target_path) = resolver.resolve(
                    import_node,
                    source_path,
                    source_code.as_bytes()
                ) {
                    // Get node indices
                    let source_idx = match self.path_to_idx.get(source_path) {
                        Some(&idx) => idx,
                        None => continue,
                    };
                    
                    let target_idx = match self.path_to_idx.get(&target_path) {
                        Some(&idx) => idx,
                        None => {
                            // Target file not in graph (external dependency or filtered out)
                            continue;
                        }
                    };
                    
                    // Don't create self-edges
                    if source_idx == target_idx {
                        continue;
                    }
                    
                    // Add Import edge
                    println!(
                        "[Import Edge] {:?} -> {:?}",
                        source_path.file_name().unwrap(),
                        target_path.file_name().unwrap()
                    );
                    
                    self.graph.update_edge(source_idx, target_idx, EdgeKind::Import);
                }
            }
        }
    }
}
```

**Verification**: Run `cargo build` - should compile.

---

### STEP 3: Refine build_semantic_edges()

**File**: `rust_core/src/graph.rs`
**Time**: 15 minutes

#### 3.1: Review Current Implementation

The existing `build_semantic_edges()` method (around line 119) should be working. Let's enhance it with better logging and deduplication:

**Replace the existing method with this improved version:**

```rust
/// After initial population, this method builds the semantic edges based on symbol usage.
/// This should be called AFTER build_import_edges() for proper edge prioritization.
pub fn build_semantic_edges(&mut self) {
    let usages_clone = self.symbol_index.usages.clone();
    let mut edge_count = 0;
    
    for (user_path, used_symbols) in usages_clone {
        let user_node_idx = match self.path_to_idx.get(&user_path) {
            Some(&idx) => idx,
            None => continue,
        };
        
        for symbol in used_symbols {
            let def_paths = match self.symbol_index.definitions.get(&symbol) {
                Some(paths) => paths,
                None => {
                    // Symbol not defined in our codebase (external or unresolved)
                    continue;
                }
            };
            
            for def_path in def_paths {
                // Skip self-references
                if &user_path == def_path {
                    continue;
                }
                
                let def_node_idx = match self.path_to_idx.get(def_path) {
                    Some(&idx) => idx,
                    None => continue,
                };
                
                // Check if an edge already exists
                let edge_exists = self.graph.find_edge(user_node_idx, def_node_idx).is_some();
                
                if !edge_exists {
                    println!(
                        "[Symbol Edge] {:?} -> {:?} (symbol: {})",
                        user_path.file_name().unwrap(),
                        def_path.file_name().unwrap(),
                        symbol
                    );
                    
                    self.graph.add_edge(user_node_idx, def_node_idx, EdgeKind::SymbolUsage);
                    edge_count += 1;
                } else {
                    // Edge exists - could upgrade from Import to SymbolUsage if needed
                    // For now, SymbolUsage takes precedence
                    if let Some(edge_idx) = self.graph.find_edge(user_node_idx, def_node_idx) {
                        self.graph[edge_idx] = EdgeKind::SymbolUsage;
                    }
                }
            }
        }
    }
    
    println!("[Graph] Added {} semantic edges", edge_count);
}
```

**Verification**: Run `cargo build` - should compile.

---

### STEP 4: Update Main Build Process

**File**: `rust_core/src/graph.rs`
**Time**: 10 minutes

#### 4.1: Add build_complete() Orchestration Method

Add this new method that orchestrates the complete build process:

```rust
/// Complete build process: parse files, harvest symbols, build import edges, build semantic edges.
pub fn build_complete(&mut self, paths: &[PathBuf], project_root: &Path) {
    println!("[Phase 3.2] Starting complete graph build...");
    println!("[Phase 3.2] Files to process: {}", paths.len());
    
    // Step 1: Initialize import resolver if not already done
    if self.import_resolver.is_none() {
        self.import_resolver = Some(ImportResolver::new(project_root));
        println!("[Phase 3.2] ImportResolver initialized");
    }
    
    // Step 2: Build nodes and harvest symbols (existing method)
    println!("[Phase 3.2] Harvesting symbols...");
    self.build(paths);
    println!("[Phase 3.2] Nodes created: {}", self.graph.node_count());
    
    // Step 3: Build import edges
    println!("[Phase 3.2] Building import edges...");
    self.build_import_edges();
    let import_edge_count = self.graph.edge_count();
    println!("[Phase 3.2] Import edges created: {}", import_edge_count);
    
    // Step 4: Build semantic edges
    println!("[Phase 3.2] Building semantic edges...");
    self.build_semantic_edges();
    let total_edges = self.graph.edge_count();
    let semantic_edge_count = total_edges - import_edge_count;
    println!("[Phase 3.2] Semantic edges created: {}", semantic_edge_count);
    
    println!("[Phase 3.2] Graph build complete!");
    println!("[Phase 3.2] Total nodes: {}, Total edges: {}", self.graph.node_count(), total_edges);
}
```

**Verification**: Run `cargo build` - should compile.

---

### STEP 5: Create Comprehensive Tests

**File**: Create `rust_core/tests/test_graph_construction.rs`
**Time**: 45 minutes

Create a new test file with this content:

```rust
use semantic_engine::graph::{RepoGraph, EdgeKind};
use semantic_engine::parser::SupportedLanguage;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

fn create_test_file(root: &std::path::Path, path: &str, content: &str) {
    let file_path = root.join(path);
    fs::create_dir_all(file_path.parent().unwrap()).unwrap();
    fs::write(file_path, content).unwrap();
}

#[test]
fn test_import_edge_creation() {
    let root = tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    
    // Create a simple project
    create_test_file(&root_path, "utils.py", r#"
def helper():
    pass
"#);
    
    create_test_file(&root_path, "main.py", r#"
import utils

utils.helper()
"#);
    
    // Build graph
    let mut graph = RepoGraph::new_with_import_resolution(&root_path);
    let paths = vec![
        root_path.join("utils.py"),
        root_path.join("main.py"),
    ];
    
    graph.build_complete(&paths, &root_path);
    
    // Verify import edge exists
    assert_eq!(graph.graph.node_count(), 2, "Should have 2 nodes");
    assert!(graph.graph.edge_count() >= 1, "Should have at least 1 edge");
    
    // Check that main.py imports utils.py
    let main_idx = graph.path_to_idx.get(&root_path.join("main.py")).unwrap();
    let utils_idx = graph.path_to_idx.get(&root_path.join("utils.py")).unwrap();
    
    let edge = graph.graph.find_edge(*main_idx, *utils_idx);
    assert!(edge.is_some(), "Should have edge from main to utils");
}

#[test]
fn test_symbol_usage_edge_creation() {
    let root = tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    
    // Create files with symbol usage
    create_test_file(&root_path, "models.py", r#"
class User:
    pass
"#);
    
    create_test_file(&root_path, "app.py", r#"
from models import User

user = User()
"#);
    
    let mut graph = RepoGraph::new_with_import_resolution(&root_path);
    let paths = vec![
        root_path.join("models.py"),
        root_path.join("app.py"),
    ];
    
    graph.build_complete(&paths, &root_path);
    
    // Verify both import and symbol edges
    assert_eq!(graph.graph.node_count(), 2);
    assert!(graph.graph.edge_count() >= 1);
    
    // Check edge type is SymbolUsage (should override Import)
    let app_idx = graph.path_to_idx.get(&root_path.join("app.py")).unwrap();
    let models_idx = graph.path_to_idx.get(&root_path.join("models.py")).unwrap();
    
    let edge_idx = graph.graph.find_edge(*app_idx, *models_idx).unwrap();
    let edge_kind = &graph.graph[edge_idx];
    
    assert_eq!(edge_kind, &EdgeKind::SymbolUsage, "Should be SymbolUsage edge");
}

#[test]
fn test_no_self_edges() {
    let root = tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    
    create_test_file(&root_path, "self_import.py", r#"
class MyClass:
    pass

obj = MyClass()
"#);
    
    let mut graph = RepoGraph::new_with_import_resolution(&root_path);
    let paths = vec![root_path.join("self_import.py")];
    
    graph.build_complete(&paths, &root_path);
    
    assert_eq!(graph.graph.node_count(), 1);
    assert_eq!(graph.graph.edge_count(), 0, "Should have no self-edges");
}

#[test]
fn test_multiple_symbol_definitions() {
    let root = tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    
    // Same symbol defined in two files
    create_test_file(&root_path, "util1.py", r#"
def process():
    pass
"#);
    
    create_test_file(&root_path, "util2.py", r#"
def process():
    pass
"#);
    
    create_test_file(&root_path, "main.py", r#"
from util1 import process

process()
"#);
    
    let mut graph = RepoGraph::new_with_import_resolution(&root_path);
    let paths = vec![
        root_path.join("util1.py"),
        root_path.join("util2.py"),
        root_path.join("main.py"),
    ];
    
    graph.build_complete(&paths, &root_path);
    
    // Should have 3 nodes
    assert_eq!(graph.graph.node_count(), 3);
    
    // Should have edges to util1 (not util2, since we import from util1)
    let main_idx = graph.path_to_idx.get(&root_path.join("main.py")).unwrap();
    let util1_idx = graph.path_to_idx.get(&root_path.join("util1.py")).unwrap();
    
    assert!(graph.graph.find_edge(*main_idx, *util1_idx).is_some());
}

#[test]
fn test_graph_statistics() {
    let root = tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    
    // Create a small project
    create_test_file(&root_path, "core.py", r#"
class Database:
    pass
"#);
    
    create_test_file(&root_path, "models.py", r#"
from core import Database

class User:
    def __init__(self):
        self.db = Database()
"#);
    
    create_test_file(&root_path, "api.py", r#"
from models import User

user = User()
"#);
    
    let mut graph = RepoGraph::new_with_import_resolution(&root_path);
    let paths = vec![
        root_path.join("core.py"),
        root_path.join("models.py"),
        root_path.join("api.py"),
    ];
    
    graph.build_complete(&paths, &root_path);
    
    println!("Nodes: {}", graph.graph.node_count());
    println!("Edges: {}", graph.graph.edge_count());
    
    // Should have 3 nodes
    assert_eq!(graph.graph.node_count(), 3);
    
    // Should have at least 2 edges (models->core, api->models)
    assert!(graph.graph.edge_count() >= 2);
}
```

**Update Cargo.toml** to register the test:

```toml
[[test]]
name = "test_graph_construction"
harness = true
```

**Verification**: Run `cargo test test_graph_construction` - tests should compile and pass.

---

### STEP 6: Add Helper Methods for Analysis

**File**: `rust_core/src/graph.rs`
**Time**: 20 minutes

Add these utility methods to help analyze the graph:

```rust
/// Get statistics about the graph
pub fn get_statistics(&self) -> GraphStatistics {
    let mut import_edges = 0;
    let mut symbol_edges = 0;
    
    for edge_idx in self.graph.edge_indices() {
        match self.graph[edge_idx] {
            EdgeKind::Import => import_edges += 1,
            EdgeKind::SymbolUsage => symbol_edges += 1,
        }
    }
    
    GraphStatistics {
        node_count: self.graph.node_count(),
        edge_count: self.graph.edge_count(),
        import_edges,
        symbol_edges,
        total_definitions: self.symbol_index.definitions.len(),
        total_files_with_usages: self.symbol_index.usages.len(),
    }
}

/// Get all edges of a specific kind
pub fn get_edges_by_kind(&self, kind: EdgeKind) -> Vec<(PathBuf, PathBuf)> {
    let mut edges = Vec::new();
    
    for edge_idx in self.graph.edge_indices() {
        if self.graph[edge_idx] == kind {
            let (source_idx, target_idx) = self.graph.edge_endpoints(edge_idx).unwrap();
            let source_path = &self.graph[source_idx].path;
            let target_path = &self.graph[target_idx].path;
            edges.push((source_path.clone(), target_path.clone()));
        }
    }
    
    edges
}

/// Get incoming dependencies for a file
pub fn get_incoming_dependencies(&self, file_path: &Path) -> Vec<(PathBuf, EdgeKind)> {
    let node_idx = match self.path_to_idx.get(file_path) {
        Some(&idx) => idx,
        None => return Vec::new(),
    };
    
    let mut deps = Vec::new();
    
    for edge in self.graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
        let source_node = &self.graph[edge.source()];
        deps.push((source_node.path.clone(), edge.weight().clone()));
    }
    
    deps
}

/// Get outgoing dependencies for a file
pub fn get_outgoing_dependencies(&self, file_path: &Path) -> Vec<(PathBuf, EdgeKind)> {
    let node_idx = match self.path_to_idx.get(file_path) {
        Some(&idx) => idx,
        None => return Vec::new(),
    };
    
    let mut deps = Vec::new();
    
    for edge in self.graph.edges_directed(node_idx, petgraph::Direction::Outgoing) {
        let target_node = &self.graph[edge.target()];
        deps.push((target_node.path.clone(), edge.weight().clone()));
    }
    
    deps
}
```

Add the statistics struct at the top of the file (after the `EdgeKind` enum):

```rust
/// Statistics about the repository graph
#[derive(Debug)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub import_edges: usize,
    pub symbol_edges: usize,
    pub total_definitions: usize,
    pub total_files_with_usages: usize,
}
```

**Verification**: Run `cargo build` - should compile.

---

## Testing & Validation

### Phase 1: Unit Tests

Run each test individually with debug output:

```bash
cd rust_core
cargo test test_import_edge_creation -- --nocapture
cargo test test_symbol_usage_edge_creation -- --nocapture
cargo test test_no_self_edges -- --nocapture
cargo test test_multiple_symbol_definitions -- --nocapture
cargo test test_graph_statistics -- --nocapture
```

**Expected**: All tests pass with debug output showing edges being created.

### Phase 2: Run All Tests

```bash
cargo test
```

**Expected**: All Phase 3.1 and Phase 3.2 tests pass.

### Phase 3: Integration Test

Create a manual test in `rust_core/examples/`:

**File**: `rust_core/examples/test_graph_build.rs`

```rust
use semantic_engine::graph::RepoGraph;
use std::path::PathBuf;
use walkdir::WalkDir;

fn main() {
    let project_root = std::env::args()
        .nth(1)
        .expect("Usage: cargo run --example test_graph_build <project_path>");
    
    let root_path = PathBuf::from(&project_root);
    
    println!("Building graph for: {:?}", root_path);
    
    // Collect all Python files
    let mut paths = Vec::new();
    for entry in WalkDir::new(&root_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("py"))
    {
        paths.push(entry.path().to_path_buf());
    }
    
    println!("Found {} Python files", paths.len());
    
    // Build graph
    let mut graph = RepoGraph::new_with_import_resolution(&root_path);
    graph.build_complete(&paths, &root_path);
    
    // Print statistics
    let stats = graph.get_statistics();
    println!("\n=== Graph Statistics ===");
    println!("Nodes: {}", stats.node_count);
    println!("Total Edges: {}", stats.edge_count);
    println!("  Import Edges: {}", stats.import_edges);
    println!("  Symbol Edges: {}", stats.symbol_edges);
    println!("Unique Symbols Defined: {}", stats.total_definitions);
    println!("Files with Symbol Usages: {}", stats.total_files_with_usages);
}
```

Run it on a test project:

```bash
cargo run --example test_graph_build /path/to/python/project
```

---

## Success Criteria

Phase 3.2 is complete when:

- [ ] All unit tests pass
- [ ] Integration test runs without panics
- [ ] Graph has both Import and SymbolUsage edges
- [ ] `build_complete()` logs show:
  - Nodes created count matches file count
  - Import edges > 0
  - Semantic edges > 0
- [ ] No self-edges in graph
- [ ] Symbol collisions handled (multiple definitions of same symbol)
- [ ] External dependencies filtered (stdlib, third-party)

### Quantitative Metrics (from roadmap)

- [ ] ✅ **Distinguishes between "Imported but unused" (Weak) and "Called" (Strong)**
  - Verify: Check `EdgeKind::Import` vs `EdgeKind::SymbolUsage` in graph
  
- [ ] ✅ **Handles multiple definitions (collisions) via heuristic weighting**
  - Verify: `symbol_index.definitions` values can have `Vec` length > 1
  
- [ ] ✅ **Graph construction time < 2 seconds for 10k files**
  - Verify: Run on large repository and measure time

---

## Troubleshooting Guide

### Issue: No Import Edges Created

**Symptoms**: `build_import_edges()` shows 0 edges created

**Diagnosis**:
1. Check ImportResolver is initialized: `graph.import_resolver.is_some()`
2. Enable debug logging in `build_import_edges()`
3. Verify import statements exist in source files
4. Check that files are in the graph: `graph.path_to_idx` contains the paths

**Fix**: Ensure `new_with_import_resolution()` was used, not `new()`

### Issue: No Semantic Edges Created

**Symptoms**: `build_semantic_edges()` shows 0 edges created

**Diagnosis**:
1. Check `symbol_index.definitions` is populated: `println!("{:?}", graph.symbol_index.definitions);`
2. Check `symbol_index.usages` is populated: `println!("{:?}", graph.symbol_index.usages);`
3. Enable debug logging to see which symbols are being processed

**Fix**: Verify `SymbolHarvester` is working correctly

### Issue: Compilation Errors

**Common Issues**:
- Missing imports: Add `use tree_sitter;` or other missing imports
- Missing `use petgraph::Direction;` in helper methods
- Lifetime errors: Check that references are valid
- Borrow checker issues: Clone data when needed

**Fix**: Follow Rust compiler suggestions carefully

### Issue: Test Failures

**Diagnosis**:
1. Run test with `--nocapture` to see debug output
2. Check file paths are correct in assertions
3. Verify temp directory is writeable
4. Check that expected edge counts are reasonable

**Common Test Issues**:
- Path comparison issues: Use `file_name()` for comparison instead of full paths
- Edge count expectations: Remember that edges can be upgraded from Import to SymbolUsage

---

## Next Steps After Phase 3.2

Once Phase 3.2 is complete, proceed to:

1. **Phase 3.3**: PageRank Implementation
   - Calculate file importance based on graph structure
   - Implement iterative PageRank algorithm
   - Identify architectural core files
   
2. **Phase 3.4**: Repository Map Serialization
   - Generate compact text representation
   - Format for LLM context window
   - Create hierarchical file views

3. **Integration**: Connect graph to Python API
   - Expose through PyO3
   - Create Python interface for querying graph
   - Enable external tools to consume graph data

---

## Appendix: Quick Reference

### Key Files Modified
- `rust_core/src/graph.rs` - Main graph implementation (MODIFIED)
- `rust_core/tests/test_graph_construction.rs` - Test suite (NEW)
- `rust_core/examples/test_graph_build.rs` - Integration test (NEW)
- `rust_core/Cargo.toml` - Add test registration (MODIFIED)

### Key Methods Added
- `RepoGraph::new_with_import_resolution()` - Constructor with import resolution
- `RepoGraph::build_import_edges()` - Build import-based edges
- `RepoGraph::build_complete()` - Orchestrate complete build
- `RepoGraph::get_statistics()` - Get graph statistics
- `RepoGraph::get_edges_by_kind()` - Filter edges by type
- `RepoGraph::get_incoming_dependencies()` - Get file's dependents
- `RepoGraph::get_outgoing_dependencies()` - Get file's dependencies

### Key Structs
- `GraphStatistics` - Statistics about graph structure
- `EdgeKind` - Enum for edge types (Import, SymbolUsage) [already existed]
- `FileNode` - Node data structure [already existed]
- `RepoGraph` - Main graph container [already existed, enhanced]

---

## Estimated Time Breakdown

- Step 1 (Add ImportResolver): 10 min
- Step 2 (Import Edges): 30 min
- Step 3 (Refine Semantic Edges): 15 min
- Step 4 (Build Complete): 10 min
- Step 5 (Tests): 45 min
- Step 6 (Helper Methods): 20 min
- Testing & Validation: 30 min
- Debugging & Iteration: 30-60 min
- **Total: 2.5-3.5 hours** (conservative estimate)

With testing and validation: **3-4 hours total**

---

## Final Checklist

Before marking Phase 3.2 as complete:

- [ ] All code compiles without warnings
- [ ] All Phase 3.1 tests still pass
- [ ] All Phase 3.2 tests pass
- [ ] Integration example runs successfully on real code
- [ ] Graph has both Import and SymbolUsage edges
- [ ] Edge counts are reasonable
- [ ] Debug output is clean and informative
- [ ] Code is properly documented
- [ ] Commit changes with message: "feat: Complete Phase 3.2 - Symbol-Based Graph Construction"
- [ ] Ready to proceed to Phase 3.3

---

## Common Pitfalls to Avoid

1. **Forgetting to use `new_with_import_resolution()`**: Always use this constructor when you want import edges
2. **Calling `build()` instead of `build_complete()`**: Use `build_complete()` for the full workflow
3. **Not handling `Option<ImportResolver>`**: Always check `is_some()` before using
4. **Self-edges**: Always check `source_idx != target_idx` before adding edges
5. **Missing imports**: Remember to add `use petgraph::Direction;` for helper methods
6. **Path comparison**: Use canonicalized paths or file names for comparison
7. **Edge upgrade logic**: SymbolUsage should take precedence over Import when both exist

Following this plan step-by-step should result in a fully functional Phase 3.2 implementation!