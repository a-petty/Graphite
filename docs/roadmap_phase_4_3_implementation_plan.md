# Phase 4.3 Implementation Plan: Graph Update Propagation

## Overview

**Goal:** Implement the core logic for updating the Repository Graph in response to file changes, specifically focusing on import modifications.

**Parent Phase:** Phase 4: Incremental Updates

**Important Clarifications:**
- **Prerequisites:** This plan assumes that the file being updated has already been re-parsed (as per Phase 4.2) and that its new import statements have been resolved to file paths (as per Phase 3.1). The `update_file_imports` function operates on this pre-processed data.
- **Scope:** This phase focuses exclusively on updating the graph based on `Import` edges. Updates related to `SymbolUsage` edges (i.e., tracking function calls or class instantiations) will be handled in a subsequent phase, likely as part of the Phase 4.4 Multi-Tier Update Strategy.

**Success Criteria:**
- ✅ Graph edges are correctly updated when imports are added or removed from a file.
- ✅ PageRank is only recalculated when structural changes (import changes) are significant.
- ✅ Update operations are fast (< 10ms per file).
- ✅ No memory leaks during prolonged operation with multiple updates.
- ✅ Python API for triggering graph updates.
- ✅ Comprehensive test coverage.

**Estimated Time:** 3-4 hours

---

## Step 1: Define Update Result Structure

**File:** `rust_core/src/graph.rs` (or a new `rust_core/src/update_logic.rs` if preferred for modularity)
**Time:** 10 minutes

### 1.1: Add `UpdateResult` struct

```rust
// rust_core/src/graph.rs (or update_logic.rs)

/// Result of a graph update operation
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct UpdateResult {
    pub edges_added: usize,
    pub edges_removed: usize,
    pub needs_pagerank_recalc: bool,
    pub file_reparsed: bool, // Track if the file was re-parsed
}
```

**Why `UpdateResult`?**
- Provides clear feedback on what happened during an update.
- Allows the caller (Python agent) to decide if further actions (like PageRank recalculation) are needed.
- `file_reparsed` can indicate if the file's skeleton or symbol index needs to be refreshed.

---

## Step 2: Implement `update_file_imports` in `RepoGraph`

**File:** `rust_core/src/graph.rs`
**Time:** 90 minutes

### 2.1: Extend `RepoGraph` with update logic

This function will:
1.  Locate the node corresponding to the changed file.
2.  Identify existing outgoing import edges from this node.
3.  Remove these old edges.
4.  Add new edges based on the provided `new_imports`.
5.  Determine if a PageRank recalculation is necessary.

```rust
// rust_core/src/graph.rs

use petgraph::graph::NodeIndex;
use std::path::PathBuf;
use std::collections::HashMap; // Assuming SymbolIndex or similar uses HashMap

// Assuming Import and EdgeKind are defined elsewhere, e.g., in graph.rs
// For example:
// pub struct Import { pub resolved_path: Option<PathBuf>, /* ... other import details ... */ }
// pub enum EdgeKind { Import, SymbolUsage }

impl RepoGraph {
    /// Updates the import edges for a given file within the graph.
    ///
    /// This function performs the following steps:
    /// 1. Finds the graph node for the specified `file_path`.
    /// 2. Identifies all existing outgoing import edges from this node.
    /// 3. Removes all previously established import edges from this node.
    /// 4. Adds new import edges based on the `new_imports` provided.
    /// 5. Determines if the change is significant enough to warrant a PageRank recalculation.
    ///
    /// # Arguments
    /// * `file_path` - The path to the file that has changed.
    /// * `new_imports` - A vector of new `Import` structures extracted from the file.
    ///
    /// # Returns
    /// An `UpdateResult` indicating the changes made and if PageRank needs recalculation.
    /// Returns `Err(GraphError::NodeNotFound)` if the file_path does not correspond to a node.
    pub fn update_file_imports(
        &mut self,
        file_path: &PathBuf,
        new_imports: Vec<Import>, // Assuming `Import` contains `resolved_path`
    ) -> Result<UpdateResult, GraphError> {
        let node_idx = *self.path_to_idx.get(file_path)
            .ok_or(GraphError::NodeNotFound)?; // `path_to_idx` is a `HashMap<PathBuf, NodeIndex>`
        
        let initial_edge_count = self.graph.edges(node_idx).count();
        let mut edges_removed_count = 0;

        // Remove all existing outgoing import edges from this node.
        // We only remove 'Import' type edges, not 'SymbolUsage' if they exist.
        // `retain_edges` closure takes (graph, edge_index)
        self.graph.retain_edges(|g, edge_idx| {
            let (source, _) = g.edge_endpoints(edge_idx).expect("Edge should have endpoints");
            if source == node_idx {
                // If it's an outgoing edge from our node, and it's an Import kind, remove it.
                if matches!(g[edge_idx], EdgeKind::Import) {
                    edges_removed_count += 1;
                    return false; // Remove this edge
                }
            }
            true // Keep other edges
        });
        
        let mut edges_added_count = 0;
        let mut significant_change = false;

        // Add new import edges.
        for import in new_imports {
            if let Some(target_path) = import.resolved_path {
                if let Some(&target_node_idx) = self.path_to_idx.get(&target_path) {
                    // Only add if an edge doesn't already exist to avoid duplicates
                    if !self.graph.contains_edge(node_idx, target_node_idx) {
                        self.graph.add_edge(node_idx, target_node_idx, EdgeKind::Import);
                        edges_added_count += 1;
                    }
                } else {
                    // If an import points to a file not yet in the graph, we might want to add it
                    // or just ignore it for now. For this phase, let's ignore.
                }
            }
        }

        // Determine if PageRank needs recalculation.
        // A significant change is when the number of imports (edges) has changed,
        // or if the set of imported files has changed.
        // For simplicity initially, if any edges are added or removed, we flag for recalculation.
        if edges_added_count > 0 || edges_removed_count > 0 {
            significant_change = true;
            // Mark the graph as dirty to trigger lazy PageRank recalculation
            self.pagerank_dirty = true;
        }

        Ok(UpdateResult {
            edges_added: edges_added_count,
            edges_removed: edges_removed_count,
            needs_pagerank_recalc: significant_change,
            file_reparsed: true, // This function implies a file was re-parsed to get new imports
        })
    }

    /// Marks the graph's PageRank as dirty, indicating it needs recalculation.
    /// This supports lazy PageRank updates.
    pub fn mark_pagerank_dirty(&mut self) {
        self.pagerank_dirty = true;
    }

    /// Recalculates PageRank if it's marked as dirty.
    /// This method can be called before operations that require up-to-date PageRank scores.
    pub fn ensure_pagerank_up_to_date(&mut self) {
        if self.pagerank_dirty {
            self.calculate_pagerank(20, 0.85); // Assuming these are default values
            self.pagerank_dirty = false;
        }
    }
}

// Add to RepoGraph struct
// pub struct RepoGraph {
//     // ... existing fields ...
//     pagerank_dirty: bool, // New field to track if PageRank needs recalculation
// }

// Initialize pagerank_dirty to false in RepoGraph::new()
// impl RepoGraph {
//     pub fn new(...) -> Self {
//         Self {
//             // ...
//             pagerank_dirty: false,
//         }
//     }
// }
```

**Verification:** Run `cargo build` - should compile successfully. Add `pagerank_dirty: bool` to the `RepoGraph` struct definition and initialize it in `RepoGraph::new()` and create a simple `GraphError` enum if it doesn't already exist.

---

## Step 3: Implement Lazy PageRank Evaluation

**File:** `rust_core/src/graph.rs`
**Time:** 30 minutes

### 3.1: Modify `get_top_ranked_files` (and other PageRank-dependent methods)

The idea is that `pagerank_dirty` flag is set to `true` when a structural change occurs. When a method that relies on PageRank is called, it first checks this flag and recalculates if necessary.

```rust
// rust_core/src/graph.rs

impl RepoGraph {
    // ... update_file_imports, etc. ...

    /// Calculates PageRank for all nodes in the graph.
    /// This method is typically called internally or explicitly when fresh ranks are needed.
    // Assuming calculate_pagerank exists from Phase 3.3
    // pub fn calculate_pagerank(&mut self, iterations: usize, damping: f64) { /* ... */ }

    /// Returns the top-ranked files by architectural importance.
    /// Automatically recalculates PageRank if the graph is marked as dirty.
    pub fn get_top_ranked_files(&mut self, limit: usize) -> Vec<(PathBuf, f64)> {
        // Ensure PageRank is up-to-date before retrieving.
        self.ensure_pagerank_up_to_date();
        
        let mut ranked: Vec<_> = self.graph
            .node_weights()
            .map(|n| (n.path.clone(), n.rank)) // Assuming `FileNode` has a `rank` field
            .collect();
        
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(limit);
        ranked
    }

    /// Retrieves the dependents (files that import/use symbols from) of a given file.
    /// This method also ensures PageRank is up-to-date.
    pub fn get_dependents(&mut self, target_path: &PathBuf) -> Vec<PathBuf> {
        self.ensure_pagerank_up_to_date(); // May not strictly need PageRank, but good for consistency
        
        if let Some(&target_node_idx) = self.path_to_idx.get(target_path) {
            self.graph
                .edges_directed(target_node_idx, petgraph::Direction::Incoming)
                .map(|edge| self.graph[edge.source()].path.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Retrieves the dependencies (files that are imported/used by) of a given file.
    /// This method also ensures PageRank is up-to-date.
    pub fn get_dependencies(&mut self, source_path: &PathBuf) -> Vec<PathBuf> {
        self.ensure_pagerank_up_to_date(); // May not strictly need PageRank, but good for consistency
        
        if let Some(&source_node_idx) = self.path_to_idx.get(source_path) {
            self.graph
                .edges_directed(source_node_idx, petgraph::Direction::Outgoing)
                .map(|edge| self.graph[edge.target()].path.clone())
                .collect()
        } else {
            Vec::new()
        }
    }
}
```

**Verification:** `cargo build`

---

## Step 4: Python API for Graph Updates

**File:** `rust_core/src/lib.rs`
**Time:** 45 minutes

### 4.1: Add Python wrapper for `update_file_imports`

This will expose the Rust `RepoGraph::update_file_imports` function to the Python agent.

```rust
// rust_core/src/lib.rs

// Add to PyRepoGraph methods
#[pymethods]
impl PyRepoGraph {
    // ... existing methods ...

    /// Python representation of an Import, suitable for passing to Rust
    /// This might need to be a separate `PyImport` struct if `Import` is complex
    #[pyclass(name = "Import")]
    #[derive(Clone)]
    pub struct PyImport {
        #[pyo3(get, set)]
        pub module_name: String,
        #[pyo3(get, set)]
        pub resolved_path: Option<String>,
        // Add other relevant fields if your Rust `Import` struct has them
    }

    #[pymethods]
    impl PyImport {
        #[new]
        fn new(module_name: String, resolved_path: Option<String>) -> Self {
            PyImport { module_name, resolved_path }
        }

        fn __repr__(&self) -> String {
            format!(
                "Import(module_name='{}', resolved_path='{}')",
                self.module_name,
                self.resolved_path.as_deref().unwrap_or("None")
            )
        }
    }

    /// Python representation of UpdateResult
    #[pyclass(name = "GraphUpdateResult")]
    #[derive(Clone)]
    pub struct PyGraphUpdateResult {
        #[pyo3(get)]
        pub edges_added: usize,
        #[pyo3(get)]
        pub edges_removed: usize,
        #[pyo3(get)]
        pub needs_pagerank_recalc: bool,
        #[pyo3(get)]
        pub file_reparsed: bool,
    }

    impl From<graph::UpdateResult> for PyGraphUpdateResult {
        fn from(res: graph::UpdateResult) -> Self {
            Self {
                edges_added: res.edges_added,
                edges_removed: res.edges_removed,
                needs_pagerank_recalc: res.needs_pagerank_recalc,
                file_reparsed: res.file_reparsed,
            }
        }
    }

    /// Update the graph's import edges for a specific file.
    ///
    /// This should be called after a file has been re-parsed and its new imports
    /// have been extracted.
    #[pyo3(signature = (file_path, new_imports))]
    fn update_file_imports(
        &mut self,
        file_path: String,
        new_imports: Vec<PyImport>,
    ) -> PyResult<PyGraphUpdateResult> {
        let file_path_buf = PathBuf::from(file_path);
        let rust_imports: Vec<graph::Import> = new_imports.into_iter().map(|py_import| {
            // Convert PyImport to Rust Import
            // Assuming graph::Import has a similar structure, or a conversion function
            graph::Import {
                module_name: py_import.module_name,
                resolved_path: py_import.resolved_path.map(PathBuf::from),
                // ... map other fields if they exist ...
            }
        }).collect();

        let result = self.graph_instance.update_file_imports(&file_path_buf, rust_imports)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update graph imports: {}", e)))?;
        
        Ok(PyGraphUpdateResult::from(result))
    }

    /// Force a PageRank recalculation if it's dirty.
    fn ensure_pagerank_up_to_date(&mut self) {
        self.graph_instance.ensure_pagerank_up_to_date();
    }
}

// Don't forget to add PyImport and PyGraphUpdateResult to the PyO3 module!
#[pymodule]
fn semantic_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    // ... existing add_class calls ...
    m.add_class::<PyImport>()?;
    m.add_class::<PyGraphUpdateResult>()?;
    Ok(())
}
```

**Verification:** Run `maturin develop` - should build and install module.

---

## Step 5: Comprehensive Tests for Graph Updates

**File:** Add to `rust_core/tests/test_graph_construction.rs` (or a new `test_graph_updates.rs`)
**Time:** 60 minutes

### 5.1: Test `update_file_imports`

```rust
// rust_core/tests/test_graph_updates.rs

use semantic_engine::graph::{RepoGraph, Import, EdgeKind, UpdateResult, GraphError};
use std::path::PathBuf;

// Helper to create a dummy RepoGraph with some files
fn setup_test_graph() -> RepoGraph {
    let mut graph = RepoGraph::new();
    // Assuming `add_file` adds a node and returns its index
    graph.add_file_node(PathBuf::from("src/main.py"));
    graph.add_file_node(PathBuf::from("src/auth.py"));
    graph.add_file_node(PathBuf::from("src/utils.py"));
    graph.add_file_node(PathBuf::from("src/models/user.py"));
    graph
}

#[test]
fn test_update_file_imports_add_new() {
    let mut graph = setup_test_graph();
    let main_py = PathBuf::from("src/main.py");

    // Add initial import
    let initial_imports = vec![
        Import {
            module_name: "auth".to_string(),
            resolved_path: Some(PathBuf::from("src/auth.py")),
        },
    ];
    graph.update_file_imports(&main_py, initial_imports).unwrap();

    // Verify initial state
    let main_idx = *graph.path_to_idx.get(&main_py).unwrap();
    let auth_idx = *graph.path_to_idx.get(&PathBuf::from("src/auth.py")).unwrap();
    assert!(graph.graph.contains_edge(main_idx, auth_idx));
    assert_eq!(graph.graph.edges(main_idx).count(), 1);

    // Update with a new import
    let new_imports = vec![
        Import {
            module_name: "auth".to_string(),
            resolved_path: Some(PathBuf::from("src/auth.py")),
        },
        Import {
            module_name: "utils".to_string(),
            resolved_path: Some(PathBuf::from("src/utils.py")),
        },
    ];
    let result = graph.update_file_imports(&main_py, new_imports).unwrap();

    // Verify
    let utils_idx = *graph.path_to_idx.get(&PathBuf::from("src/utils.py")).unwrap();
    assert!(graph.graph.contains_edge(main_idx, auth_idx));
    assert!(graph.graph.contains_edge(main_idx, utils_idx));
    assert_eq!(graph.graph.edges(main_idx).count(), 2);
    assert!(result.edges_added == 1); // Because old edge was removed, then re-added implicitly
    assert!(result.edges_removed == 1);
    assert!(result.needs_pagerank_recalc);
}

#[test]
fn test_update_file_imports_remove_existing() {
    let mut graph = setup_test_graph();
    let main_py = PathBuf::from("src/main.py");
    let auth_py = PathBuf::from("src/auth.py");
    let utils_py = PathBuf::from("src/utils.py");

    // Initial imports
    let initial_imports = vec![
        Import { module_name: "auth".to_string(), resolved_path: Some(auth_py.clone()) },
        Import { module_name: "utils".to_string(), resolved_path: Some(utils_py.clone()) },
    ];
    graph.update_file_imports(&main_py, initial_imports).unwrap();
    let main_idx = *graph.path_to_idx.get(&main_py).unwrap();
    let auth_idx = *graph.path_to_idx.get(&auth_py).unwrap();
    let utils_idx = *graph.path_to_idx.get(&utils_py).unwrap();
    assert!(graph.graph.contains_edge(main_idx, auth_idx));
    assert!(graph.graph.contains_edge(main_idx, utils_idx));
    assert_eq!(graph.graph.edges(main_idx).count(), 2);

    // Update to remove one import
    let new_imports = vec![
        Import { module_name: "auth".to_string(), resolved_path: Some(auth_py.clone()) },
    ];
    let result = graph.update_file_imports(&main_py, new_imports).unwrap();

    // Verify
    assert!(graph.graph.contains_edge(main_idx, auth_idx));
    assert!(!graph.graph.contains_edge(main_idx, utils_idx)); // Should be removed
    assert_eq!(graph.graph.edges(main_idx).count(), 1);
    assert!(result.edges_added == 0); // Auth was re-added but counted as 0 new.
    assert!(result.edges_removed == 1); // One edge was truly removed.
    assert!(result.needs_pagerank_recalc);
}

#[test]
fn test_update_file_imports_no_change() {
    let mut graph = setup_test_graph();
    let main_py = PathBuf::from("src/main.py");
    let auth_py = PathBuf::from("src/auth.py");

    // Initial imports
    let initial_imports = vec![
        Import { module_name: "auth".to_string(), resolved_path: Some(auth_py.clone()) },
    ];
    graph.update_file_imports(&main_py, initial_imports.clone()).unwrap();
    let main_idx = *graph.path_to_idx.get(&main_py).unwrap();
    assert_eq!(graph.graph.edges(main_idx).count(), 1);

    // Update with same imports
    let result = graph.update_file_imports(&main_py, initial_imports).unwrap();

    // Verify no change
    assert_eq!(graph.graph.edges(main_idx).count(), 1);
    assert!(result.edges_added == 0);
    assert!(result.edges_removed == 0);
    assert!(!result.needs_pagerank_recalc); // Should not need recalc
}

#[test]
fn test_update_file_imports_non_existent_file() {
    let mut graph = setup_test_graph();
    let non_existent_file = PathBuf::from("src/non_existent.py");
    let imports = vec![]; // No imports for a non-existent file
    let result = graph.update_file_imports(&non_existent_file, imports);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), GraphError::NodeNotFound);
}

#[test]
fn test_lazy_pagerank_recalculation() {
    let mut graph = setup_test_graph();
    let main_py = PathBuf::from("src/main.py");
    let auth_py = PathBuf::from("src/auth.py");

    // Initial build should set dirty to false after implicit or explicit initial calc
    // Assuming RepoGraph::new() might call calculate_pagerank
    graph.pagerank_dirty = false; // Manually ensure clean state for test

    // Perform an update that requires recalc
    let new_imports = vec![
        Import { module_name: "auth".to_string(), resolved_path: Some(auth_py.clone()) },
    ];
    let result = graph.update_file_imports(&main_py, new_imports).unwrap();
    assert!(result.needs_pagerank_recalc);
    assert!(graph.pagerank_dirty); // Should be marked dirty

    // Accessing ranked files should trigger recalc
    let _ = graph.get_top_ranked_files(1);
    assert!(!graph.pagerank_dirty); // Should be clean after access
}

#[test]
#[ignore] // This is a benchmark, run with `cargo test -- --ignored`
fn test_update_performance() {
    let mut graph = RepoGraph::new();
    let num_files = 1000;
    let mut file_paths = Vec::new();

    // Setup a large graph
    for i in 0..num_files {
        let path = PathBuf::from(format!("src/file_{}.py", i));
        graph.add_file_node(path.clone());
        file_paths.push(path);
    }
    
    // Create some initial connections
    for i in 0..num_files {
        let target_idx = (i + 1) % num_files;
        let imports = vec![Import {
            module_name: format!("file_{}", target_idx),
            resolved_path: Some(file_paths[target_idx].clone()),
        }];
        graph.update_file_imports(&file_paths[i], imports).unwrap();
    }

    let file_to_update = file_paths[0].clone();
    let new_imports = vec![
        Import {
            module_name: "file_2".to_string(),
            resolved_path: Some(file_paths[2].clone()),
        },
        Import {
            module_name: "file_3".to_string(),
            resolved_path: Some(file_paths[3].clone()),
        },
    ];

    let start = std::time::Instant::now();
    let _ = graph.update_file_imports(&file_to_update, new_imports).unwrap();
    let duration = start.elapsed();

    println!("Graph update for one file in a 1000-node graph took: {:?}", duration);
    // Success criteria is < 10ms. This test will verify it's in the right ballpark.
    assert!(duration < std::time::Duration::from_millis(10));
}
```

**Verification:** Run `cargo test`

---

## Step 6: Python Integration Test

**File:** Create `python_shell/tests/test_graph_updates.py` (or extend existing)
**Time:** 45 minutes

### 6.1: Test Python API for `update_file_imports`

```python
# python_shell/tests/test_graph_updates.py

import pytest
import tempfile
import os
from pathlib import Path
from semantic_engine import RepoGraph, Import, GraphUpdateResult

# Helper function to create a dummy repository structure
@pytest.fixture
def dummy_repo(tmp_path):
    repo_root = tmp_path / "test_repo"
    repo_root.mkdir()
    
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.py").write_text("import src.auth\nimport src.utils")
    (repo_root / "src" / "auth.py").write_text("def authenticate(): pass")
    (repo_root / "src" / "utils.py").write_text("def helper(): pass")
    (repo_root / "src" / "models").mkdir()
    (repo_root / "src" / "models" / "user.py").write_text("class User: pass")
    
    return repo_root

def test_python_update_file_imports_add_remove(dummy_repo):
    graph = RepoGraph(str(dummy_repo))
    # Assuming initial build or scan for Graph is done in RepoGraph constructor
    # If not, need to call a `build` method here: graph.build_initial_graph()

    main_path = str(dummy_repo / "src" / "main.py")
    auth_path = str(dummy_repo / "src" / "auth.py")
    utils_path = str(dummy_repo / "src" / "utils.py")
    user_path = str(dummy_repo / "src" / "models" / "user.py")

    # Initial state verification (check graph contains nodes for these paths)
    assert graph.has_file(main_path)
    assert graph.has_file(auth_path)
    assert graph.has_file(utils_path)

    # First update: `main.py` initially imports `auth` and `utils`
    # Simulate extraction of these imports
    initial_imports = [
        Import(module_name="src.auth", resolved_path=auth_path),
        Import(module_name="src.utils", resolved_path=utils_path),
    ]
    # Assuming `RepoGraph` constructor or a `scan_and_build` method
    # would have populated the initial imports.
    # For this test, we simulate an update after the fact.
    # The current rust implementation for `update_file_imports` will first clear ALL
    # existing `Import` edges from `main_path` and then add the new ones.
    # So for testing, we can directly call `update_file_imports` with the 'current'
    # imports for the file.

    # Simulating the actual step: modify main.py to remove utils and add models.user
    (dummy_repo / "src" / "main.py").write_text("import src.auth\nfrom src.models import user")
    
    new_imports = [
        Import(module_name="src.auth", resolved_path=auth_path),
        Import(module_name="src.models.user", resolved_path=user_path),
    ]

    update_result = graph.update_file_imports(main_path, new_imports)

    assert isinstance(update_result, GraphUpdateResult)
    assert update_result.edges_added == 1 # utils removed, user added, auth is re-added
    assert update_result.edges_removed == 1 # utils was removed
    assert update_result.needs_pagerank_recalc is True

    # Verify graph state from Python side (e.g., via `get_dependencies` method)
    # Assuming get_dependencies exists and returns list of PathBuf-like strings
    dependencies = graph.get_dependencies(main_path)
    print(f"Main.py dependencies: {dependencies}")
    assert auth_path in dependencies
    assert user_path in dependencies
    assert utils_path not in dependencies

    # Test lazy PageRank recalculation
    # Before getting top ranked files, pagerank_dirty should be true (Rust side)
    # After getting top ranked files, it should be false
    top_ranked = graph.get_top_ranked_files(3)
    print(f"Top ranked files: {top_ranked}")
    # Add assertions about expected top ranked files based on the new graph structure
    assert any(user_path in path for path, _ in top_ranked)

    # Test no-change update
    # The previous `update_file_imports` already set `pagerank_dirty` to False when `get_top_ranked_files` was called
    # So we call it again with the same imports
    no_change_result = graph.update_file_imports(main_path, new_imports)
    assert no_change_result.edges_added == 0
    assert no_change_result.edges_removed == 0
    assert no_change_result.needs_pagerank_recalc is False


def test_python_pagerank_recalculation(dummy_repo):
    graph = RepoGraph(str(dummy_repo))

    # Initially, force PageRank to be up-to-date (or assume it is after initial build)
    graph.ensure_pagerank_up_to_date()

    main_path = str(dummy_repo / "src" / "main.py")
    auth_path = str(dummy_repo / "src" / "auth.py")
    
    # Modify main.py to change imports
    (dummy_repo / "src" / "main.py").write_text("import src.auth") # Remove utils
    
    new_imports = [
        Import(module_name="src.auth", resolved_path=auth_path),
    ]

    update_result = graph.update_file_imports(main_path, new_imports)
    assert update_result.needs_pagerank_recalc is True # Should be true

    # At this point, `pagerank_dirty` should be true on the Rust side
    # Accessing PageRank-dependent methods should trigger recalculation
    top_ranked = graph.get_top_ranked_files(1)
    # Check that after calling `get_top_ranked_files`, `pagerank_dirty` is reset
    # This might require another wrapper on the Python side to expose the `pagerank_dirty` state
    # For now, we assume the `ensure_pagerank_up_to_date` is working correctly on Rust side. 
    
    assert top_ranked[0][0] == auth_path or top_ranked[0][0] == main_path # Expecting main or auth to be high
```

**Verification:** Run `pytest python_shell/tests/test_graph_updates.py`

---

## Troubleshooting Guide

### Issue: Edges Not Updating Correctly

**Symptoms:** Graph queries show old dependencies or missing new ones.

**Debug:**
1.  **Rust side:** Add `println!` statements within `update_file_imports` to see which edges are being removed and added.
2.  **`new_imports` correctness:** Verify that the `new_imports` vector passed to the Rust function accurately reflects the current imports of the file after re-parsing.

**Common Causes:**
1.  **Incorrect `retain_edges` logic:** Ensure it's only removing `Import` edges from the specific node, not other types or nodes.
2.  **Duplicate edge creation:** If adding edges without checking `contains_edge`, duplicates might be created if `retain_edges` isn't fully removing old ones.
3.  **`resolved_path` issues:** Imports might not be resolving to correct `PathBuf`s in the `ImportResolver`, leading to `target_node_idx` being `None`.

### Issue: PageRank Recalculating Too Often (or Not At All)

**Symptoms:** Performance hit on every small change, or PageRank scores are stale.

**Debug:**
1.  **`needs_pagerank_recalc` logic:** Review the conditions for setting `significant_change = true` in `update_file_imports`. Is it too aggressive or too lenient?
2.  **`pagerank_dirty` flag:** Verify that `pagerank_dirty` is correctly set to `true` when structural changes occur and reset to `false` after `calculate_pagerank` is called.

**Common Causes:**
1.  **Overly aggressive `significant_change`:** If any minor text change causes a "significant change" flag, PageRank will run too often. Focus on actual import structure changes.
2.  **Missing `self.pagerank_dirty = true`:** If structural changes don't mark the graph dirty, PageRank won't recalculate.
3.  **Missing `self.pagerank_dirty = false`:** If `calculate_pagerank` doesn't reset the flag, it will always recalculate.

---

## Completion Checklist

- [ ] `UpdateResult` struct defined in Rust.
- [ ] `RepoGraph::update_file_imports` implemented in Rust.
- [ ] `pagerank_dirty` flag added to `RepoGraph` and initialized.
- [ ] Lazy PageRank logic implemented in `get_top_ranked_files` (and other dependent methods).
- [ ] Python `PyImport` and `PyGraphUpdateResult` classes added to `lib.rs`.
- [ ] Python wrapper for `update_file_imports` in `lib.rs` (`PyRepoGraph`).
- [ ] All new Rust tests for graph update logic (`cargo test`).
- [ ] Performance benchmark test passes (`cargo test -- --ignored`).
- [ ] All new Python integration tests for graph update API (`pytest`).
- [ ] Performance criteria met (updates < 10ms).
- [ ] No memory leaks observed.
- [ ] Documentation updated for new methods.

---

## Next Phase

After Phase 4.3 is complete:

**Phase 4.4**: Multi-Tier Update Strategy
- Classify changes into different tiers (Local, FileScope, GraphScope, FullRebuild).
- Implement tiered update execution to optimize performance based on change magnitude.

```