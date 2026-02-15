# Project Atlas: Comprehensive Gap Remediation Implementation Plan

**Document Purpose:** Detailed implementation plan to address all critical gaps identified in the Codebase Review document and align with Project Atlas's roadmap goals.

**Date:** January 29, 2026  
**Target Completion:** 4-6 Stages

---

## Executive Summary

Based on the comprehensive codebase review, Project Atlas has a solid architectural foundation with a hybrid Rust/Python approach. However, critical gaps exist in three areas:

1. **CRITICAL (P0):** Testing gaps for SymbolUsage edge updates could lead to silent graph corruption
2. **HIGH (P1):** Performance bottlenecks in `classify_change` and inefficient file I/O in `update_file`
3. **MEDIUM (P2):** Python orchestration layer remains unimplemented, blocking end-to-end functionality

This plan provides step-by-step implementation details to close these gaps while maintaining development momentum.

---

## Phase 1: Critical Testing Infrastructure (Stage 1)
**Priority:** P0 - URGENT  
**Risk:** HIGH - Current gaps could cause silent graph corruption

### 1.1 Expand `test_graph_updates.rs` - SymbolUsage Testing

**Problem:** Current tests only cover `EdgeKind::Import` changes. The complex `SymbolUsage` edge logic (which forms the semantic understanding of the codebase) is completely untested.

**Impact:** Any bug in symbol edge handling will silently corrupt the graph, causing the agent to misunderstand code architecture.

#### Implementation Steps

**Step 1.1.1: Symbol Definition Change Test**
```rust
// Location: rust_core/tests/test_graph_updates.rs

#[test]
fn test_symbol_definition_removal() {
    // SETUP
    let dir = tempdir().unwrap();
    let root = dir.path();
    
    // Create file structure
    fs::create_dir_all(root.join("src")).unwrap();
    
    // File A defines func_a
    fs::write(
        root.join("src/module_a.py"),
        "def func_a():\n    pass"
    ).unwrap();
    
    // File B calls func_a
    fs::write(
        root.join("src/module_b.py"),
        "from src.module_a import func_a\nfunc_a()"
    ).unwrap();
    
    let files = vec![
        root.join("src/module_a.py"),
        root.join("src/module_b.py"),
    ];
    
    let mut graph = RepoGraph::new_with_import_resolution(root);
    graph.build_complete(&files, root);
    
    // VERIFY: Initial state has symbol edge B -> A
    let module_a = root.join("src/module_a.py");
    let module_b = root.join("src/module_b.py");
    
    let initial_edges = graph.get_outgoing_dependencies(&module_b);
    assert!(
        initial_edges.iter().any(|(path, kind)| {
            path == &module_a && *kind == EdgeKind::SymbolUsage
        }),
        "Expected SymbolUsage edge from B to A"
    );
    
    // ACTION: Rename func_a to func_renamed in File A
    fs::write(
        root.join("src/module_a.py"),
        "def func_renamed():\n    pass"
    ).unwrap();
    
    let result = graph.update_file(&module_a).unwrap();
    
    // ASSERT: Symbol edge B -> A is removed
    let final_edges = graph.get_outgoing_dependencies(&module_b);
    assert!(
        !final_edges.iter().any(|(path, kind)| {
            path == &module_a && *kind == EdgeKind::SymbolUsage
        }),
        "SymbolUsage edge should be removed after definition changed"
    );
    
    // ASSERT: SymbolIndex no longer contains func_a from File A
    assert!(
        !graph.symbol_index.definitions.contains_key("func_a"),
        "func_a should no longer be in symbol index"
    );
    
    // ASSERT: Update result indicates edges were removed
    assert!(result.edges_removed > 0);
    assert!(result.needs_pagerank_recalc);
}

#[test]
fn test_symbol_definition_addition() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    
    fs::create_dir_all(root.join("src")).unwrap();
    
    // File A initially empty
    fs::write(root.join("src/module_a.py"), "# Empty file").unwrap();
    
    // File B calls func_a (currently undefined)
    fs::write(
        root.join("src/module_b.py"),
        "from src.module_a import func_a\nfunc_a()"
    ).unwrap();
    
    let files = vec![
        root.join("src/module_a.py"),
        root.join("src/module_b.py"),
    ];
    
    let mut graph = RepoGraph::new_with_import_resolution(root);
    graph.build_complete(&files, root);
    
    let module_a = root.join("src/module_a.py");
    let module_b = root.join("src/module_b.py");
    
    // VERIFY: No SymbolUsage edge exists initially
    let initial_edges = graph.get_outgoing_dependencies(&module_b);
    assert_eq!(
        initial_edges.iter()
            .filter(|(_, kind)| *kind == EdgeKind::SymbolUsage)
            .count(),
        0,
        "Should have no SymbolUsage edges initially"
    );
    
    // ACTION: Add func_a definition to File A
    fs::write(
        root.join("src/module_a.py"),
        "def func_a():\n    pass"
    ).unwrap();
    
    let result = graph.update_file(&module_a).unwrap();
    
    // ASSERT: SymbolUsage edge B -> A is created
    let final_edges = graph.get_outgoing_dependencies(&module_b);
    assert!(
        final_edges.iter().any(|(path, kind)| {
            path == &module_a && *kind == EdgeKind::SymbolUsage
        }),
        "SymbolUsage edge should be created after definition added"
    );
    
    assert!(result.edges_added > 0);
    assert!(result.needs_pagerank_recalc);
}
```

**Step 1.1.2: Symbol Usage Change Test**
```rust
#[test]
fn test_symbol_usage_change() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    
    fs::create_dir_all(root.join("src")).unwrap();
    
    // File A defines func_a
    fs::write(
        root.join("src/module_a.py"),
        "def func_a():\n    pass"
    ).unwrap();
    
    // File C defines func_c
    fs::write(
        root.join("src/module_c.py"),
        "def func_c():\n    pass"
    ).unwrap();
    
    // File B initially calls func_a
    fs::write(
        root.join("src/module_b.py"),
        "from src.module_a import func_a\nfunc_a()"
    ).unwrap();
    
    let files = vec![
        root.join("src/module_a.py"),
        root.join("src/module_b.py"),
        root.join("src/module_c.py"),
    ];
    
    let mut graph = RepoGraph::new_with_import_resolution(root);
    graph.build_complete(&files, root);
    
    let module_a = root.join("src/module_a.py");
    let module_b = root.join("src/module_b.py");
    let module_c = root.join("src/module_c.py");
    
    // VERIFY: Edge B -> A exists
    let initial_edges = graph.get_outgoing_dependencies(&module_b);
    assert!(
        initial_edges.iter().any(|(path, kind)| {
            path == &module_a && *kind == EdgeKind::SymbolUsage
        }),
        "Expected initial edge B -> A"
    );
    
    // ACTION: Change File B to call func_c instead
    fs::write(
        root.join("src/module_b.py"),
        "from src.module_c import func_c\nfunc_c()"
    ).unwrap();
    
    let result = graph.update_file(&module_b).unwrap();
    
    // ASSERT: Edge B -> A is removed
    let final_edges = graph.get_outgoing_dependencies(&module_b);
    assert!(
        !final_edges.iter().any(|(path, kind)| {
            path == &module_a && *kind == EdgeKind::SymbolUsage
        }),
        "Old edge B -> A should be removed"
    );
    
    // ASSERT: New edge B -> C is created
    assert!(
        final_edges.iter().any(|(path, kind)| {
            path == &module_c && *kind == EdgeKind::SymbolUsage
        }),
        "New edge B -> C should be created"
    );
    
    assert!(result.edges_added > 0);
    assert!(result.edges_removed > 0);
    assert!(result.needs_pagerank_recalc);
}
```

**Step 1.1.3: Multiple Definitions (Collision) Test**
```rust
#[test]
fn test_symbol_collision_handling() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    
    fs::create_dir_all(root.join("src")).unwrap();
    
    // File A defines func_common
    fs::write(
        root.join("src/module_a.py"),
        "def func_common():\n    pass"
    ).unwrap();
    
    // File C also defines func_common (collision)
    fs::write(
        root.join("src/module_c.py"),
        "def func_common():\n    pass"
    ).unwrap();
    
    // File B uses func_common (ambiguous)
    fs::write(
        root.join("src/module_b.py"),
        "func_common()"
    ).unwrap();
    
    let files = vec![
        root.join("src/module_a.py"),
        root.join("src/module_b.py"),
        root.join("src/module_c.py"),
    ];
    
    let mut graph = RepoGraph::new_with_import_resolution(root);
    graph.build_complete(&files, root);
    
    let module_a = root.join("src/module_a.py");
    let module_b = root.join("src/module_b.py");
    let module_c = root.join("src/module_c.py");
    
    // VERIFY: B has edges to BOTH A and C (collision handling)
    let edges = graph.get_outgoing_dependencies(&module_b);
    let symbol_edges: Vec<_> = edges.iter()
        .filter(|(_, kind)| *kind == EdgeKind::SymbolUsage)
        .collect();
    
    assert!(
        symbol_edges.iter().any(|(path, _)| path == &module_a),
        "Should have edge to module_a"
    );
    assert!(
        symbol_edges.iter().any(|(path, _)| path == &module_c),
        "Should have edge to module_c"
    );
    
    // VERIFY: SymbolIndex tracks both definitions
    let defs = graph.symbol_index.definitions.get("func_common").unwrap();
    assert_eq!(defs.len(), 2, "Should track both definitions");
    assert!(defs.contains(&module_a));
    assert!(defs.contains(&module_c));
}
```

**Step 1.1.4: File Creation/Deletion Tests**
```rust
#[test]
fn test_file_creation_with_symbols() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    
    fs::create_dir_all(root.join("src")).unwrap();
    
    // Existing file with a function
    fs::write(
        root.join("src/old_file.py"),
        "def old_func():\n    pass"
    ).unwrap();
    
    let files = vec![root.join("src/old_file.py")];
    
    let mut graph = RepoGraph::new_with_import_resolution(root);
    graph.build_complete(&files, root);
    
    let old_file = root.join("src/old_file.py");
    let new_file = root.join("src/new_file.py");
    
    // ACTION: Create new file that uses old_func and defines new_func
    fs::write(
        &new_file,
        "from src.old_file import old_func\ndef new_func():\n    old_func()"
    ).unwrap();
    
    // Simulate file creation by adding it to the graph
    // Note: This would typically be done through a higher-level API
    graph.build(&[new_file.clone()]);
    graph.build_import_edges();
    graph.build_semantic_edges();
    
    // VERIFY: New node exists
    assert!(graph.path_to_idx.contains_key(&new_file));
    
    // VERIFY: Edge from new_file -> old_file exists
    let edges = graph.get_outgoing_dependencies(&new_file);
    assert!(
        edges.iter().any(|(path, _)| path == &old_file),
        "Should have edge to old_file"
    );
    
    // VERIFY: Symbol index updated
    let defs = graph.symbol_index.definitions.get("new_func").unwrap();
    assert!(defs.contains(&new_file));
}

#[test]
fn test_file_deletion_cleanup() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    
    fs::create_dir_all(root.join("src")).unwrap();
    
    // File A defines func_a
    fs::write(
        root.join("src/module_a.py"),
        "def func_a():\n    pass"
    ).unwrap();
    
    // File B calls func_a
    fs::write(
        root.join("src/module_b.py"),
        "from src.module_a import func_a\nfunc_a()"
    ).unwrap();
    
    let files = vec![
        root.join("src/module_a.py"),
        root.join("src/module_b.py"),
    ];
    
    let mut graph = RepoGraph::new_with_import_resolution(root);
    graph.build_complete(&files, root);
    
    let module_b = root.join("src/module_b.py");
    
    // VERIFY: Initial state
    let initial_node_count = graph.graph.node_count();
    let initial_edge_count = graph.graph.edge_count();
    
    // ACTION: Delete File B
    // This would need a new method: graph.remove_file(&module_b)
    // For now, test the conceptual requirement
    
    // FUTURE: Implement graph.remove_file method
    // let result = graph.remove_file(&module_b).unwrap();
    
    // EXPECTED ASSERTIONS (once implemented):
    // assert!(!graph.path_to_idx.contains_key(&module_b));
    // assert!(graph.graph.node_count() == initial_node_count - 1);
    // assert!(graph.graph.edge_count() < initial_edge_count);
}
```

**Deliverables:**
- [ ] All 6+ new test functions added to `test_graph_updates.rs`
- [ ] All tests pass with current implementation
- [ ] Test coverage for SymbolUsage edges >= 80%
- [ ] Documentation comments explaining what each test validates

**Success Criteria:**
- Tests catch intentional bugs when symbol update logic is temporarily broken
- CI pipeline includes these tests and fails on symbol edge corruption
- Test runtime < 5 seconds for entire suite

---

### 1.2 Add Symbol Index Integrity Tests

**Problem:** The `SymbolIndex` is the source of truth for symbol relationships. If it becomes inconsistent, the entire graph is corrupted.

#### Implementation Steps

**Step 1.2.1: Consistency Validation Helper**
```rust
// Location: rust_core/src/graph.rs

impl RepoGraph {
    /// Validates the internal consistency of the symbol index and graph.
    /// Used in tests and can be called after updates in debug mode.
    #[cfg(test)]
    pub fn validate_consistency(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        
        // Check 1: Every definition in symbol_index points to a real file node
        for (symbol, def_paths) in &self.symbol_index.definitions {
            for path in def_paths {
                if !self.path_to_idx.contains_key(path) {
                    errors.push(format!(
                        "Symbol '{}' has definition in non-existent file: {}",
                        symbol, path.display()
                    ));
                }
            }
        }
        
        // Check 2: Every file in symbol_index.usages exists in the graph
        for (path, _) in &self.symbol_index.usages {
            if !self.path_to_idx.contains_key(path) {
                errors.push(format!(
                    "Symbol usage tracked for non-existent file: {}",
                    path.display()
                ));
            }
        }
        
        // Check 3: File node definitions match symbol_index
        for (path, &node_idx) in &self.path_to_idx {
            let node = &self.graph[node_idx];
            
            // Every definition in the node should be in symbol_index
            for def in &node.definitions {
                if let Some(paths) = self.symbol_index.definitions.get(def) {
                    if !paths.contains(path) {
                        errors.push(format!(
                            "File {} defines '{}' but not tracked in symbol_index",
                            path.display(), def
                        ));
                    }
                } else {
                    errors.push(format!(
                        "File {} defines '{}' but not in symbol_index at all",
                        path.display(), def
                    ));
                }
            }
        }
        
        // Check 4: SymbolUsage edges correspond to symbol_index entries
        for edge_idx in self.graph.edge_indices() {
            if self.graph[edge_idx] == EdgeKind::SymbolUsage {
                let (source_idx, target_idx) = self.graph.edge_endpoints(edge_idx).unwrap();
                let source_path = &self.graph[source_idx].path;
                let target_path = &self.graph[target_idx].path;
                
                // Source should have some usages
                if !self.symbol_index.usages.contains_key(source_path) {
                    errors.push(format!(
                        "SymbolUsage edge from {} but no usages tracked",
                        source_path.display()
                    ));
                }
                
                // Target should define something the source uses
                let target_node = &self.graph[target_idx];
                if let Some(source_usages) = self.symbol_index.usages.get(source_path) {
                    let has_matching_symbol = target_node.definitions.iter()
                        .any(|def| source_usages.contains(def));
                    
                    if !has_matching_symbol {
                        errors.push(format!(
                            "SymbolUsage edge {} -> {} but no matching symbol found",
                            source_path.display(), target_path.display()
                        ));
                    }
                }
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
```

**Step 1.2.2: Add Consistency Checks to Update Tests**
```rust
// Location: rust_core/tests/test_graph_updates.rs

#[test]
fn test_update_maintains_consistency() {
    let (_dir, mut graph) = setup_test_repo_graph();
    
    // Initial state should be consistent
    graph.validate_consistency().expect("Initial graph should be consistent");
    
    // Perform multiple updates
    let files_to_update = vec![
        graph.import_resolver.as_ref().unwrap().project_root.join("src/main.py"),
        graph.import_resolver.as_ref().unwrap().project_root.join("src/auth.py"),
        graph.import_resolver.as_ref().unwrap().project_root.join("src/utils.py"),
    ];
    
    for file in &files_to_update {
        // Make some change
        let new_content = "# Modified content\nimport src.auth";
        fs::write(file, new_content).unwrap();
        
        graph.update_file(file).unwrap();
        
        // After each update, graph should remain consistent
        graph.validate_consistency().expect(&format!(
            "Graph inconsistent after updating {}",
            file.display()
        ));
    }
}

#[test]
fn test_batch_updates_maintain_consistency() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    
    // Create a complex graph
    fs::create_dir_all(root.join("src")).unwrap();
    for i in 0..20 {
        fs::write(
            root.join(format!("src/module_{}.py", i)),
            format!("def func_{}():\n    pass", i)
        ).unwrap();
    }
    
    let files: Vec<_> = (0..20)
        .map(|i| root.join(format!("src/module_{}.py", i)))
        .collect();
    
    let mut graph = RepoGraph::new_with_import_resolution(root);
    graph.build_complete(&files, root);
    
    graph.validate_consistency().expect("Initial build should be consistent");
    
    // Simulate rapid file changes
    for i in 0..20 {
        let file = root.join(format!("src/module_{}.py", i));
        let target = (i + 1) % 20;
        fs::write(
            &file,
            format!("from src.module_{} import func_{}\ndef func_{}():\n    func_{}()", target, target, i, target)
        ).unwrap();
        
        graph.update_file(&file).unwrap();
    }
    
    graph.validate_consistency().expect("Graph should remain consistent after batch updates");
}
```

**Deliverables:**
- [ ] `validate_consistency()` method implemented
- [ ] Integration into all existing update tests
- [ ] Separate test specifically for consistency checks
- [ ] Optional: Debug mode assertion after each update

**Success Criteria:**
- Validation catches all types of inconsistencies
- Performance overhead < 10% in debug mode
- Zero false positives in tests

---

## Phase 2: Performance Optimization (Stage 2)
**Priority:** P1 - HIGH  
**Risk:** MEDIUM - Current bottlenecks limit scalability

### 2.1 Refactor `FileNode` to Cache Comparison Data

**Problem:** `classify_change()` reads files from disk to compare old vs. new imports, causing unnecessary I/O and race conditions.

#### Implementation Steps

**Step 2.1.1: Extend FileNode with Cached Hashes**
```rust
// Location: rust_core/src/graph.rs

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct FileNode {
    pub path: PathBuf,
    pub definitions: Vec<String>,
    pub usages: Vec<String>,
    pub rank: f64,
    
    // NEW: Cached data for fast change classification
    pub imports_hash: u64,
    pub definitions_hash: u64,
    pub content_hash: u64,  // Full content hash for change detection
}

impl FileNode {
    /// Creates a new FileNode with computed hashes
    pub fn new(
        path: PathBuf,
        definitions: Vec<String>,
        usages: Vec<String>,
        imports: &HashSet<PathBuf>,
        content: &str,
    ) -> Self {
        Self {
            path,
            definitions: definitions.clone(),
            usages: usages.clone(),
            rank: 0.0,
            imports_hash: Self::hash_imports(imports),
            definitions_hash: Self::hash_definitions(&definitions),
            content_hash: Self::hash_content(content),
        }
    }
    
    fn hash_imports(imports: &HashSet<PathBuf>) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut sorted: Vec<_> = imports.iter().collect();
        sorted.sort();
        for import in sorted {
            import.hash(&mut hasher);
        }
        hasher.finish()
    }
    
    fn hash_definitions(definitions: &[String]) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut sorted = definitions.to_vec();
        sorted.sort();
        for def in sorted {
            def.hash(&mut hasher);
        }
        hasher.finish()
    }
    
    fn hash_content(content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
}
```

**Step 2.1.2: Refactor `classify_change` to Use Cached Hashes**
```rust
// Location: rust_core/src/graph.rs

impl RepoGraph {
    /// Classifies the magnitude of change without disk I/O.
    /// Now accepts the new content as a parameter.
    pub fn classify_change(
        &self,
        file_path: &Path,
        new_source: &str
    ) -> Result<UpdateTier, GraphError> {
        let node_idx = *self
            .path_to_idx
            .get(file_path)
            .ok_or_else(|| GraphError::NodeNotFound(file_path.to_path_buf()))?;
        
        let old_node = self.graph.node_weight(node_idx)
            .ok_or_else(|| GraphError::NodeNotFound(file_path.to_path_buf()))?;
        
        // Quick content check
        let new_content_hash = FileNode::hash_content(new_source);
        if new_content_hash == old_node.content_hash {
            return Ok(UpdateTier::Local);  // No change at all
        }
        
        // Parse new file to get new symbols and imports
        let lang = SupportedLanguage::from_extension(
            file_path.extension().and_then(|s| s.to_str()).unwrap_or(""),
        );
        if lang == SupportedLanguage::Unknown {
            return Ok(UpdateTier::Local);
        }
        
        let mut parser = TreeSitterParser::new();
        parser
            .set_language(lang.get_parser()
                .ok_or_else(|| GraphError::ParseError(file_path.to_path_buf()))?)
            .map_err(|_| GraphError::ParseError(file_path.to_path_buf()))?;
        
        let tree = parser
            .parse(new_source, None)
            .ok_or_else(|| GraphError::ParseError(file_path.to_path_buf()))?;
        
        let harvester = SymbolHarvester::new(lang);
        let (new_defs, new_usages) = harvester.harvest(&tree, new_source.as_bytes());
        
        // Get new imports (no disk I/O!)
        let new_imports = self._get_file_imports(file_path, new_source);
        
        // Compare hashes instead of full data
        let new_defs_hash = FileNode::hash_definitions(&new_defs);
        let new_imports_hash = FileNode::hash_imports(&new_imports);
        
        // Classification logic using hashes
        if new_imports_hash != old_node.imports_hash {
            return Ok(UpdateTier::GraphScope);  // Import changed
        }
        
        if new_defs_hash != old_node.definitions_hash {
            return Ok(UpdateTier::FileScope);  // Definitions changed
        }
        
        // Usages changed but not definitions or imports
        Ok(UpdateTier::Local)
    }
}
```

**Step 2.1.3: Update `build` Method to Use New Constructor**
```rust
// Location: rust_core/src/graph.rs

impl RepoGraph {
    pub fn build(&mut self, paths: &[PathBuf]) {
        let parsed_files: Vec<_> = paths
            .par_iter()
            .filter_map(|path| {
                let source_code = match fs::read_to_string(path) {
                    Ok(s) => s,
                    Err(_) => return None,
                };
                
                // ... existing parsing logic ...
                
                let imports = self._get_file_imports(path, &source_code);
                
                Some((
                    path.clone(),
                    definitions,
                    usages,
                    imports,
                    source_code,  // Pass content for hashing
                ))
            })
            .collect();
        
        for (path, definitions, usages, imports, content) in parsed_files {
            let file_node = FileNode::new(
                path.clone(),
                definitions.clone(),
                usages.clone(),
                &imports,
                &content,  // NEW: compute hashes at creation
            );
            
            let node_idx = self.graph.add_node(file_node);
            self.path_to_idx.insert(path.clone(), node_idx);
            
            // ... rest of symbol index population ...
        }
    }
}
```

**Deliverables:**
- [ ] FileNode extended with hash fields
- [ ] `classify_change` refactored to use hashes
- [ ] `build` method updated to compute hashes
- [ ] `update_file` updated to recompute hashes
- [ ] Benchmarks showing I/O elimination

**Success Criteria:**
- `classify_change` makes zero disk reads
- Performance improvement >= 5x for change classification
- All existing tests pass with new implementation
- No race conditions possible

---

### 2.2 Refactor `update_file` API to Accept Content

**Problem:** Current signature forces redundant file read. Python caller already has content.

#### Implementation Steps

**Step 2.2.1: Update Rust Core API**
```rust
// Location: rust_core/src/graph.rs

impl RepoGraph {
    /// Updates a file in the graph using provided content (no disk I/O).
    pub fn update_file(
        &mut self,
        file_path: &PathBuf,
        new_content: &str,  // NEW: content parameter
    ) -> Result<UpdateResult, GraphError> {
        // Classify the change first
        let tier = self.classify_change(file_path, new_content)?;
        
        if tier == UpdateTier::Local {
            // Only update content hash, no graph changes
            if let Some(&node_idx) = self.path_to_idx.get(file_path) {
                let node = self.graph.node_weight_mut(node_idx).unwrap();
                node.content_hash = FileNode::hash_content(new_content);
            }
            
            return Ok(UpdateResult {
                edges_added: 0,
                edges_removed: 0,
                needs_pagerank_recalc: false,
            });
        }
        
        // ... rest of existing update logic, but using new_content instead of reading from disk ...
        
        // Parse the NEW content (not from disk)
        let lang = SupportedLanguage::from_extension(
            file_path.extension().and_then(|s| s.to_str()).unwrap_or(""),
        );
        let mut parser = TreeSitterParser::new();
        parser.set_language(lang.get_parser().unwrap())?;
        let tree = parser.parse(new_content, None)
            .ok_or_else(|| GraphError::ParseError(file_path.clone()))?;
        
        let harvester = SymbolHarvester::new(lang);
        let (new_defs, new_usages) = harvester.harvest(&tree, new_content.as_bytes());
        
        // ... rest of update logic ...
        
        // Update hashes in FileNode
        let new_imports = self._get_file_imports(file_path, new_content);
        let node = self.graph.node_weight_mut(node_idx).unwrap();
        node.imports_hash = FileNode::hash_imports(&new_imports);
        node.definitions_hash = FileNode::hash_definitions(&new_defs);
        node.content_hash = FileNode::hash_content(new_content);
        
        Ok(result)
    }
}
```

**Step 2.2.2: Update Python Bindings**
```rust
// Location: rust_core/src/lib.rs

#[pymethods]
impl PyRepoGraph {
    /// Update a file with provided content (avoids redundant file read).
    fn update_file(
        &mut self,
        file_path: &str,
        content: &str  // NEW: required parameter
    ) -> PyResult<PyGraphUpdateResult> {
        let path = PathBuf::from(file_path);
        let result = self.graph.update_file(&path, content)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to update file: {}", e)))?;
        Ok(result.into())
    }
}
```

**Step 2.2.3: Update All Test Calls**
```rust
// Location: rust_core/tests/test_graph_updates.rs

#[test]
fn test_update_file_add_import() {
    let (_dir, mut graph) = setup_test_repo_graph();
    let utils_path = // ...
    
    // NEW: Read content and pass it
    let new_content = "import src.auth";
    let result = graph.update_file(&utils_path, new_content).unwrap();
    
    // ... assertions ...
}
```

**Deliverables:**
- [ ] All `update_file` signatures updated
- [ ] Python bindings updated
- [ ] All tests updated to pass content
- [ ] Documentation updated with new API

**Success Criteria:**
- Zero file reads in `update_file`
- Python->Rust boundary optimized
- All tests pass
- Backward compatibility maintained where possible

---

### 2.3 Optimize Clone Usage with Arc<String>

**Problem:** Excessive string cloning in hot paths causes heap pressure.

**Note:** This is a lower priority optimization to be done after profiling shows it's necessary.

#### Implementation Steps (Deferred)

**Step 2.3.1: Profile Current Performance**
```bash
# Use cargo flamegraph to identify clone hotspots
cargo install flamegraph
cargo flamegraph --bench incremental_benchmark
```

**Step 2.3.2: Conditionally Implement Arc Wrapper** *(Only if profiling shows >10% time in string clones)*
```rust
// Location: rust_core/src/symbol_table.rs

use std::sync::Arc;

type Symbol = Arc<String>;

pub struct SymbolIndex {
    pub definitions: HashMap<Symbol, Vec<PathBuf>>,
    pub usages: HashMap<PathBuf, Vec<Symbol>>,
}

// Benefits: Clone becomes pointer copy instead of heap allocation
// Costs: Slightly more complex API, Arc overhead
```

**Deliverables:**
- [ ] Profiling data captured
- [ ] Decision doc on whether to proceed
- [ ] If yes: Implementation + benchmarks showing improvement

**Success Criteria:**
- Only implement if profiling shows >=10% time in clones
- Improvement >= 20% in update_file benchmark

---

## Phase 3: Python Orchestration Layer (Stages 3-4)
**Priority:** P2 - MEDIUM  
**Risk:** LOW - Architecture is clear, mostly plumbing

### 3.1 Implement Core Agent Loop

**Problem:** Python shell is currently a placeholder.

#### Implementation Steps

**Step 3.1.1: Basic Agent Structure**
```python
# Location: python_shell/atlas/agent.py

from pathlib import Path
from typing import Optional
import time

from semantic_engine import RepoGraph, FileWatcher, scan_repository
from .context import ContextManager
from .tools import FileTools


class AtlasAgent:
    """
    Main agent orchestrator that combines graph analysis,
    context management, and LLM interaction.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.repo_graph = RepoGraph(str(project_root))
        self.watcher: Optional[FileWatcher] = None
        self.context_manager = ContextManager()
        self.file_tools = FileTools(project_root)
        self.running = False
        
    def initialize(self):
        """Perform initial repository scan and graph build."""
        print(f"Initializing Atlas for {self.project_root}")
        
        print("  [1/3] Scanning repository...")
        files = scan_repository(str(self.project_root))
        print(f"      Found {len(files)} files")
        
        print("  [2/3] Building dependency graph...")
        self.repo_graph.build_complete(files, str(self.project_root))
        
        print("  [3/3] Calculating PageRank...")
        self.repo_graph.ensure_pagerank_up_to_date()
        
        # Display top files
        top_files = self.repo_graph.get_top_ranked_files(5)
        print("\n  ✓ Repository indexed! Top architectural files:")
        for i, (path, rank) in enumerate(top_files, 1):
            print(f"    {i}. {Path(path).name} (rank: {rank:.3f})")
        
        print("\n  Ready! Starting file watcher...")
        self.watcher = FileWatcher(
            str(self.project_root),
            extensions=["py", "rs", "js", "ts", "go"],
            ignored_dirs=["node_modules", "target", ".git", "__pycache__"]
        )
        
    def run(self):
        """Main agent event loop."""
        if self.watcher is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        self.running = True
        print("\nAgent running. Watching for file changes...\n")
        
        try:
            while self.running:
                # Poll for file system events
                events = self.watcher.poll_events()
                for event in events:
                    self._handle_file_event(event)
                
                # TODO: Handle user input (stdin or socket)
                # For now, just sleep
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Graceful shutdown."""
        self.running = False
        if self.watcher:
            self.watcher.stop()
    
    def _handle_file_event(self, event):
        """Process a single file system change event."""
        print(f"  Detected: {event}")
        
        if event.event_type == "modified":
            self._handle_file_modified(Path(event.path))
        elif event.event_type == "created":
            self._handle_file_created(Path(event.path))
        elif event.event_type == "deleted":
            self._handle_file_deleted(Path(event.path))
        # TODO: Handle "renamed"
    
    def _handle_file_modified(self, file_path: Path):
        """Handle file modification event."""
        try:
            # Read new content
            content = file_path.read_text()
            
            # Update graph
            result = self.repo_graph.update_file(str(file_path), content)
            
            if result.needs_pagerank_recalc:
                print(f"    → Graph structure changed "
                      f"(+{result.edges_added} -{result.edges_removed} edges)")
            else:
                print(f"    → Local change only")
                
        except Exception as e:
            print(f"    ✗ Error updating {file_path}: {e}")
    
    def _handle_file_created(self, file_path: Path):
        """Handle file creation event."""
        # For now, just log
        # TODO: Implement graph.add_file()
        print(f"    → File created (graph update not yet implemented)")
    
    def _handle_file_deleted(self, file_path: Path):
        """Handle file deletion event."""
        # For now, just log
        # TODO: Implement graph.remove_file()
        print(f"    → File deleted (graph update not yet implemented)")
    
    def get_architecture_map(self, max_files: int = 50) -> str:
        """Generate a text map of the repository architecture."""
        return self.repo_graph.generate_map(max_files)
    
    def query(self, user_input: str) -> str:
        """
        Process a user query and generate a response.
        This is where LLM integration will happen.
        """
        # TODO: Implement LLM integration
        # For now, return a basic architectural response
        
        if "architecture" in user_input.lower() or "map" in user_input.lower():
            return self.get_architecture_map()
        
        return "Query processing not yet implemented"
```

**Step 3.1.2: CLI Entry Point**
```python
# Location: python_shell/atlas/cli.py

import sys
from pathlib import Path
import argparse

from .agent import AtlasAgent


def main():
    """Main CLI entry point for Atlas."""
    parser = argparse.ArgumentParser(
        description="Atlas - Sentient Repository Agent"
    )
    parser.add_argument(
        "project_root",
        type=Path,
        help="Path to the project root directory"
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Don't watch for file changes, just build and exit"
    )
    
    args = parser.parse_args()
    
    if not args.project_root.exists():
        print(f"Error: {args.project_root} does not exist")
        sys.exit(1)
    
    # Create and initialize agent
    agent = AtlasAgent(args.project_root)
    agent.initialize()
    
    if args.no_watch:
        # Just show the map and exit
        print("\n" + "="*80)
        print(agent.get_architecture_map())
        sys.exit(0)
    
    # Run the main loop
    agent.run()


if __name__ == "__main__":
    main()
```

**Step 3.1.3: Setup Package**
```python
# Location: python_shell/pyproject.toml

[project]
name = "atlas"
version = "0.1.0"
description = "Sentient Repository - Autonomous Coding Agent"
requires-python = ">=3.10"

dependencies = [
    "semantic-engine",  # The Rust extension
    "pydantic>=2.0",
    "rich>=13.0",
    "tiktoken>=0.5",
]

[project.scripts]
atlas = "atlas.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "mypy>=1.0",
]
```

**Deliverables:**
- [ ] `agent.py` with core loop implemented
- [ ] `cli.py` with argument parsing
- [ ] Package setup with entry point
- [ ] Basic manual testing shows file watching works

**Success Criteria:**
- `atlas /path/to/project` runs without crashing
- File changes are detected and logged
- Graph updates execute successfully
- Architecture map can be displayed

---

### 3.2 Add Context Management

**Problem:** Need intelligent context selection for LLM queries.

#### Implementation Steps

**Step 3.2.1: Context Manager Implementation**
```python
# Location: python_shell/atlas/context.py

from pathlib import Path
from typing import List, Dict, Tuple
import tiktoken


class ContextManager:
    """
    Manages context window for LLM queries using the
    "Anchor & Expand" strategy from the roadmap.
    """
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 100000):
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
    
    def build_context(
        self,
        anchor_files: List[Path],
        repo_graph,
        user_query: str,
        include_map: bool = True
    ) -> str:
        """
        Build context using Anchor & Expand strategy.
        
        Strategy:
        1. Start with anchor files (e.g., files mentioned in query)
        2. Add their direct dependencies
        3. Add architectural map
        4. Fill remaining space with related definitions
        """
        context_parts = []
        token_budget = self.max_tokens
        
        # Reserve space for system prompt and user query
        token_budget -= self.count_tokens(user_query)
        token_budget -= 1000  # System prompt overhead
        
        # 1. Architectural map (if requested)
        if include_map:
            map_text = repo_graph.generate_map(max_files=30)
            map_tokens = self.count_tokens(map_text)
            
            if map_tokens < token_budget * 0.2:  # Max 20% for map
                context_parts.append(("ARCHITECTURE_MAP", map_text))
                token_budget -= map_tokens
        
        # 2. Anchor files (full content)
        for anchor_file in anchor_files:
            try:
                content = anchor_file.read_text()
                tokens = self.count_tokens(content)
                
                if tokens < token_budget:
                    context_parts.append((
                        f"FILE: {anchor_file.name}",
                        content
                    ))
                    token_budget -= tokens
                else:
                    # Truncate if too large
                    truncated = self._truncate_to_tokens(content, token_budget)
                    context_parts.append((
                        f"FILE (truncated): {anchor_file.name}",
                        truncated
                    ))
                    token_budget = 0
                    break
                    
            except Exception as e:
                print(f"Warning: Could not read {anchor_file}: {e}")
        
        # 3. Direct dependencies (signatures only)
        if token_budget > 1000:
            deps = self._get_dependency_signatures(
                anchor_files,
                repo_graph,
                max_tokens=int(token_budget * 0.3)
            )
            if deps:
                context_parts.append(("DEPENDENCIES", deps))
        
        # 4. Assemble final context
        return self._format_context(context_parts)
    
    def _get_dependency_signatures(
        self,
        files: List[Path],
        repo_graph,
        max_tokens: int
    ) -> str:
        """Extract function signatures from dependencies."""
        signatures = []
        token_count = 0
        
        for file in files:
            deps = repo_graph.get_outgoing_dependencies(str(file))
            
            for dep_path, edge_kind in deps:
                try:
                    content = Path(dep_path).read_text()
                    # Extract signatures (simplified - could use tree-sitter)
                    sigs = self._extract_signatures(content)
                    
                    sig_text = f"\n# From {Path(dep_path).name}:\n{sigs}"
                    sig_tokens = self.count_tokens(sig_text)
                    
                    if token_count + sig_tokens > max_tokens:
                        break
                    
                    signatures.append(sig_text)
                    token_count += sig_tokens
                    
                except Exception:
                    continue
        
        return "\n".join(signatures)
    
    def _extract_signatures(self, content: str) -> str:
        """
        Extract function/class signatures from source code.
        This is a simplified version - could be enhanced with tree-sitter.
        """
        lines = content.split("\n")
        signatures = []
        
        for line in lines:
            stripped = line.strip()
            # Python function/class definitions
            if stripped.startswith("def ") or stripped.startswith("class "):
                # Take the line plus any docstring on next line
                signatures.append(line)
        
        return "\n".join(signatures)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens) + "\n\n[... truncated ...]"
    
    def _format_context(self, parts: List[Tuple[str, str]]) -> str:
        """Format context parts into a single string."""
        formatted = []
        for label, content in parts:
            formatted.append(f"{'='*80}\n{label}\n{'='*80}\n{content}\n")
        return "\n".join(formatted)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
```

**Deliverables:**
- [ ] `context.py` with ContextManager class
- [ ] Token counting using tiktoken
- [ ] Anchor & Expand strategy implemented
- [ ] Unit tests for context building

**Success Criteria:**
- Context always fits within token budget
- Most important files prioritized
- Graceful handling of large files
- Token counting accurate

---

### 3.3 Stub LLM Integration

**Problem:** Need basic LLM interaction for end-to-end testing.

#### Implementation Steps

**Step 3.3.1: LLM Client Interface**
```python
# Location: python_shell/atlas/llm.py

from typing import List, Dict
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages and get response."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama models."""
    
    def __init__(self, model: str = "deepseek-r1:32b-qwen-distill-q4_K_M"):
        self.model = model
        # TODO: Add ollama library when ready
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat messages to Ollama.
        
        For now, this is a stub that returns a placeholder.
        """
        # TODO: Implement actual Ollama integration
        return "[LLM Response - Not Yet Implemented]"


class StubClient(LLMClient):
    """Stub client for testing without a real LLM."""
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Return a canned response for testing."""
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        
        if "architecture" in user_msg.lower():
            return "Based on the repository map, this appears to be a Python project with a core module and several utilities."
        
        return "I'm a stub LLM client. Real integration coming soon!"
```

**Step 3.3.2: Integrate with Agent**
```python
# Location: python_shell/atlas/agent.py (additions)

from .llm import StubClient, OllamaClient

class AtlasAgent:
    def __init__(self, project_root: Path, use_real_llm: bool = False):
        # ... existing init ...
        
        # LLM client (stub by default)
        if use_real_llm:
            self.llm = OllamaClient()
        else:
            self.llm = StubClient()
    
    def query(self, user_input: str) -> str:
        """Process a user query with LLM."""
        
        # Build context
        context = self.context_manager.build_context(
            anchor_files=[],  # TODO: Extract from query
            repo_graph=self.repo_graph,
            user_query=user_input,
            include_map=True
        )
        
        # Create messages
        messages = [
            {
                "role": "system",
                "content": f"You are Atlas, an AI coding agent with deep knowledge of this repository.\n\n{context}"
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        # Get LLM response
        response = self.llm.chat(messages)
        return response
```

**Deliverables:**
- [ ] LLM client abstraction
- [ ] Stub client for testing
- [ ] Ollama client skeleton
- [ ] Integration with agent query method

**Success Criteria:**
- Query flow works end-to-end with stub
- Easy to swap in real LLM later
- Context properly formatted in system prompt

---

## Phase 4: Additional Graph Operations (Stage 5)
**Priority:** P3 - LOW  
**Can be deferred if timeline is tight**

### 4.1 Implement File Deletion Support

**Problem:** Graph has no way to remove files when they're deleted.

#### Implementation Steps

**Step 4.1.1: Add `remove_file` Method**
```rust
// Location: rust_core/src/graph.rs

impl RepoGraph {
    /// Removes a file from the graph and cleans up all associated data.
    pub fn remove_file(&mut self, file_path: &PathBuf) -> Result<UpdateResult, GraphError> {
        let node_idx = self.path_to_idx
            .get(file_path)
            .ok_or_else(|| GraphError::NodeNotFound(file_path.clone()))?
            .clone();
        
        let node = self.graph.node_weight(node_idx)
            .ok_or_else(|| GraphError::NodeNotFound(file_path.clone()))?;
        
        let mut edges_removed = 0;
        
        // 1. Clean up symbol index
        // Remove all definitions from this file
        for def in &node.definitions {
            if let Some(paths) = self.symbol_index.definitions.get_mut(def) {
                paths.retain(|p| p != file_path);
                if paths.is_empty() {
                    self.symbol_index.definitions.remove(def);
                }
            }
        }
        
        // Remove this file from usages
        self.symbol_index.usages.remove(file_path);
        
        // 2. Count and remove edges
        let incoming: Vec<_> = self.graph
            .edges_directed(node_idx, petgraph::Direction::Incoming)
            .map(|e| e.id())
            .collect();
        
        let outgoing: Vec<_> = self.graph
            .edges_directed(node_idx, petgraph::Direction::Outgoing)
            .map(|e| e.id())
            .collect();
        
        edges_removed = incoming.len() + outgoing.len();
        
        for edge_id in incoming.iter().chain(outgoing.iter()) {
            self.graph.remove_edge(*edge_id);
        }
        
        // 3. Remove the node itself
        self.graph.remove_node(node_idx);
        self.path_to_idx.remove(file_path);
        
        // 4. Mark PageRank as dirty
        self.pagerank_dirty = true;
        
        Ok(UpdateResult {
            edges_added: 0,
            edges_removed,
            needs_pagerank_recalc: true,
        })
    }
}
```

**Step 4.1.2: Add Python Binding**
```rust
// Location: rust_core/src/lib.rs

#[pymethods]
impl PyRepoGraph {
    /// Remove a file from the graph.
    fn remove_file(&mut self, file_path: &str) -> PyResult<PyGraphUpdateResult> {
        let path = PathBuf::from(file_path);
        let result = self.graph.remove_file(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to remove file: {}", e)))?;
        Ok(result.into())
    }
}
```

**Step 4.1.3: Add Tests**
```rust
#[test]
fn test_file_removal_cleanup() {
    let (_dir, mut graph) = setup_test_repo_graph();
    
    let file_to_remove = graph.import_resolver.as_ref().unwrap()
        .project_root.join("src/utils.py");
    
    let initial_node_count = graph.graph.node_count();
    let initial_edge_count = graph.graph.edge_count();
    
    // Remove the file
    let result = graph.remove_file(&file_to_remove).unwrap();
    
    // Verify node removed
    assert_eq!(graph.graph.node_count(), initial_node_count - 1);
    assert!(!graph.path_to_idx.contains_key(&file_to_remove));
    
    // Verify edges removed
    assert!(result.edges_removed > 0);
    assert!(graph.graph.edge_count() < initial_edge_count);
    
    // Verify consistency
    graph.validate_consistency().expect("Graph should remain consistent");
}
```

**Deliverables:**
- [ ] `remove_file` method in Rust
- [ ] Python binding
- [ ] Comprehensive tests
- [ ] Integration with agent deletion handler

---

### 4.2 Implement File Addition Support

**Problem:** Need cleaner API for adding new files than rebuilding entire graph.

#### Implementation Steps

**Step 4.2.1: Add `add_file` Method**
```rust
// Location: rust_core/src/graph.rs

impl RepoGraph {
    /// Adds a new file to the graph.
    pub fn add_file(
        &mut self,
        file_path: &PathBuf,
        content: &str,
    ) -> Result<UpdateResult, GraphError> {
        // Check if file already exists
        if self.path_to_idx.contains_key(file_path) {
            // If it exists, just update it
            return self.update_file(file_path, content);
        }
        
        // Parse the file
        let lang = SupportedLanguage::from_extension(
            file_path.extension().and_then(|s| s.to_str()).unwrap_or(""),
        );
        
        if lang == SupportedLanguage::Unknown {
            return Err(GraphError::ParseError(file_path.clone()));
        }
        
        let mut parser = TreeSitterParser::new();
        parser.set_language(lang.get_parser()
            .ok_or_else(|| GraphError::ParseError(file_path.clone()))?)
            .map_err(|_| GraphError::ParseError(file_path.clone()))?;
        
        let tree = parser.parse(content, None)
            .ok_or_else(|| GraphError::ParseError(file_path.clone()))?;
        
        let harvester = SymbolHarvester::new(lang);
        let (definitions, usages) = harvester.harvest(&tree, content.as_bytes());
        
        // Get imports
        let imports = self._get_file_imports(file_path, content);
        
        // Create node
        let file_node = FileNode::new(
            file_path.clone(),
            definitions.clone(),
            usages.clone(),
            &imports,
            content,
        );
        
        let node_idx = self.graph.add_node(file_node);
        self.path_to_idx.insert(file_path.clone(), node_idx);
        
        // Update symbol index
        for def in &definitions {
            self.symbol_index.definitions
                .entry(def.clone())
                .or_default()
                .push(file_path.clone());
        }
        
        if !usages.is_empty() {
            self.symbol_index.usages.insert(file_path.clone(), usages.clone());
        }
        
        // Create edges
        let mut edges_added = 0;
        
        // Import edges
        for import_path in &imports {
            if let Some(&target_idx) = self.path_to_idx.get(import_path) {
                self.graph.add_edge(node_idx, target_idx, EdgeKind::Import);
                edges_added += 1;
            }
        }
        
        // Symbol usage edges (outgoing)
        for symbol in &usages {
            if let Some(def_paths) = self.symbol_index.definitions.get(symbol) {
                for def_path in def_paths {
                    if def_path == file_path { continue; }
                    if let Some(&target_idx) = self.path_to_idx.get(def_path) {
                        self.graph.add_edge(node_idx, target_idx, EdgeKind::SymbolUsage);
                        edges_added += 1;
                    }
                }
            }
        }
        
        // Symbol usage edges (incoming - from files that use our definitions)
        for def in &definitions {
            if let Some(users) = self.symbol_index.usages.iter()
                .filter(|(_, syms)| syms.contains(def))
                .map(|(path, _)| path)
                .collect::<Vec<_>>()
                .into_iter()
                .collect::<Vec<_>>()
            {
                for user_path in users {
                    if user_path == file_path { continue; }
                    if let Some(&user_idx) = self.path_to_idx.get(user_path) {
                        self.graph.add_edge(user_idx, node_idx, EdgeKind::SymbolUsage);
                        edges_added += 1;
                    }
                }
            }
        }
        
        self.pagerank_dirty = true;
        
        Ok(UpdateResult {
            edges_added,
            edges_removed: 0,
            needs_pagerank_recalc: true,
        })
    }
}
```

**Deliverables:**
- [ ] `add_file` method
- [ ] Python binding
- [ ] Tests for file addition
- [ ] Integration with agent

---

## Phase 5: Observability & Polish (Stage 6)
**Priority:** P2-P3  
**Essential for production use**

### 5.1 Add Logging Infrastructure

**Problem:** No visibility into what the engine is doing, making debugging hard.

#### Implementation Steps

**Step 5.1.1: Rust Logging**
```rust
// Add to Cargo.toml
[dependencies]
log = "0.4"
env_logger = "0.11"

// In rust_core/src/lib.rs
use log::{info, debug, warn};

// In critical functions:
impl RepoGraph {
    pub fn update_file(&mut self, file_path: &PathBuf, content: &str) -> Result<UpdateResult, GraphError> {
        debug!("Updating file: {}", file_path.display());
        
        let tier = self.classify_change(file_path, content)?;
        info!("Change classification for {}: {:?}", file_path.display(), tier);
        
        // ... rest of logic ...
        
        info!("Update complete: {:?}", result);
        Ok(result)
    }
}
```

**Step 5.1.2: Python Logging**
```python
# Location: python_shell/atlas/agent.py

import logging

logger = logging.getLogger("atlas")

class AtlasAgent:
    def __init__(self, project_root: Path, log_level: str = "INFO"):
        # ... existing init ...
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info(f"Initializing agent for {project_root}")
    
    def _handle_file_modified(self, file_path: Path):
        logger.debug(f"Processing modification: {file_path}")
        try:
            content = file_path.read_text()
            result = self.repo_graph.update_file(str(file_path), content)
            
            logger.info(
                f"Updated {file_path.name}: "
                f"+{result.edges_added} -{result.edges_removed} edges"
            )
        except Exception as e:
            logger.error(f"Failed to update {file_path}: {e}", exc_info=True)
```

**Deliverables:**
- [ ] Rust logging with `log` crate
- [ ] Python logging configured
- [ ] Critical operations logged at INFO
- [ ] Debug information at DEBUG level

---

### 5.2 Add Progress Indicators

**Problem:** Long operations (initial scan) have no feedback.

#### Implementation Steps

**Step 5.2.1: Rich Progress Bars**
```python
# Location: python_shell/atlas/agent.py

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console

console = Console()

class AtlasAgent:
    def initialize(self):
        """Initialize with beautiful progress indicators."""
        console.print(f"[bold cyan]Initializing Atlas for {self.project_root}[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            # Task 1: Scanning
            task1 = progress.add_task("[cyan]Scanning repository...", total=None)
            files = scan_repository(str(self.project_root))
            progress.update(task1, completed=True, total=1)
            console.print(f"  ✓ Found {len(files)} files")
            
            # Task 2: Building
            task2 = progress.add_task(
                "[yellow]Building dependency graph...",
                total=len(files)
            )
            
            # This would require Rust to call back to Python with progress
            # For now, show indeterminate progress
            self.repo_graph.build_complete(files, str(self.project_root))
            progress.update(task2, completed=len(files))
            
            # Task 3: PageRank
            task3 = progress.add_task("[magenta]Calculating PageRank...", total=None)
            self.repo_graph.ensure_pagerank_up_to_date()
            progress.update(task3, completed=True, total=1)
        
        # Show results
        top_files = self.repo_graph.get_top_ranked_files(5)
        console.print("\n[bold green]✓ Repository indexed![/bold green] Top architectural files:")
        
        for i, (path, rank) in enumerate(top_files, 1):
            console.print(f"  {i}. [cyan]{Path(path).name}[/cyan] (rank: {rank:.3f})")
```

**Deliverables:**
- [ ] Rich progress bars for initialization
- [ ] Status indicators for file events
- [ ] Colorful, beautiful terminal output
- [ ] Estimated time remaining for long operations

---

### 5.3 Error Handling & Recovery

**Problem:** Need graceful degradation when things go wrong.

#### Implementation Steps

**Step 5.3.1: Graph Health Checks**
```python
# Location: python_shell/atlas/health.py

from dataclasses import dataclass
from typing import List
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    status: HealthStatus
    issues: List[str]
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

class GraphHealthChecker:
    def check_health(self, repo_graph) -> HealthCheck:
        """Check graph health and return issues."""
        issues = []
        
        stats = repo_graph.get_statistics()
        
        # Check 1: Graph not empty
        if stats.node_count == 0:
            issues.append("Graph is empty - no files indexed")
            return HealthCheck(HealthStatus.CRITICAL, issues)
        
        # Check 2: Reasonable edge/node ratio
        if stats.node_count > 10 and stats.edge_count == 0:
            issues.append("No edges in graph - imports may not be resolving")
        
        # Check 3: Symbol edges exist
        if stats.symbol_edges == 0 and stats.node_count > 5:
            issues.append("No symbol edges - semantic analysis may have failed")
        
        # Check 4: Too many orphans
        # Would need to implement in Rust to get orphan count
        
        # Determine status
        if not issues:
            status = HealthStatus.HEALTHY
        elif len(issues) <= 2:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL
        
        return HealthCheck(status, issues)
```

**Step 5.3.2: Auto-Recovery**
```python
# Location: python_shell/atlas/agent.py

class AtlasAgent:
    def _handle_file_modified(self, file_path: Path):
        """Handle file modification with error recovery."""
        try:
            content = file_path.read_text()
            result = self.repo_graph.update_file(str(file_path), content)
            
            # Success path
            if result.needs_pagerank_recalc:
                logger.info(
                    f"Updated {file_path.name}: "
                    f"+{result.edges_added} -{result.edges_removed} edges"
                )
        
        except GraphError as e:
            # Graph-specific error - try to recover
            logger.error(f"Graph error updating {file_path}: {e}")
            
            if "ParseError" in str(e):
                # Syntax error in file - non-fatal
                logger.info(f"Skipping {file_path} due to parse error")
            else:
                # Unknown graph error - check health
                self._check_health_and_recover()
        
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error updating {file_path}: {e}", exc_info=True)
            self._check_health_and_recover()
    
    def _check_health_and_recover(self):
        """Check graph health and attempt recovery if needed."""
        health = GraphHealthChecker().check_health(self.repo_graph)
        
        if health.status == HealthStatus.CRITICAL:
            logger.error("Graph in critical state!")
            for issue in health.issues:
                logger.error(f"  - {issue}")
            
            # Offer to rebuild
            console.print("[bold red]Graph corrupted. Rebuilding...[/bold red]")
            self.initialize()  # Full rebuild
        
        elif health.status == HealthStatus.DEGRADED:
            logger.warning("Graph health degraded:")
            for issue in health.issues:
                logger.warning(f"  - {issue}")
```

**Deliverables:**
- [ ] Health check system
- [ ] Auto-recovery on errors
- [ ] Graceful handling of parse errors
- [ ] Clear error messages for users

---

## Testing & Validation Strategy

### Acceptance Criteria for Each Phase

**Phase 1 (Testing) - DONE WHEN:**
- [ ] All 6+ SymbolUsage tests pass
- [ ] `validate_consistency()` integrated
- [ ] Zero test failures in CI
- [ ] Code coverage >= 80% for graph updates

**Phase 2 (Performance) - DONE WHEN:**
- [ ] `classify_change` makes zero disk reads (measured)
- [ ] `update_file` accepts content parameter
- [ ] All benchmarks show >= 5x improvement
- [ ] No performance regressions in any test

**Phase 3 (Python) - DONE WHEN:**
- [ ] `atlas /project` runs successfully
- [ ] File changes detected and processed
- [ ] Architecture map displays correctly
- [ ] Query interface returns responses (even if stub)

**Phase 4 (Operations) - DONE WHEN:**
- [ ] Files can be added/removed dynamically
- [ ] Graph remains consistent after operations
- [ ] Integration tests cover full lifecycle

**Phase 5 (Polish) - DONE WHEN:**
- [ ] All operations logged appropriately
- [ ] Progress indicators work smoothly
- [ ] Error recovery tested and validated
- [ ] User experience is smooth and professional

---

## Risk Mitigation

### Technical Risks

**Risk 1: SymbolUsage edge logic is more complex than anticipated**
- **Mitigation:** Comprehensive tests will reveal issues early
- **Contingency:** Can fall back to import-only edges temporarily

**Risk 2: Performance improvements don't materialize**
- **Mitigation:** Profile before and after each optimization
- **Contingency:** Defer Arc<String> work if not needed

**Risk 3: Python/Rust boundary introduces new bugs**
- **Mitigation:** Extensive integration tests
- **Contingency:** Add Python-side validation layer

### Schedule Risks

**Risk 1: Testing takes longer than 1 Stage**
- **Mitigation:** Parallelize test writing with design review
- **Contingency:** Defer Phase 4/5 if needed

**Risk 2: Context management is more complex than expected**
- **Mitigation:** Start with simple Anchor strategy first
- **Contingency:** Use fixed-size context windows initially

---

## Success Metrics

### Quantitative Goals

- **Test Coverage:** >= 80% for critical paths
- **Performance:** `update_file` < 10ms on 500-node graph
- **Reliability:** Zero graph corruptions in 1000 update operations
- **Responsiveness:** Initial scan < 10s for 10k files

### Qualitative Goals

- **Developer Experience:** Clear error messages, helpful logs
- **Code Quality:** All code passes clippy + mypy
- **Documentation:** Every public API documented
- **Maintainability:** New contributors can understand architecture

---

## Appendix A: Quick Reference Commands

### Running Tests
```bash
# Rust unit tests
cd rust_core
cargo test

# Specific test
cargo test test_symbol_definition_removal

# With logging
RUST_LOG=debug cargo test test_update_file_add_import -- --nocapture

# Python tests
cd python_shell
pytest
```

### Development Workflow
```bash
# Build Rust extension
cd rust_core
maturin develop

# Run agent in debug mode
cd python_shell
RUST_LOG=debug python -m atlas.cli /path/to/project --no-watch
```

### Benchmarking
```bash
cd rust_core
cargo bench
cargo flamegraph --bench incremental_benchmark
```

---

## Appendix B: Implementation Checklist

Copy this into your project tracker:

### Stage 1: Critical Testing
- [ ] Add test_symbol_definition_removal
- [ ] Add test_symbol_definition_addition
- [ ] Add test_symbol_usage_change
- [ ] Add test_symbol_collision_handling
- [ ] Add test_file_creation_with_symbols
- [ ] Add test_file_deletion_cleanup stub
- [ ] Implement validate_consistency()
- [ ] Add consistency checks to all update tests
- [ ] Run full test suite and verify 100% pass rate

### Stage 2: Performance
- [ ] Extend FileNode with hash fields
- [ ] Implement hash calculation methods
- [ ] Refactor classify_change to use hashes
- [ ] Update build() to compute hashes
- [ ] Update update_file signature (Rust)
- [ ] Update update_file signature (Python bindings)
- [ ] Update all test calls
- [ ] Benchmark before/after and document improvements

### Stage 3: Python Agent Core
- [ ] Implement AtlasAgent class
- [ ] Implement file event handlers
- [ ] Create CLI entry point
- [ ] Set up package with pyproject.toml
- [ ] Manual test: watch mode works
- [ ] Manual test: graph updates work

### Stage 4: Context & LLM Stub
- [ ] Implement ContextManager
- [ ] Implement token counting
- [ ] Implement Anchor & Expand strategy
- [ ] Create LLM client abstraction
- [ ] Implement StubClient
- [ ] Integrate with agent query method
- [ ] End-to-end test with stub LLM

### Stage 5: Additional Operations
- [ ] Implement remove_file() in Rust
- [ ] Add Python binding for remove_file
- [ ] Write tests for file removal
- [ ] Implement add_file() in Rust
- [ ] Add Python binding for add_file
- [ ] Write tests for file addition
- [ ] Update agent to use new methods

### Stage 6: Polish
- [ ] Add Rust logging with log crate
- [ ] Add Python logging configuration
- [ ] Implement Rich progress bars
- [ ] Create health check system
- [ ] Implement auto-recovery logic
- [ ] Test error scenarios
- [ ] Final end-to-end validation

---

## Conclusion

This implementation plan provides a clear, phased approach to closing all gaps identified in the codebase review. By prioritizing critical testing infrastructure first, we ensure quality while building out remaining functionality.

The plan is realistic, measurable, and aligned with Project Atlas's roadmap goals. Each phase has clear deliverables and success criteria, making progress easy to track.

**Estimated Total Effort:** 4-6 Stages (1 developer)  
**Critical Path:** Phase 1 → Phase 2 → Phase 3  
**Can Be Parallelized:** Phases 4 & 5  

Start with Phase 1 immediately - the testing gaps represent the highest risk to project success.