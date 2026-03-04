// rust_core/tests/test_graph_construction.rs
//
// Tests graph construction without tree-sitter.
// With parsing removed, build_complete creates nodes but no edges.
// Edge creation will return in Phase 1 (LLM extraction pipeline).

use semantic_engine::graph::RepoGraph;
use std::fs;
use tempfile::tempdir;

fn create_test_file(root: &std::path::Path, path: &str, content: &str) {
    let file_path = root.join(path);
    fs::create_dir_all(file_path.parent().unwrap()).unwrap();
    fs::write(file_path, content).unwrap();
}

#[test]
fn test_node_creation() {
    let root = tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();

    create_test_file(&root_path, "utils.py", "def helper():\n    pass\n");
    create_test_file(&root_path, "main.py", "import utils\nutils.helper()\n");

    let mut graph = RepoGraph::new(&root_path, "python", &[], None);
    let paths = vec![
        root_path.join("utils.py"),
        root_path.join("main.py"),
    ];

    graph.build_complete(&paths, &root_path);

    assert_eq!(graph.graph.node_count(), 2, "Should have 2 nodes");
    assert!(graph.path_to_idx.contains_key(&root_path.join("utils.py")));
    assert!(graph.path_to_idx.contains_key(&root_path.join("main.py")));
}

#[test]
fn test_no_self_edges() {
    let root = tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();

    create_test_file(&root_path, "self_import.py", "class MyClass:\n    pass\nobj = MyClass()\n");

    let mut graph = RepoGraph::new(&root_path, "python", &[], None);
    let paths = vec![root_path.join("self_import.py")];

    graph.build_complete(&paths, &root_path);

    assert_eq!(graph.graph.node_count(), 1);
    assert_eq!(graph.graph.edge_count(), 0, "Should have no edges");
}

#[test]
fn test_graph_statistics() {
    let root = tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();

    create_test_file(&root_path, "core.py", "class Database:\n    pass\n");
    create_test_file(&root_path, "models.py", "from core import Database\n");
    create_test_file(&root_path, "api.py", "from models import User\n");

    let mut graph = RepoGraph::new(&root_path, "python", &[], None);
    let paths = vec![
        root_path.join("core.py"),
        root_path.join("models.py"),
        root_path.join("api.py"),
    ];

    graph.build_complete(&paths, &root_path);

    assert_eq!(graph.graph.node_count(), 3);
    // No edges without tree-sitter parsing
    assert_eq!(graph.graph.edge_count(), 0);
}

#[test]
fn test_unsupported_language_skipped() {
    let root = tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();

    create_test_file(&root_path, "readme.md", "# Readme");
    create_test_file(&root_path, "main.py", "print('hello')");

    let mut graph = RepoGraph::new(&root_path, "python", &[], None);
    let paths = vec![
        root_path.join("readme.md"),
        root_path.join("main.py"),
    ];

    graph.build_complete(&paths, &root_path);

    // .md files are unsupported, only .py should be in graph
    assert_eq!(graph.graph.node_count(), 1);
}

#[test]
fn test_consistency_after_build() {
    let root = tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();

    for i in 0..10 {
        create_test_file(&root_path, &format!("module_{}.py", i), &format!("def func_{}():\n    pass", i));
    }

    let mut graph = RepoGraph::new(&root_path, "python", &[], None);
    let paths: Vec<_> = (0..10).map(|i| root_path.join(format!("module_{}.py", i))).collect();

    graph.build_complete(&paths, &root_path);
    graph.validate_consistency().expect("Graph should be consistent after build");
}
