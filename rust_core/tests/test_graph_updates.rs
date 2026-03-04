// rust_core/tests/test_graph_updates.rs
//
// Tests graph update operations without tree-sitter.
// With parsing removed, update_file only updates content hash.
// Full update behavior will return in Phase 1.

use semantic_engine::graph::RepoGraph;
use tempfile::tempdir;
use std::fs;

fn setup_test_repo_graph() -> (tempfile::TempDir, RepoGraph) {
    let dir = tempdir().unwrap();
    let root = dir.path();

    fs::create_dir_all(root.join("src/models")).unwrap();
    fs::write(root.join("src/main.py"), "import src.auth\nimport src.utils").unwrap();
    fs::write(root.join("src/auth.py"), "from .models import user").unwrap();
    fs::write(root.join("src/utils.py"), "# No imports").unwrap();
    fs::write(root.join("src/models/user.py"), "# Defines User model").unwrap();

    let files_to_scan = vec![
        root.join("src/main.py"),
        root.join("src/auth.py"),
        root.join("src/utils.py"),
        root.join("src/models/user.py"),
    ];

    let mut graph = RepoGraph::new(root, "python", &[], None);
    graph.build_complete(&files_to_scan, root);

    (dir, graph)
}

#[test]
fn test_update_file_no_change() {
    let (_dir, mut graph) = setup_test_repo_graph();
    let main_path = graph.project_root.join("src/main.py");

    let original_content = fs::read_to_string(&main_path).unwrap();

    let result = graph.update_file(&main_path, &original_content).unwrap();

    assert_eq!(result.edges_added, 0);
    assert_eq!(result.edges_removed, 0);
    assert!(!result.needs_pagerank_recalc);
}

#[test]
fn test_update_file_content_change() {
    let (_dir, mut graph) = setup_test_repo_graph();
    let main_path = graph.project_root.join("src/main.py");

    let new_content = "# Modified content";
    fs::write(&main_path, new_content).unwrap();

    let result = graph.update_file(&main_path, new_content).unwrap();

    // With simplified update, content hash changes but no edge changes
    assert_eq!(result.edges_added, 0);
    assert_eq!(result.edges_removed, 0);
    assert!(!result.needs_pagerank_recalc);
}

#[test]
fn test_lazy_pagerank_recalculation() {
    let (_dir, mut graph) = setup_test_repo_graph();

    // After build_complete, pagerank should be clean
    assert!(!graph.pagerank_dirty);

    // Accessing a rank-dependent method should keep the flag clean
    let _ = graph.get_top_ranked_files(1);
    assert!(!graph.pagerank_dirty);
}

#[test]
fn test_update_performance() {
    let dir = tempdir().unwrap();
    let root = &dir.path().canonicalize().unwrap();
    let mut graph = RepoGraph::new(root, "python", &[], None);

    let num_files = 500;
    let mut file_paths = Vec::new();

    for i in 0..num_files {
        let path = root.join(format!("src/file_{}.py", i));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "").unwrap();
        file_paths.push(path);
    }
    graph.build_complete(&file_paths, root);

    let file_to_update = file_paths[0].clone();
    let new_content = "# Updated content";
    fs::write(&file_to_update, &new_content).unwrap();

    let start = std::time::Instant::now();
    let _ = graph.update_file(&file_to_update, &new_content).unwrap();
    let duration = start.elapsed();

    println!("\nGraph update for one file in a {}-node graph took: {:?}", num_files, duration);
    assert!(duration < std::time::Duration::from_millis(10));
}

#[test]
fn test_file_deletion_cleanup() {
    let dir = tempdir().unwrap();
    let root = &dir.path().canonicalize().unwrap();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/module_a.py"), "def func_a():\n    pass").unwrap();
    fs::write(root.join("src/module_b.py"), "# uses module_a").unwrap();

    let files = vec![
        root.join("src/module_a.py"),
        root.join("src/module_b.py"),
    ];

    let mut graph = RepoGraph::new(root, "python", &[], None);
    graph.build_complete(&files, root);

    let module_b = root.join("src/module_b.py");
    let initial_node_count = graph.graph.node_count();
    assert_eq!(initial_node_count, 2);

    graph.remove_file(&module_b).unwrap();

    assert!(!graph.path_to_idx.contains_key(&module_b));
    assert_eq!(graph.graph.node_count(), initial_node_count - 1);
}

#[test]
fn test_update_maintains_consistency() {
    let (_dir, mut graph) = setup_test_repo_graph();

    graph.validate_consistency().expect("Initial graph should be consistent");

    let files_to_update = vec![
        graph.project_root.join("src/main.py"),
        graph.project_root.join("src/auth.py"),
        graph.project_root.join("src/utils.py"),
    ];

    for file in &files_to_update {
        let new_content = "# Modified content";
        fs::write(file, new_content).unwrap();

        graph.update_file(file, new_content).unwrap();

        graph.validate_consistency().expect(&format!(
            "Graph inconsistent after updating {}",
            file.display()
        ));
    }
}

#[test]
fn test_batch_updates_maintain_consistency() {
    let dir = tempdir().unwrap();
    let root = &dir.path().canonicalize().unwrap();

    fs::create_dir_all(root.join("src")).unwrap();
    for i in 0..20 {
        fs::write(
            root.join(format!("src/module_{}.py", i)),
            format!("def func_{}():\n    pass", i),
        )
        .unwrap();
    }

    let files: Vec<_> = (0..20)
        .map(|i| root.join(format!("src/module_{}.py", i)))
        .collect();

    let mut graph = RepoGraph::new(root, "python", &[], None);
    graph.build_complete(&files, root);

    graph.validate_consistency().expect("Initial build should be consistent");

    for i in 0..20 {
        let file = root.join(format!("src/module_{}.py", i));
        let content = format!("# Updated module {}", i);
        fs::write(&file, &content).unwrap();

        graph.update_file(&file, &content).unwrap();
    }

    graph.validate_consistency().expect("Graph should remain consistent after batch updates");
}

#[test]
fn test_add_file_to_existing_graph() {
    let dir = tempdir().unwrap();
    let root = &dir.path().canonicalize().unwrap();

    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/existing.py"), "# existing file").unwrap();

    let mut graph = RepoGraph::new(root, "python", &[], None);
    graph.build_complete(&[root.join("src/existing.py")], root);

    assert_eq!(graph.graph.node_count(), 1);

    // Add a new file
    let new_file = root.join("src/new_file.py");
    fs::write(&new_file, "# new file").unwrap();
    graph.add_file(new_file.clone(), "# new file").unwrap();

    assert_eq!(graph.graph.node_count(), 2);
    assert!(graph.path_to_idx.contains_key(&new_file));
    graph.validate_consistency().expect("Graph should be consistent after add_file");
}
