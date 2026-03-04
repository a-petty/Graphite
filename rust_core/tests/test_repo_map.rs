// rust_core/tests/test_repo_map.rs
//
// Tests map generation. Without tree-sitter, graphs have nodes but no edges.
// Map generation still works — files appear in directory structure with uniform ranks.

use semantic_engine::graph::RepoGraph;
use std::fs;
use tempfile::tempdir;

fn create_test_file(root: &std::path::Path, path: &str, content: &str) {
    let file_path = root.join(path);
    fs::create_dir_all(file_path.parent().unwrap()).unwrap();
    fs::write(file_path, content).unwrap();
}

fn create_test_repo_with_structure() -> (tempfile::TempDir, RepoGraph) {
    let root = tempdir().unwrap();
    let root_path = root.path().canonicalize().unwrap();

    create_test_file(&root_path, "src/core.py", "class Database:\n    pass\n");
    create_test_file(&root_path, "src/models/user.py", "class User:\n    pass\n");
    create_test_file(&root_path, "src/models/product.py", "class Product:\n    pass\n");
    create_test_file(&root_path, "src/api/routes.py", "# routes\n");
    create_test_file(&root_path, "tests/test_api.py", "def test_user():\n    pass\n");

    let mut graph = RepoGraph::new(&root_path, "python", &[], None);
    let paths = vec![
        root_path.join("src/core.py"),
        root_path.join("src/models/user.py"),
        root_path.join("src/models/product.py"),
        root_path.join("src/api/routes.py"),
        root_path.join("tests/test_api.py"),
    ];

    graph.build_complete(&paths, &root_path);

    (root, graph)
}

#[test]
fn test_generate_map_basic() {
    let (_root, mut graph) = create_test_repo_with_structure();

    let map = graph.generate_map(5);

    println!("\n=== GENERATED MAP ===\n{}\n=====================\n", map);

    assert!(map.contains("Repository Map (5 files)"));
    assert!(map.contains("TOP RANKED FILES"));
    assert!(map.contains("DIRECTORY STRUCTURE:"));
}

#[test]
fn test_generate_map_includes_ranks() {
    let (_root, mut graph) = create_test_repo_with_structure();

    let map = graph.generate_map(5);

    assert!(map.contains("[rank:"));
    assert!(map.contains("imported by:"));
}

#[test]
fn test_generate_map_includes_stars() {
    let (_root, mut graph) = create_test_repo_with_structure();

    let map = graph.generate_map(5);

    assert!(map.contains("★"));
}

#[test]
fn test_generate_map_respects_max_files() {
    let (_root, mut graph) = create_test_repo_with_structure();

    let map_3 = graph.generate_map(3);

    // Top ranked section should list at most 3 files
    let top_section = map_3.split("DIRECTORY STRUCTURE:").next().unwrap();
    let rank_count = top_section.matches("[rank:").count();
    assert!(rank_count <= 3, "Should list at most 3 top files, got {}", rank_count);
}

#[test]
fn test_generate_map_performance() {
    let (_root, mut graph) = create_test_repo_with_structure();

    use std::time::Instant;

    let start = Instant::now();
    let _map = graph.generate_map(10);
    let elapsed = start.elapsed();

    println!("Map generation took: {:?}", elapsed);
    assert!(elapsed.as_millis() < 50, "Map generation took too long: {:?}", elapsed);
}
