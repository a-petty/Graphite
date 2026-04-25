use std::path::Path;

use semantic_engine::chunk::{Chunk, ChunkType, MemoryCategory};
use semantic_engine::cooccurrence::CoOccurrenceEdge;
use semantic_engine::entity::{EntityNode, EntityType};
use semantic_engine::knowledge_graph::KnowledgeGraph;
use semantic_engine::persistence::GraphStore;

/// Build a KnowledgeGraph with zero entities (empty graph).
fn make_empty_graph(root: &Path) -> KnowledgeGraph {
    KnowledgeGraph::new(root)
}

fn make_test_graph(root: &Path) -> KnowledgeGraph {
    let mut kg = KnowledgeGraph::new(root);

    let mut alice = EntityNode::new("Alice".to_string(), EntityType::Person);
    alice.aliases.push("A".to_string());
    alice.embedding = Some(vec![0.1, 0.2, 0.3]);
    let alice_id = alice.id.clone();

    let bob = EntityNode::new("Bob".to_string(), EntityType::Person);
    let bob_id = bob.id.clone();

    kg.add_entity(alice);
    kg.add_entity(bob);

    let edge = CoOccurrenceEdge::new(
        "chunk-1".to_string(),
        ChunkType::Decision,
        MemoryCategory::Episodic,
        Some(1000),
        "standup.md".to_string(),
    );
    kg.add_cooccurrence(&alice_id, &bob_id, edge).unwrap();

    let mut chunk = Chunk::new(
        "standup.md".to_string(),
        ChunkType::Decision,
        MemoryCategory::Episodic,
        "Alice and Bob decided to use Rust.".to_string(),
    );
    chunk.tags = vec![alice_id.clone(), bob_id.clone()];
    chunk.timestamp = Some(1000);
    kg.store_chunk(chunk);

    kg
}

#[test]
fn test_save_and_load_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let original = make_test_graph(root);
    let original_stats = original.get_statistics();

    let store = GraphStore::new(root);
    store.save(&original).unwrap();

    let loaded = store.load(root).unwrap();
    let loaded_stats = loaded.get_statistics();

    assert_eq!(original_stats.entity_count, loaded_stats.entity_count);
    assert_eq!(original_stats.edge_count, loaded_stats.edge_count);
    assert_eq!(original_stats.chunk_count, loaded_stats.chunk_count);

    // Verify entities are searchable after reload
    let results = loaded.search_entities("Alice", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].canonical_name, "Alice");

    // Verify alias search works after reload
    let results = loaded.search_entities("A", 10);
    assert!(!results.is_empty());

    // Verify embedding survived round-trip
    let alice = results.iter().find(|e| e.canonical_name == "Alice").unwrap();
    assert!(alice.embedding.is_some());
    assert_eq!(alice.embedding.as_ref().unwrap().len(), 3);
}

#[test]
fn test_save_with_backup() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let graph = make_test_graph(root);
    let store = GraphStore::new(root);

    // First save — no backup yet
    store.save(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack").exists());
    assert!(!root.join(".graphite/graph.msgpack.bak").exists());

    // Second save with backup — old primary becomes .bak
    store.save_with_backup(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack").exists());
    assert!(root.join(".graphite/graph.msgpack.bak").exists());
}

#[test]
fn test_recover_from_backup() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let graph = make_test_graph(root);
    let store = GraphStore::new(root);

    // Save normally, then save with backup
    store.save(&graph).unwrap();
    store.save_with_backup(&graph).unwrap();

    // Corrupt primary
    std::fs::write(root.join(".graphite/graph.msgpack"), b"corrupted data").unwrap();

    // Load should fail, but recover should succeed from backup
    assert!(store.load(root).is_err());

    let recovered = store.recover(root).unwrap();
    assert_eq!(recovered.entity_count(), 2);
}

#[test]
fn test_export_json() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let graph = make_test_graph(root);
    let store = GraphStore::new(root);

    let json = store.export_json(&graph).unwrap();

    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed["entities"].is_array());
    assert!(parsed["edges"].is_array());
    assert!(parsed["chunks"].is_array());
    assert_eq!(parsed["entities"].as_array().unwrap().len(), 2);
}

#[test]
fn test_persistence_with_document_hashes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let mut original = make_test_graph(root);
    original.set_document_hash("standup.md".to_string(), "sha256_abc123".to_string());
    original.set_document_hash("retro.md".to_string(), "sha256_def456".to_string());

    let store = GraphStore::new(root);
    store.save(&original).unwrap();

    let loaded = store.load(root).unwrap();

    // Hashes should survive round-trip
    assert_eq!(loaded.get_document_hash("standup.md"), Some("sha256_abc123"));
    assert_eq!(loaded.get_document_hash("retro.md"), Some("sha256_def456"));
    assert_eq!(loaded.tracked_documents().len(), 2);
    assert!(loaded.get_document_hash("nonexistent.md").is_none());
}

// ── Phase 1 Safety Tests ──

#[test]
fn test_save_refuses_empty_over_nonempty_file() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // First, save a populated graph
    let populated = make_test_graph(root);
    let store = GraphStore::new(root);
    store.save(&populated).unwrap();

    // Verify primary file exists and is non-trivial
    let primary = root.join(".graphite/graph.msgpack");
    assert!(primary.exists());
    let metadata = std::fs::metadata(&primary).unwrap();
    assert!(metadata.len() > 10, "populated graph file should be >10 bytes");

    // Now try to save an empty graph — should be refused
    let empty = make_empty_graph(root);
    let result = store.save(&empty);
    assert!(
        result.is_err(),
        "save() should refuse to overwrite non-empty file with empty graph"
    );
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Refusing to save empty graph"),
        "error message should explain the refusal, got: {}",
        err_msg
    );

    // Original file should still be intact
    let loaded = store.load(root).unwrap();
    assert_eq!(loaded.entity_count(), 2, "original data should survive the refused save");
}

#[test]
fn test_save_with_backup_refuses_empty_over_nonempty() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // Save a populated graph first
    let populated = make_test_graph(root);
    let store = GraphStore::new(root);
    store.save_with_backup(&populated).unwrap();

    // Try save_with_backup with an empty graph — should be refused
    let empty = make_empty_graph(root);
    let result = store.save_with_backup(&empty);
    assert!(
        result.is_err(),
        "save_with_backup() should refuse to overwrite non-empty file with empty graph"
    );
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Refusing to save empty graph"),
        "error message should explain the refusal, got: {}",
        err_msg
    );

    // Primary should still be intact
    let loaded = store.load(root).unwrap();
    assert_eq!(loaded.entity_count(), 2);
}

#[test]
fn test_timestamped_backup_rotation() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let graph = make_test_graph(root);
    let store = GraphStore::new(root);

    // Perform 3 saves with backup — should produce .bak, .bak.1, .bak.2
    // (after each save_with_backup, the old primary becomes .bak, then rotates)
    store.save_with_backup(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack").exists());
    // After first save_with_backup: no prior file, so no .bak yet
    // (the initial save() inside save_with_backup creates the primary;
    //  but the first save_with_backup sees no existing file so no .bak)

    // Second save_with_backup: moves primary→.bak, saves new primary
    store.save_with_backup(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack").exists());
    assert!(root.join(".graphite/graph.msgpack.bak").exists());

    // Third save_with_backup: rotates .bak→.bak.1, moves primary→.bak
    store.save_with_backup(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack").exists());
    assert!(root.join(".graphite/graph.msgpack.bak").exists());
    assert!(root.join(".graphite/graph.msgpack.bak.1").exists());

    // Fourth save_with_backup: rotates further → .bak.2
    store.save_with_backup(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack").exists());
    assert!(root.join(".graphite/graph.msgpack.bak").exists());
    assert!(root.join(".graphite/graph.msgpack.bak.1").exists());
    assert!(root.join(".graphite/graph.msgpack.bak.2").exists());

    // Fifth save_with_backup: should create .bak.3 and still have all
    store.save_with_backup(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack.bak.3").exists());
}

#[test]
fn test_atomic_write_leaves_no_tmpfile() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let graph = make_test_graph(root);
    let store = GraphStore::new(root);

    store.save(&graph).unwrap();
    assert!(
        !root.join(".graphite/graph.msgpack.tmp").exists(),
        "successful save must not leave a tmpfile behind"
    );

    store.save_with_backup(&graph).unwrap();
    assert!(
        !root.join(".graphite/graph.msgpack.tmp").exists(),
        "successful save_with_backup must not leave a tmpfile behind"
    );
}

#[test]
fn test_stale_tmpfile_does_not_prevent_load() {
    // Simulates a crash during a prior save: the primary is still the old,
    // good content; a partially-written tmpfile is stranded. load() must
    // ignore the tmpfile entirely and return the prior good state.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = GraphStore::new(root);

    let graph = make_test_graph(root);
    store.save(&graph).unwrap();

    // Plant a junk tmpfile as if a save had been killed mid-write.
    std::fs::write(
        root.join(".graphite/graph.msgpack.tmp"),
        b"partial write from a killed save",
    )
    .unwrap();

    // Primary still loads cleanly.
    let loaded = store.load(root).unwrap();
    assert_eq!(loaded.entity_count(), 2);

    // A fresh save cleans the tmpfile up.
    store.save(&graph).unwrap();
    assert!(
        !root.join(".graphite/graph.msgpack.tmp").exists(),
        "a subsequent save must clean up the stale tmpfile"
    );
}

#[test]
fn test_save_with_backup_keeps_primary_present_throughout() {
    // Regression guard for the prior implementation, which renamed the
    // primary to .bak *before* writing the new primary — leaving a window
    // in which the primary path did not exist on disk. Under the new
    // atomic-write implementation, the primary is always present once the
    // first save has landed, even across many subsequent save_with_backup
    // calls.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let store = GraphStore::new(root);

    let graph = make_test_graph(root);
    store.save(&graph).unwrap();
    assert!(root.join(".graphite/graph.msgpack").exists());

    for _ in 0..5 {
        store.save_with_backup(&graph).unwrap();
        assert!(
            root.join(".graphite/graph.msgpack").exists(),
            "primary must be present after every save_with_backup call"
        );
    }
}

#[test]
fn test_roundtrip_entity_count_matches() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    let original = make_test_graph(root);
    let original_entity_count = original.entity_count();
    assert!(original_entity_count > 0, "test graph should have entities");

    let store = GraphStore::new(root);
    store.save_with_backup(&original).unwrap();

    let loaded = store.load(root).unwrap();
    assert_eq!(
        loaded.entity_count(),
        original_entity_count,
        "entity count must match after round-trip"
    );

    // Also verify via search
    let results = loaded.search_entities("Alice", 10);
    assert_eq!(results.len(), 1);
    let results = loaded.search_entities("Bob", 10);
    assert_eq!(results.len(), 1);
}
