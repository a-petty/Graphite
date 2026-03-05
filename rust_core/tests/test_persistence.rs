use std::path::Path;

use semantic_engine::chunk::{Chunk, ChunkType, MemoryCategory};
use semantic_engine::cooccurrence::CoOccurrenceEdge;
use semantic_engine::entity::{EntityNode, EntityType};
use semantic_engine::knowledge_graph::KnowledgeGraph;
use semantic_engine::persistence::GraphStore;

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
    assert!(root.join(".cortex/graph.msgpack").exists());
    assert!(!root.join(".cortex/graph.msgpack.bak").exists());

    // Second save with backup — old primary becomes .bak
    store.save_with_backup(&graph).unwrap();
    assert!(root.join(".cortex/graph.msgpack").exists());
    assert!(root.join(".cortex/graph.msgpack.bak").exists());
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
    std::fs::write(root.join(".cortex/graph.msgpack"), b"corrupted data").unwrap();

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
