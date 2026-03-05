use std::path::Path;

use semantic_engine::chunk::{Chunk, ChunkType, MemoryCategory};
use semantic_engine::cooccurrence::CoOccurrenceEdge;
use semantic_engine::entity::{EntityNode, EntityType};
use semantic_engine::knowledge_graph::{KnowledgeGraph, DocumentRemovalResult};

fn make_graph() -> KnowledgeGraph {
    KnowledgeGraph::new(Path::new("/tmp/test"))
}

fn make_entity(name: &str, etype: EntityType) -> EntityNode {
    EntityNode::new(name.to_string(), etype)
}

fn make_edge(chunk_id: &str) -> CoOccurrenceEdge {
    CoOccurrenceEdge::new(
        chunk_id.to_string(),
        ChunkType::Decision,
        MemoryCategory::Episodic,
        Some(1000),
        "meetings/standup.md".to_string(),
    )
}

#[test]
fn test_add_and_get_entity() {
    let mut kg = make_graph();
    let entity = make_entity("John Doe", EntityType::Person);
    let id = entity.id.clone();

    kg.add_entity(entity);

    let retrieved = kg.get_entity(&id).unwrap();
    assert_eq!(retrieved.canonical_name, "John Doe");
    assert_eq!(retrieved.entity_type, EntityType::Person);
    assert_eq!(kg.entity_count(), 1);
}

#[test]
fn test_add_cooccurrence() {
    let mut kg = make_graph();

    let alice = make_entity("Alice", EntityType::Person);
    let bob = make_entity("Bob", EntityType::Person);
    let alice_id = alice.id.clone();
    let bob_id = bob.id.clone();

    kg.add_entity(alice);
    kg.add_entity(bob);

    let edge = make_edge("chunk-1");
    kg.add_cooccurrence(&alice_id, &bob_id, edge).unwrap();

    // Bidirectional — both sides should see the co-occurrence
    let alice_cooc = kg.get_cooccurrences(&alice_id);
    assert_eq!(alice_cooc.len(), 1);
    assert_eq!(alice_cooc[0].0, bob_id);

    let bob_cooc = kg.get_cooccurrences(&bob_id);
    assert_eq!(bob_cooc.len(), 1);
    assert_eq!(bob_cooc[0].0, alice_id);

    // 2 edges total (A→B and B→A)
    assert_eq!(kg.edge_count(), 2);
}

#[test]
fn test_store_and_get_chunk() {
    let mut kg = make_graph();

    let chunk = Chunk::new(
        "meetings/standup.md".to_string(),
        ChunkType::Decision,
        MemoryCategory::Episodic,
        "We decided to use Rust.".to_string(),
    );
    let chunk_id = chunk.id.clone();

    kg.store_chunk(chunk);

    let retrieved = kg.get_chunk(&chunk_id).unwrap();
    assert_eq!(retrieved.text, "We decided to use Rust.");
    assert_eq!(retrieved.chunk_type, ChunkType::Decision);
    assert_eq!(kg.chunk_count(), 1);
}

#[test]
fn test_get_chunks_for_entities() {
    let mut kg = make_graph();

    let alice = make_entity("Alice", EntityType::Person);
    let bob = make_entity("Bob", EntityType::Person);
    let alice_id = alice.id.clone();
    let bob_id = bob.id.clone();

    kg.add_entity(alice);
    kg.add_entity(bob);

    let mut chunk1 = Chunk::new(
        "doc.md".to_string(),
        ChunkType::Discussion,
        MemoryCategory::Episodic,
        "Alice and Bob discussed the project.".to_string(),
    );
    chunk1.tags = vec![alice_id.clone(), bob_id.clone()];

    let mut chunk2 = Chunk::new(
        "doc.md".to_string(),
        ChunkType::Background,
        MemoryCategory::Episodic,
        "Alice has a background in ML.".to_string(),
    );
    chunk2.tags = vec![alice_id.clone()];

    kg.store_chunk(chunk1);
    kg.store_chunk(chunk2);

    let alice_chunks = kg.get_chunks_for_entities(&[alice_id.clone()]);
    assert_eq!(alice_chunks.len(), 2);

    let bob_chunks = kg.get_chunks_for_entities(&[bob_id.clone()]);
    assert_eq!(bob_chunks.len(), 1);

    let both_chunks = kg.get_chunks_for_entities(&[alice_id, bob_id]);
    assert_eq!(both_chunks.len(), 2);
}

#[test]
fn test_get_temporal_chain() {
    let mut kg = make_graph();

    let entity = make_entity("Alice", EntityType::Person);
    let id = entity.id.clone();
    kg.add_entity(entity);

    let mut c1 = Chunk::new("d.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, "First".to_string());
    c1.tags = vec![id.clone()];
    c1.timestamp = Some(3000);

    let mut c2 = Chunk::new("d.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, "Second".to_string());
    c2.tags = vec![id.clone()];
    c2.timestamp = Some(1000);

    let mut c3 = Chunk::new("d.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, "Third".to_string());
    c3.tags = vec![id.clone()];
    c3.timestamp = Some(2000);

    kg.store_chunk(c1);
    kg.store_chunk(c2);
    kg.store_chunk(c3);

    let chain = kg.get_temporal_chain(&id);
    assert_eq!(chain.len(), 3);
    assert_eq!(chain[0].text, "Second");  // ts=1000
    assert_eq!(chain[1].text, "Third");   // ts=2000
    assert_eq!(chain[2].text, "First");   // ts=3000
}

#[test]
fn test_query_neighborhood_1hop() {
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Person);
    let c = make_entity("C", EntityType::Person);
    let a_id = a.id.clone();
    let b_id = b.id.clone();
    let c_id = c.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_entity(c);

    // A <-> B, B <-> C (1-hop from A should reach B but not C)
    kg.add_cooccurrence(&a_id, &b_id, make_edge("c1")).unwrap();
    kg.add_cooccurrence(&b_id, &c_id, make_edge("c2")).unwrap();

    let result = kg.query_neighborhood(&a_id, 1, None, None);
    let names: Vec<&str> = result.entities.iter().map(|e| e.canonical_name.as_str()).collect();
    assert!(names.contains(&"A"));
    assert!(names.contains(&"B"));
    assert!(!names.contains(&"C"));  // 2 hops away
}

#[test]
fn test_query_neighborhood_time_filtered() {
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Person);
    let c = make_entity("C", EntityType::Person);
    let a_id = a.id.clone();
    let b_id = b.id.clone();
    let c_id = c.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_entity(c);

    // Edge A→B at time 500, edge A→C at time 2000
    let mut edge_ab = make_edge("c1");
    edge_ab.timestamp = Some(500);
    kg.add_cooccurrence(&a_id, &b_id, edge_ab).unwrap();

    let mut edge_ac = make_edge("c2");
    edge_ac.timestamp = Some(2000);
    kg.add_cooccurrence(&a_id, &c_id, edge_ac).unwrap();

    // Query with time filter 0..1000 — should only reach B
    let result = kg.query_neighborhood(&a_id, 1, Some(0), Some(1000));
    let names: Vec<&str> = result.entities.iter().map(|e| e.canonical_name.as_str()).collect();
    assert!(names.contains(&"A"));
    assert!(names.contains(&"B"));
    assert!(!names.contains(&"C"));
}

#[test]
fn test_merge_entities() {
    let mut kg = make_graph();

    let mut alice = make_entity("Alice Smith", EntityType::Person);
    alice.aliases.push("Alice".to_string());
    let alice_id = alice.id.clone();

    let mut alice2 = make_entity("A. Smith", EntityType::Person);
    alice2.aliases.push("AS".to_string());
    let alice2_id = alice2.id.clone();

    let bob = make_entity("Bob", EntityType::Person);
    let bob_id = bob.id.clone();

    kg.add_entity(alice);
    kg.add_entity(alice2);
    kg.add_entity(bob);

    // Connect alice2 to bob
    kg.add_cooccurrence(&alice2_id, &bob_id, make_edge("c1")).unwrap();

    // Merge alice2 into alice
    let kept_id = kg.merge_entities(&alice_id, &alice2_id).unwrap();
    assert_eq!(kept_id, alice_id);

    // alice2 should be gone
    assert!(kg.get_entity(&alice2_id).is_none());

    // alice should have the merged aliases
    let alice = kg.get_entity(&alice_id).unwrap();
    assert!(alice.aliases.contains(&"A. Smith".to_string()));
    assert!(alice.aliases.contains(&"AS".to_string()));
    assert!(alice.aliases.contains(&"Alice".to_string()));

    // alice should now have edges to bob (redirected from alice2)
    let cooc = kg.get_cooccurrences(&alice_id);
    assert!(!cooc.is_empty());
}

#[test]
fn test_remove_entity_swap_safety() {
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Person);
    let c = make_entity("C", EntityType::Person);
    let a_id = a.id.clone();
    let b_id = b.id.clone();
    let c_id = c.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_entity(c);

    // Remove A (first node). C (last node) should swap into A's position.
    kg.remove_entity(&a_id);

    assert!(kg.get_entity(&a_id).is_none());
    assert_eq!(kg.entity_count(), 2);

    // B and C should still be findable
    assert!(kg.get_entity(&b_id).is_some());
    assert!(kg.get_entity(&c_id).is_some());

    // Search should still work
    let results = kg.search_entities("B", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].canonical_name, "B");

    let results = kg.search_entities("C", 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].canonical_name, "C");
}

#[test]
fn test_pagerank() {
    let mut kg = make_graph();

    let hub = make_entity("Hub", EntityType::Concept);
    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Person);
    let c = make_entity("C", EntityType::Person);
    let hub_id = hub.id.clone();
    let a_id = a.id.clone();
    let b_id = b.id.clone();
    let c_id = c.id.clone();

    kg.add_entity(hub);
    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_entity(c);

    // Hub connects to all — should get highest rank
    kg.add_cooccurrence(&hub_id, &a_id, make_edge("c1")).unwrap();
    kg.add_cooccurrence(&hub_id, &b_id, make_edge("c2")).unwrap();
    kg.add_cooccurrence(&hub_id, &c_id, make_edge("c3")).unwrap();

    kg.compute_pagerank();

    let hub_entity = kg.get_entity(&hub_id).unwrap();
    let a_entity = kg.get_entity(&a_id).unwrap();

    // Hub should have higher rank than a leaf node
    assert!(hub_entity.rank > a_entity.rank, "Hub rank {} should be > A rank {}", hub_entity.rank, a_entity.rank);

    // get_top_entities should return hub first
    let top = kg.get_top_entities(1);
    assert_eq!(top[0].canonical_name, "Hub");
}

#[test]
fn test_search_entities() {
    let mut kg = make_graph();

    let mut e1 = make_entity("John Doe", EntityType::Person);
    e1.aliases.push("JD".to_string());
    let mut e2 = make_entity("Jane Doe", EntityType::Person);
    e2.aliases.push("JD2".to_string());
    let e3 = make_entity("Rust", EntityType::Technology);

    kg.add_entity(e1);
    kg.add_entity(e2);
    kg.add_entity(e3);

    let results = kg.search_entities("doe", 10);
    assert_eq!(results.len(), 2);

    let results = kg.search_entities("rust", 10);
    assert_eq!(results.len(), 1);

    let results = kg.search_entities("xyz", 10);
    assert!(results.is_empty());
}

#[test]
fn test_decay_scores() {
    let mut kg = make_graph();

    let mut entity = make_entity("Test", EntityType::Concept);
    entity.access_count = 100;
    // Set updated_at to 30 days ago
    entity.updated_at = chrono::Utc::now().timestamp() - (30 * 86400);
    let id = entity.id.clone();

    kg.add_entity(entity);

    // Apply decay with 30-day half-life
    kg.decay_scores(30.0);

    let decayed = kg.get_entity(&id).unwrap();
    // After 30 days with 30-day half-life, access_count should be ~50
    assert!(decayed.access_count < 100, "access_count should have decayed from 100");
    assert!(decayed.access_count >= 40 && decayed.access_count <= 60,
        "access_count {} should be roughly 50 (half of 100)", decayed.access_count);
}

#[test]
fn test_statistics() {
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Technology);
    let a_id = a.id.clone();
    let b_id = b.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_cooccurrence(&a_id, &b_id, make_edge("c1")).unwrap();

    let chunk = Chunk::new("doc.md".to_string(), ChunkType::Decision, MemoryCategory::Episodic, "text".to_string());
    kg.store_chunk(chunk);

    let stats = kg.get_statistics();
    assert_eq!(stats.entity_count, 2);
    assert_eq!(stats.edge_count, 2); // bidirectional
    assert_eq!(stats.chunk_count, 1);
    assert_eq!(*stats.entities_by_type.get("Person").unwrap(), 1);
    assert_eq!(*stats.entities_by_type.get("Technology").unwrap(), 1);
}

#[test]
fn test_export_subgraph() {
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Person);
    let c = make_entity("C", EntityType::Person);
    let a_id = a.id.clone();
    let b_id = b.id.clone();
    let c_id = c.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_entity(c);

    kg.add_cooccurrence(&a_id, &b_id, make_edge("c1")).unwrap();
    kg.add_cooccurrence(&b_id, &c_id, make_edge("c2")).unwrap();

    // Export only A and B
    let result = kg.export_subgraph(&[a_id.clone(), b_id.clone()]);
    assert_eq!(result.entities.len(), 2);
    // Only edges between A and B should be included
    assert!(result.edges.len() >= 2); // A→B and B→A
    for (from, to, _) in &result.edges {
        assert!(from == &a_id || from == &b_id);
        assert!(to == &a_id || to == &b_id);
    }
}

#[test]
fn test_all_entity_ids() {
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Technology);
    let a_id = a.id.clone();
    let b_id = b.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);

    let ids = kg.all_entity_ids();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&a_id));
    assert!(ids.contains(&b_id));
}

#[test]
fn test_find_orphan_entities() {
    let mut kg = make_graph();

    let a = make_entity("Connected", EntityType::Person);
    let b = make_entity("Also Connected", EntityType::Person);
    let c = make_entity("Orphan", EntityType::Person);
    let a_id = a.id.clone();
    let b_id = b.id.clone();
    let c_id = c.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_entity(c);

    // Connect A and B, leave C orphaned
    kg.add_cooccurrence(&a_id, &b_id, make_edge("c1")).unwrap();

    let orphans = kg.find_orphan_entities();
    assert_eq!(orphans.len(), 1);
    assert_eq!(orphans[0], c_id);
}

#[test]
fn test_recalculate_edge_weights() {
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Person);
    let a_id = a.id.clone();
    let b_id = b.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);

    // Add two edges between A and B (simulating co-occurrence in 2 chunks)
    kg.add_cooccurrence(&a_id, &b_id, make_edge("c1")).unwrap();
    kg.add_cooccurrence(&a_id, &b_id, make_edge("c2")).unwrap();

    // Before recalculation, each edge has weight from chunk_type
    let updated = kg.recalculate_edge_weights();
    // 4 directed edges (2 per add_cooccurrence call, bidirectional)
    assert_eq!(updated, 4);

    // Each A→B edge should now have weight 2.0 (2 parallel A→B edges)
    let coocs = kg.get_cooccurrences(&a_id);
    // All edges pointing to B should have weight reflecting the count
    for (_, edge) in &coocs {
        assert!(edge.weight >= 2.0, "Weight should be >= 2.0, got {}", edge.weight);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 6: Incremental update tests
// ═══════════════════════════════════════════════════════════════════════

fn make_entity_with_doc(name: &str, etype: EntityType, doc: &str) -> EntityNode {
    let mut e = EntityNode::new(name.to_string(), etype);
    e.source_documents.push(doc.to_string());
    e
}

fn make_edge_for_doc(chunk_id: &str, doc: &str) -> CoOccurrenceEdge {
    CoOccurrenceEdge::new(
        chunk_id.to_string(),
        ChunkType::Decision,
        MemoryCategory::Episodic,
        Some(1000),
        doc.to_string(),
    )
}

#[test]
fn test_get_chunks_by_document() {
    let mut kg = make_graph();

    let c1 = Chunk::new("doc1.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, "Chunk 1".to_string());
    let c2 = Chunk::new("doc1.md".to_string(), ChunkType::Decision, MemoryCategory::Episodic, "Chunk 2".to_string());
    let c3 = Chunk::new("doc2.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, "Chunk 3".to_string());

    kg.store_chunk(c1);
    kg.store_chunk(c2);
    kg.store_chunk(c3);

    let doc1_chunks = kg.get_chunks_by_document("doc1.md");
    assert_eq!(doc1_chunks.len(), 2);

    let doc2_chunks = kg.get_chunks_by_document("doc2.md");
    assert_eq!(doc2_chunks.len(), 1);

    let no_chunks = kg.get_chunks_by_document("nonexistent.md");
    assert!(no_chunks.is_empty());
}

#[test]
fn test_remove_document_chunks_and_edges() {
    let mut kg = make_graph();

    // Two documents, each with entities, chunks, and edges
    let mut alice = make_entity_with_doc("Alice", EntityType::Person, "doc1.md");
    let mut bob = make_entity_with_doc("Bob", EntityType::Person, "doc1.md");
    let mut charlie = make_entity_with_doc("Charlie", EntityType::Person, "doc2.md");
    let alice_id = alice.id.clone();
    let bob_id = bob.id.clone();
    let charlie_id = charlie.id.clone();

    kg.add_entity(alice);
    kg.add_entity(bob);
    kg.add_entity(charlie);

    // Chunks for doc1
    let mut c1 = Chunk::new("doc1.md".to_string(), ChunkType::Decision, MemoryCategory::Episodic, "Alice and Bob.".to_string());
    let c1_id = c1.id.clone();
    c1.tags = vec![alice_id.clone(), bob_id.clone()];
    kg.store_chunk(c1);

    // Chunks for doc2
    let mut c2 = Chunk::new("doc2.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, "Charlie here.".to_string());
    c2.tags = vec![charlie_id.clone()];
    kg.store_chunk(c2);

    // Edges for doc1
    kg.add_cooccurrence(&alice_id, &bob_id, make_edge_for_doc(&c1_id, "doc1.md")).unwrap();

    // Remove doc1
    let result = kg.remove_document("doc1.md");
    assert_eq!(result.chunks_removed, 1);
    assert_eq!(result.edges_removed, 2); // bidirectional
    assert_eq!(result.entities_removed, 2); // Alice and Bob (sole-source)

    // doc2 should be intact
    assert_eq!(kg.chunk_count(), 1);
    assert!(kg.get_entity(&charlie_id).is_some());
    assert!(kg.get_entity(&alice_id).is_none());
    assert!(kg.get_entity(&bob_id).is_none());
}

#[test]
fn test_remove_document_entity_cleanup() {
    let mut kg = make_graph();

    // "Shared" entity appears in both doc1 and doc2
    let mut shared = EntityNode::new("Shared".to_string(), EntityType::Concept);
    shared.source_documents.push("doc1.md".to_string());
    shared.source_documents.push("doc2.md".to_string());
    let shared_id = shared.id.clone();

    // "Sole" entity only in doc1
    let mut sole = make_entity_with_doc("Sole", EntityType::Person, "doc1.md");
    let sole_id = sole.id.clone();

    kg.add_entity(shared);
    kg.add_entity(sole);

    let mut c1 = Chunk::new("doc1.md".to_string(), ChunkType::Decision, MemoryCategory::Episodic, "text".to_string());
    let c1_id = c1.id.clone();
    c1.tags = vec![shared_id.clone(), sole_id.clone()];
    kg.store_chunk(c1);

    kg.add_cooccurrence(&shared_id, &sole_id, make_edge_for_doc(&c1_id, "doc1.md")).unwrap();

    let result = kg.remove_document("doc1.md");
    assert_eq!(result.chunks_removed, 1);
    assert_eq!(result.entities_removed, 1); // sole removed
    assert_eq!(result.entities_updated, 1); // shared updated

    // Shared still exists but source_documents trimmed
    let shared_entity = kg.get_entity(&shared_id).unwrap();
    assert!(!shared_entity.source_documents.contains(&"doc1.md".to_string()));
    assert!(shared_entity.source_documents.contains(&"doc2.md".to_string()));

    // Sole is gone
    assert!(kg.get_entity(&sole_id).is_none());
}

#[test]
fn test_remove_document_entity_with_remaining_edges() {
    let mut kg = make_graph();

    // Entity sourced only from doc1, but has edges from doc2
    let mut entity = make_entity_with_doc("EdgeKeeper", EntityType::Person, "doc1.md");
    let entity_id = entity.id.clone();

    let other = make_entity_with_doc("Other", EntityType::Person, "doc2.md");
    let other_id = other.id.clone();

    kg.add_entity(entity);
    kg.add_entity(other);

    // Chunk from doc1
    let mut c1 = Chunk::new("doc1.md".to_string(), ChunkType::Decision, MemoryCategory::Episodic, "text".to_string());
    let c1_id = c1.id.clone();
    c1.tags = vec![entity_id.clone()];
    kg.store_chunk(c1);

    // Edge from doc2 connecting entity and other
    let mut c2 = Chunk::new("doc2.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, "text".to_string());
    let c2_id = c2.id.clone();
    c2.tags = vec![entity_id.clone(), other_id.clone()];
    kg.store_chunk(c2);
    kg.add_cooccurrence(&entity_id, &other_id, make_edge_for_doc(&c2_id, "doc2.md")).unwrap();

    let result = kg.remove_document("doc1.md");
    assert_eq!(result.chunks_removed, 1);
    assert_eq!(result.entities_removed, 0); // entity kept because of edges from doc2
    assert_eq!(result.entities_updated, 1);
    assert!(kg.get_entity(&entity_id).is_some());
}

#[test]
fn test_remove_document_empty() {
    let mut kg = make_graph();

    let result = kg.remove_document("nonexistent.md");
    assert_eq!(result.chunks_removed, 0);
    assert_eq!(result.edges_removed, 0);
    assert_eq!(result.entities_removed, 0);
    assert_eq!(result.entities_updated, 0);
}

#[test]
fn test_remove_document_swap_remove_safety() {
    let mut kg = make_graph();

    // Create 5 entities, all from the same document — removing them all
    // exercises the swap-remove path multiple times
    let mut ids = Vec::new();
    for i in 0..5 {
        let e = make_entity_with_doc(&format!("Entity{}", i), EntityType::Person, "doc1.md");
        ids.push(e.id.clone());
        kg.add_entity(e);
    }

    // Chunks tagging various entities
    for i in 0..5 {
        let mut c = Chunk::new("doc1.md".to_string(), ChunkType::Discussion, MemoryCategory::Episodic, format!("Chunk {}", i));
        c.tags = vec![ids[i].clone()];
        kg.store_chunk(c);
    }

    let result = kg.remove_document("doc1.md");
    assert_eq!(result.chunks_removed, 5);
    assert_eq!(result.entities_removed, 5);
    assert_eq!(kg.entity_count(), 0);
    assert_eq!(kg.chunk_count(), 0);

    // Graph should be in a consistent state — no dangling references
    let orphans = kg.find_orphan_entities();
    assert!(orphans.is_empty());
    let all_ids = kg.all_entity_ids();
    assert!(all_ids.is_empty());
}

#[test]
fn test_document_hash_crud() {
    let mut kg = make_graph();

    // Initially no hashes
    assert!(kg.get_document_hash("doc1.md").is_none());
    assert!(kg.tracked_documents().is_empty());

    // Set hash
    kg.set_document_hash("doc1.md".to_string(), "abc123".to_string());
    assert_eq!(kg.get_document_hash("doc1.md"), Some("abc123"));

    // Update hash
    kg.set_document_hash("doc1.md".to_string(), "def456".to_string());
    assert_eq!(kg.get_document_hash("doc1.md"), Some("def456"));

    // Add another
    kg.set_document_hash("doc2.md".to_string(), "ghi789".to_string());
    let tracked = kg.tracked_documents();
    assert_eq!(tracked.len(), 2);

    // Remove hash
    let removed = kg.remove_document_hash("doc1.md");
    assert_eq!(removed, Some("def456".to_string()));
    assert!(kg.get_document_hash("doc1.md").is_none());
    assert_eq!(kg.tracked_documents().len(), 1);

    // Remove non-existent
    let removed = kg.remove_document_hash("nonexistent.md");
    assert!(removed.is_none());
}

// ═══════════════════════════════════════════════════════════════════════
// Bug fix tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_query_neighborhood_excludes_none_timestamp() {
    // Bug #3: edges with timestamp=None should be excluded when a time filter is active
    let mut kg = make_graph();

    let a = make_entity("A", EntityType::Person);
    let b = make_entity("B", EntityType::Person);
    let c = make_entity("C", EntityType::Person);
    let a_id = a.id.clone();
    let b_id = b.id.clone();
    let c_id = c.id.clone();

    kg.add_entity(a);
    kg.add_entity(b);
    kg.add_entity(c);

    // Edge A→B with timestamp=500 (in range)
    let mut edge_ab = make_edge("c1");
    edge_ab.timestamp = Some(500);
    kg.add_cooccurrence(&a_id, &b_id, edge_ab).unwrap();

    // Edge A→C with timestamp=None (should be excluded by time filter)
    let mut edge_ac = CoOccurrenceEdge::new(
        "c2".to_string(), ChunkType::Discussion, MemoryCategory::Episodic,
        None, "doc.md".to_string(),
    );
    kg.add_cooccurrence(&a_id, &c_id, edge_ac).unwrap();

    // Query with time filter — C should NOT be reached via the None-timestamp edge
    let result = kg.query_neighborhood(&a_id, 1, Some(0), Some(1000));
    let names: Vec<&str> = result.entities.iter().map(|e| e.canonical_name.as_str()).collect();
    assert!(names.contains(&"A"));
    assert!(names.contains(&"B"));
    assert!(!names.contains(&"C"), "C should be excluded: its edge has no timestamp");

    // Without time filter, C should be reachable
    let result_no_filter = kg.query_neighborhood(&a_id, 1, None, None);
    let names: Vec<&str> = result_no_filter.entities.iter().map(|e| e.canonical_name.as_str()).collect();
    assert!(names.contains(&"C"), "C should be reachable without time filter");
}

#[test]
fn test_add_cooccurrence_rejects_self_loop() {
    // Bug #2: self-loops should be rejected
    let mut kg = make_graph();

    let entity = make_entity("Alice", EntityType::Person);
    let id = entity.id.clone();
    kg.add_entity(entity);

    let edge = make_edge("c1");
    let result = kg.add_cooccurrence(&id, &id, edge);
    assert!(result.is_err(), "Self-loop should be rejected");
    assert!(result.unwrap_err().contains("self-loop"));
    assert_eq!(kg.edge_count(), 0, "No edges should be added");
}

#[test]
fn test_store_chunk_populates_entity_source_chunks() {
    // Bug #9: store_chunk should update tagged entities' source_chunks
    let mut kg = make_graph();

    let alice = make_entity("Alice", EntityType::Person);
    let bob = make_entity("Bob", EntityType::Person);
    let alice_id = alice.id.clone();
    let bob_id = bob.id.clone();
    kg.add_entity(alice);
    kg.add_entity(bob);

    let mut chunk = Chunk::new(
        "doc.md".to_string(), ChunkType::Decision,
        MemoryCategory::Episodic, "Alice and Bob discussed.".to_string(),
    );
    let chunk_id = chunk.id.clone();
    chunk.tags = vec![alice_id.clone(), bob_id.clone()];
    kg.store_chunk(chunk);

    // Both entities should now list this chunk in source_chunks
    let alice_entity = kg.get_entity(&alice_id).unwrap();
    assert!(alice_entity.source_chunks.contains(&chunk_id),
        "Alice's source_chunks should contain the chunk ID");

    let bob_entity = kg.get_entity(&bob_id).unwrap();
    assert!(bob_entity.source_chunks.contains(&chunk_id),
        "Bob's source_chunks should contain the chunk ID");
}

#[test]
fn test_store_chunk_no_duplicate_source_chunks() {
    // Bug #9: storing the same chunk twice should not create duplicate entries
    let mut kg = make_graph();

    let entity = make_entity("Alice", EntityType::Person);
    let entity_id = entity.id.clone();
    kg.add_entity(entity);

    let mut chunk = Chunk::new(
        "doc.md".to_string(), ChunkType::Decision,
        MemoryCategory::Episodic, "Alice decided.".to_string(),
    );
    let chunk_id = chunk.id.clone();
    chunk.tags = vec![entity_id.clone()];

    // Store the same chunk twice (e.g., re-ingestion)
    kg.store_chunk(chunk.clone());
    kg.store_chunk(chunk);

    let entity = kg.get_entity(&entity_id).unwrap();
    let count = entity.source_chunks.iter().filter(|c| *c == &chunk_id).count();
    assert_eq!(count, 1, "source_chunks should not contain duplicates");
}

#[test]
fn test_merge_entities_records_history() {
    // Bug #13: merge should record audit trail
    let mut kg = make_graph();

    let alice = make_entity("Alice Smith", EntityType::Person);
    let alice2 = make_entity("A. Smith", EntityType::Person);
    let alice_id = alice.id.clone();
    let alice2_id = alice2.id.clone();
    let alice2_name = alice2.canonical_name.clone();

    kg.add_entity(alice);
    kg.add_entity(alice2);

    let kept_id = kg.merge_entities(&alice_id, &alice2_id).unwrap();

    let entity = kg.get_entity(&kept_id).unwrap();
    assert_eq!(entity.merge_history.len(), 1, "Should have one merge record");

    let record = &entity.merge_history[0];
    assert_eq!(record.merged_entity_id, alice2_id);
    assert_eq!(record.merged_entity_name, alice2_name);
    assert_eq!(record.method, "direct");
    assert!(record.merged_at > 0);
}
