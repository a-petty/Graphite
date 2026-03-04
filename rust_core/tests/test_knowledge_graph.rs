use std::path::Path;

use semantic_engine::chunk::{Chunk, ChunkType, MemoryCategory};
use semantic_engine::cooccurrence::CoOccurrenceEdge;
use semantic_engine::entity::{EntityNode, EntityType};
use semantic_engine::knowledge_graph::KnowledgeGraph;

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
