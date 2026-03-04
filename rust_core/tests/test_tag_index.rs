use petgraph::graph::NodeIndex;
use semantic_engine::entity::EntityType;
use semantic_engine::tag_index::TagIndex;

#[test]
fn test_add_and_lookup_by_name() {
    let mut index = TagIndex::new();
    let idx = NodeIndex::new(0);

    index.add_entity(
        "John Doe",
        &["JD".to_string()],
        &EntityType::Person,
        &["meetings/standup.md".to_string()],
        idx,
    );

    assert_eq!(index.lookup_by_name("John Doe"), Some(idx));
    assert_eq!(index.lookup_by_name("john doe"), Some(idx)); // case-insensitive
    assert_eq!(index.lookup_by_name("Jane"), None);
}

#[test]
fn test_alias_lookup() {
    let mut index = TagIndex::new();
    let idx = NodeIndex::new(0);

    index.add_entity(
        "John Doe",
        &["JD".to_string(), "Johnny".to_string()],
        &EntityType::Person,
        &[],
        idx,
    );

    assert_eq!(index.lookup_by_name("jd"), Some(idx));
    assert_eq!(index.lookup_by_name("johnny"), Some(idx));
    assert_eq!(index.lookup_by_name("JOHNNY"), Some(idx));
}

#[test]
fn test_lookup_by_document() {
    let mut index = TagIndex::new();
    let idx0 = NodeIndex::new(0);
    let idx1 = NodeIndex::new(1);

    index.add_entity("Alice", &[], &EntityType::Person, &["doc1.md".to_string()], idx0);
    index.add_entity("Bob", &[], &EntityType::Person, &["doc1.md".to_string(), "doc2.md".to_string()], idx1);

    let doc1_entities = index.lookup_by_document("doc1.md");
    assert_eq!(doc1_entities.len(), 2);
    assert!(doc1_entities.contains(&idx0));
    assert!(doc1_entities.contains(&idx1));

    let doc2_entities = index.lookup_by_document("doc2.md");
    assert_eq!(doc2_entities.len(), 1);
    assert!(doc2_entities.contains(&idx1));

    assert!(index.lookup_by_document("doc3.md").is_empty());
}

#[test]
fn test_lookup_by_type() {
    let mut index = TagIndex::new();
    let idx0 = NodeIndex::new(0);
    let idx1 = NodeIndex::new(1);
    let idx2 = NodeIndex::new(2);

    index.add_entity("Alice", &[], &EntityType::Person, &[], idx0);
    index.add_entity("Bob", &[], &EntityType::Person, &[], idx1);
    index.add_entity("Rust", &[], &EntityType::Technology, &[], idx2);

    let people = index.lookup_by_type(&EntityType::Person);
    assert_eq!(people.len(), 2);

    let tech = index.lookup_by_type(&EntityType::Technology);
    assert_eq!(tech.len(), 1);
    assert_eq!(tech[0], idx2);

    assert!(index.lookup_by_type(&EntityType::Project).is_empty());
}

#[test]
fn test_remap_after_remove() {
    let mut index = TagIndex::new();
    let idx0 = NodeIndex::new(0);
    let idx1 = NodeIndex::new(1);
    let idx2 = NodeIndex::new(2);

    index.add_entity("Alice", &["A".to_string()], &EntityType::Person, &["doc.md".to_string()], idx0);
    index.add_entity("Bob", &[], &EntityType::Person, &[], idx1);
    index.add_entity("Charlie", &[], &EntityType::Person, &["doc.md".to_string()], idx2);

    // Simulate removing idx0 (Alice). Petgraph swap-removes: idx2 (Charlie) moves to idx0.
    index.remove_entity("Alice", &["A".to_string()], &EntityType::Person, &["doc.md".to_string()], idx0);
    index.remap_node_index(idx2, idx0);

    // Alice should be gone
    assert_eq!(index.lookup_by_name("Alice"), None);
    assert_eq!(index.lookup_by_name("A"), None);

    // Charlie should now be at idx0
    assert_eq!(index.lookup_by_name("Charlie"), Some(idx0));

    // Bob still at idx1
    assert_eq!(index.lookup_by_name("Bob"), Some(idx1));
}

#[test]
fn test_search() {
    let mut index = TagIndex::new();
    let idx0 = NodeIndex::new(0);
    let idx1 = NodeIndex::new(1);
    let idx2 = NodeIndex::new(2);

    index.add_entity("John Doe", &["JD".to_string()], &EntityType::Person, &[], idx0);
    index.add_entity("Jane Doe", &[], &EntityType::Person, &[], idx1);
    index.add_entity("Rust Language", &[], &EntityType::Technology, &[], idx2);

    let results = index.search("doe", 10);
    assert_eq!(results.len(), 2);

    let results = index.search("rust", 10);
    assert_eq!(results.len(), 1);

    let results = index.search("xyz", 10);
    assert!(results.is_empty());

    // Limit
    let results = index.search("doe", 1);
    assert_eq!(results.len(), 1);
}
