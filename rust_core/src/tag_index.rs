use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::entity::EntityType;

/// Index for fast entity lookup by name, alias, document, and type.
/// Adapted from SymbolIndex — same HashMap-based side-map pattern.
#[derive(Debug, Default)]
pub struct TagIndex {
    /// canonical_name (lowercased) → NodeIndex
    pub name_to_node: HashMap<String, NodeIndex>,
    /// alias (lowercased) → NodeIndex
    pub alias_to_node: HashMap<String, NodeIndex>,
    /// source document path → entity NodeIndexes
    pub document_to_entities: HashMap<String, Vec<NodeIndex>>,
    /// entity type → NodeIndexes
    pub type_to_entities: HashMap<EntityType, Vec<NodeIndex>>,
}

impl TagIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an entity in all lookup maps.
    pub fn add_entity(
        &mut self,
        name: &str,
        aliases: &[String],
        entity_type: &EntityType,
        documents: &[String],
        idx: NodeIndex,
    ) {
        self.name_to_node.insert(name.to_lowercase(), idx);

        for alias in aliases {
            self.alias_to_node.insert(alias.to_lowercase(), idx);
        }

        for doc in documents {
            self.document_to_entities
                .entry(doc.clone())
                .or_default()
                .push(idx);
        }

        self.type_to_entities
            .entry(entity_type.clone())
            .or_default()
            .push(idx);
    }

    /// Remove an entity from all lookup maps.
    pub fn remove_entity(
        &mut self,
        name: &str,
        aliases: &[String],
        entity_type: &EntityType,
        documents: &[String],
        idx: NodeIndex,
    ) {
        self.name_to_node.remove(&name.to_lowercase());

        for alias in aliases {
            self.alias_to_node.remove(&alias.to_lowercase());
        }

        for doc in documents {
            if let Some(entities) = self.document_to_entities.get_mut(doc) {
                entities.retain(|&i| i != idx);
                if entities.is_empty() {
                    self.document_to_entities.remove(doc);
                }
            }
        }

        if let Some(entities) = self.type_to_entities.get_mut(entity_type) {
            entities.retain(|&i| i != idx);
            if entities.is_empty() {
                self.type_to_entities.remove(entity_type);
            }
        }
    }

    /// Lookup by canonical name first, then aliases (case-insensitive).
    pub fn lookup_by_name(&self, name: &str) -> Option<NodeIndex> {
        let lower = name.to_lowercase();
        self.name_to_node
            .get(&lower)
            .or_else(|| self.alias_to_node.get(&lower))
            .copied()
    }

    /// Get all entity NodeIndexes associated with a source document.
    pub fn lookup_by_document(&self, doc: &str) -> Vec<NodeIndex> {
        self.document_to_entities
            .get(doc)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all entity NodeIndexes of a given type.
    pub fn lookup_by_type(&self, entity_type: &EntityType) -> Vec<NodeIndex> {
        self.type_to_entities
            .get(entity_type)
            .cloned()
            .unwrap_or_default()
    }

    /// Update all side-maps when petgraph swap-removes a node.
    /// The node that was at `old` is now at `new`.
    pub fn remap_node_index(&mut self, old: NodeIndex, new: NodeIndex) {
        for idx in self.name_to_node.values_mut() {
            if *idx == old {
                *idx = new;
            }
        }
        for idx in self.alias_to_node.values_mut() {
            if *idx == old {
                *idx = new;
            }
        }
        for entities in self.document_to_entities.values_mut() {
            for idx in entities.iter_mut() {
                if *idx == old {
                    *idx = new;
                }
            }
        }
        for entities in self.type_to_entities.values_mut() {
            for idx in entities.iter_mut() {
                if *idx == old {
                    *idx = new;
                }
            }
        }
    }

    /// Simple substring search on canonical names and aliases.
    /// Returns matching NodeIndexes, deduplicated, up to `limit`.
    pub fn search(&self, query: &str, limit: usize) -> Vec<NodeIndex> {
        let lower_query = query.to_lowercase();
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Search canonical names first
        for (name, &idx) in &self.name_to_node {
            if name.contains(&lower_query) && seen.insert(idx) {
                results.push(idx);
                if results.len() >= limit {
                    return results;
                }
            }
        }

        // Then aliases
        for (alias, &idx) in &self.alias_to_node {
            if alias.contains(&lower_query) && seen.insert(idx) {
                results.push(idx);
                if results.len() >= limit {
                    return results;
                }
            }
        }

        results
    }
}
