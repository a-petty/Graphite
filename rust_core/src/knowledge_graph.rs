use std::collections::HashMap;
use std::path::{Path, PathBuf};

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

use crate::chunk::Chunk;
use crate::cooccurrence::CoOccurrenceEdge;
use crate::entity::EntityNode;
use crate::tag_index::TagIndex;

/// A JSON-serializable subgraph result for LLM context assembly.
#[derive(Debug, Serialize, Deserialize)]
pub struct SubgraphResult {
    pub entities: Vec<EntityNode>,
    pub edges: Vec<(String, String, CoOccurrenceEdge)>,
    pub chunks: Vec<Chunk>,
}

/// Statistics about the knowledge graph.
#[derive(Debug, Serialize, Deserialize)]
pub struct KnowledgeGraphStatistics {
    pub entity_count: usize,
    pub edge_count: usize,
    pub chunk_count: usize,
    pub entities_by_type: HashMap<String, usize>,
    pub documents_indexed: usize,
}

/// The core knowledge graph, wrapping a petgraph DiGraph.
///
/// Stores entities as nodes and co-occurrence relationships as edges.
/// Chunks are stored in a side-map for O(1) evidence retrieval.
pub struct KnowledgeGraph {
    graph: DiGraph<EntityNode, CoOccurrenceEdge>,
    entity_index: HashMap<String, NodeIndex>,
    chunks: HashMap<String, Chunk>,
    tag_index: TagIndex,
    pagerank_dirty: bool,
    root_path: PathBuf,
}

impl KnowledgeGraph {
    // ── Constructor ──

    pub fn new(root_path: &Path) -> Self {
        Self {
            graph: DiGraph::new(),
            entity_index: HashMap::new(),
            chunks: HashMap::new(),
            tag_index: TagIndex::new(),
            pagerank_dirty: true,
            root_path: root_path.to_path_buf(),
        }
    }

    pub fn root_path(&self) -> &Path {
        &self.root_path
    }

    // ── Entity CRUD ──

    /// Add an entity to the graph. Returns its NodeIndex.
    pub fn add_entity(&mut self, node: EntityNode) -> NodeIndex {
        let id = node.id.clone();
        let idx = self.graph.add_node(node.clone());
        self.entity_index.insert(id, idx);
        self.tag_index.add_entity(
            &node.canonical_name,
            &node.aliases,
            &node.entity_type,
            &node.source_documents,
            idx,
        );
        self.pagerank_dirty = true;
        idx
    }

    /// Get an entity by its UUID.
    pub fn get_entity(&self, id: &str) -> Option<&EntityNode> {
        self.entity_index
            .get(id)
            .and_then(|&idx| self.graph.node_weight(idx))
    }

    /// Get a mutable reference to an entity by its UUID.
    pub fn get_entity_mut(&mut self, id: &str) -> Option<&mut EntityNode> {
        self.entity_index
            .get(id)
            .copied()
            .and_then(move |idx| self.graph.node_weight_mut(idx))
    }

    /// Remove an entity from the graph. Handles petgraph swap-remove:
    /// when a node is removed, the last node takes its index.
    pub fn remove_entity(&mut self, id: &str) -> Option<EntityNode> {
        let target_idx = *self.entity_index.get(id)?;
        let last_idx = NodeIndex::new(self.graph.node_count() - 1);
        let will_swap = target_idx != last_idx;

        // Capture the swapped node's ID before mutation
        let swapped_id = if will_swap {
            Some(self.graph[last_idx].id.clone())
        } else {
            None
        };

        // Remove from tag_index before graph mutation
        let node = &self.graph[target_idx];
        self.tag_index.remove_entity(
            &node.canonical_name,
            &node.aliases,
            &node.entity_type,
            &node.source_documents,
            target_idx,
        );

        // Remove from entity_index
        self.entity_index.remove(id);

        // Graph mutation (swap-remove)
        let removed = self.graph.remove_node(target_idx)?;

        // If a different node was swapped in, update all side-maps
        if let Some(swapped_id) = swapped_id {
            self.entity_index.insert(swapped_id, target_idx);
            self.tag_index.remap_node_index(last_idx, target_idx);
        }

        self.pagerank_dirty = true;
        Some(removed)
    }

    /// Merge two entities: keep one, absorb the other's aliases, chunks, docs, and edges.
    /// Returns the ID of the kept entity.
    pub fn merge_entities(&mut self, keep_id: &str, merge_id: &str) -> Result<String, String> {
        let keep_idx = *self
            .entity_index
            .get(keep_id)
            .ok_or_else(|| format!("Entity not found: {}", keep_id))?;
        let merge_idx = *self
            .entity_index
            .get(merge_id)
            .ok_or_else(|| format!("Entity not found: {}", merge_id))?;

        if keep_idx == merge_idx {
            return Err("Cannot merge an entity with itself".to_string());
        }

        // Collect data from the entity being merged
        let merge_node = self.graph[merge_idx].clone();
        let merge_aliases = merge_node.aliases.clone();
        let merge_chunks = merge_node.source_chunks.clone();
        let merge_docs = merge_node.source_documents.clone();
        let merge_name = merge_node.canonical_name.clone();

        // Collect edges from the merged entity
        let outgoing: Vec<(NodeIndex, CoOccurrenceEdge)> = self
            .graph
            .edges_directed(merge_idx, petgraph::Direction::Outgoing)
            .map(|e| (e.target(), e.weight().clone()))
            .collect();
        let incoming: Vec<(NodeIndex, CoOccurrenceEdge)> = self
            .graph
            .edges_directed(merge_idx, petgraph::Direction::Incoming)
            .map(|e| (e.source(), e.weight().clone()))
            .collect();

        // Update the kept entity
        let keep_node = &mut self.graph[keep_idx];
        if !keep_node.aliases.contains(&merge_name) {
            keep_node.aliases.push(merge_name);
        }
        for alias in merge_aliases {
            if !keep_node.aliases.contains(&alias) {
                keep_node.aliases.push(alias);
            }
        }
        for chunk_id in merge_chunks {
            if !keep_node.source_chunks.contains(&chunk_id) {
                keep_node.source_chunks.push(chunk_id);
            }
        }
        for doc in merge_docs {
            if !keep_node.source_documents.contains(&doc) {
                keep_node.source_documents.push(doc);
            }
        }
        keep_node.access_count += merge_node.access_count;
        keep_node.updated_at = chrono::Utc::now().timestamp();

        // Redirect edges to the kept entity (skip self-loops)
        for (target, edge) in outgoing {
            if target != keep_idx && target != merge_idx {
                self.graph.add_edge(keep_idx, target, edge);
            }
        }
        for (source, edge) in incoming {
            if source != keep_idx && source != merge_idx {
                self.graph.add_edge(source, keep_idx, edge);
            }
        }

        // Update tag_index with new aliases/docs for kept entity
        let keep_node = &self.graph[keep_idx];
        let updated_aliases = keep_node.aliases.clone();
        let updated_docs = keep_node.source_documents.clone();

        // Re-register kept entity's aliases in tag_index
        for alias in &updated_aliases {
            self.tag_index
                .alias_to_node
                .insert(alias.to_lowercase(), keep_idx);
        }
        for doc in &updated_docs {
            let entities = self
                .tag_index
                .document_to_entities
                .entry(doc.clone())
                .or_default();
            if !entities.contains(&keep_idx) {
                entities.push(keep_idx);
            }
        }

        // Update chunk tags: replace merge_id with keep_id
        for chunk in self.chunks.values_mut() {
            if let Some(pos) = chunk.tags.iter().position(|t| t == merge_id) {
                chunk.tags[pos] = keep_id.to_string();
            }
        }

        // Remove the merged entity
        self.remove_entity(merge_id);

        // Re-verify keep entity's index (may have been swapped during remove)
        // The entity_index is already updated by remove_entity, so we just
        // need to make sure tag_index is consistent
        if let Some(&new_keep_idx) = self.entity_index.get(keep_id) {
            if new_keep_idx != keep_idx {
                // Keep entity was swapped — tag_index already remapped by remove_entity
                let _ = new_keep_idx; // index is correct post-remap
            }
        }

        Ok(keep_id.to_string())
    }

    // ── Edge operations ──

    /// Add a bidirectional co-occurrence edge between two entities.
    pub fn add_cooccurrence(
        &mut self,
        entity_a_id: &str,
        entity_b_id: &str,
        edge: CoOccurrenceEdge,
    ) -> Result<(), String> {
        let &a_idx = self
            .entity_index
            .get(entity_a_id)
            .ok_or_else(|| format!("Entity not found: {}", entity_a_id))?;
        let &b_idx = self
            .entity_index
            .get(entity_b_id)
            .ok_or_else(|| format!("Entity not found: {}", entity_b_id))?;

        self.graph.add_edge(a_idx, b_idx, edge.clone());
        self.graph.add_edge(b_idx, a_idx, edge);
        self.pagerank_dirty = true;
        Ok(())
    }

    /// Get all co-occurring entities and their edges for a given entity.
    pub fn get_cooccurrences(&self, entity_id: &str) -> Vec<(String, CoOccurrenceEdge)> {
        let Some(&idx) = self.entity_index.get(entity_id) else {
            return Vec::new();
        };

        self.graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .map(|e| {
                let neighbor_id = self.graph[e.target()].id.clone();
                (neighbor_id, e.weight().clone())
            })
            .collect()
    }

    // ── Chunk storage ──

    /// Store a chunk in the side-map.
    pub fn store_chunk(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk.id.clone(), chunk);
    }

    /// Get a chunk by ID.
    pub fn get_chunk(&self, chunk_id: &str) -> Option<&Chunk> {
        self.chunks.get(chunk_id)
    }

    /// Get all chunks that tag any of the given entity IDs.
    pub fn get_chunks_for_entities(&self, entity_ids: &[String]) -> Vec<&Chunk> {
        self.chunks
            .values()
            .filter(|chunk| chunk.tags.iter().any(|tag| entity_ids.contains(tag)))
            .collect()
    }

    /// Get chunks for a single entity, sorted by timestamp (ascending).
    pub fn get_temporal_chain(&self, entity_id: &str) -> Vec<&Chunk> {
        let mut chunks: Vec<&Chunk> = self
            .chunks
            .values()
            .filter(|chunk| chunk.tags.contains(&entity_id.to_string()))
            .collect();
        chunks.sort_by_key(|c| c.timestamp.unwrap_or(c.created_at));
        chunks
    }

    // ── Query / traversal ──

    /// BFS neighborhood query from an entity, with optional time filter on edges.
    pub fn query_neighborhood(
        &self,
        entity_id: &str,
        hops: usize,
        time_start: Option<i64>,
        time_end: Option<i64>,
    ) -> SubgraphResult {
        let Some(&start_idx) = self.entity_index.get(entity_id) else {
            return SubgraphResult {
                entities: Vec::new(),
                edges: Vec::new(),
                chunks: Vec::new(),
            };
        };

        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![start_idx];
        visited.insert(start_idx);

        for _ in 0..hops {
            let mut next_frontier = Vec::new();
            for &node_idx in &frontier {
                for edge in self
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Outgoing)
                {
                    // Apply temporal filter on edge timestamps
                    if let Some(ts) = edge.weight().timestamp {
                        if let Some(start) = time_start {
                            if ts < start {
                                continue;
                            }
                        }
                        if let Some(end) = time_end {
                            if ts > end {
                                continue;
                            }
                        }
                    }

                    let target = edge.target();
                    if visited.insert(target) {
                        next_frontier.push(target);
                    }
                }
            }
            frontier = next_frontier;
        }

        // Collect results
        let entities: Vec<EntityNode> = visited
            .iter()
            .filter_map(|&idx| self.graph.node_weight(idx).cloned())
            .collect();

        let entity_ids: std::collections::HashSet<&str> =
            entities.iter().map(|e| e.id.as_str()).collect();

        let mut edges = Vec::new();
        for &idx in &visited {
            for edge in self
                .graph
                .edges_directed(idx, petgraph::Direction::Outgoing)
            {
                let target = edge.target();
                if visited.contains(&target) {
                    let from_id = self.graph[idx].id.clone();
                    let to_id = self.graph[target].id.clone();
                    edges.push((from_id, to_id, edge.weight().clone()));
                }
            }
        }

        // Collect chunks that reference any entity in the subgraph
        let entity_id_strings: Vec<String> = entity_ids.iter().map(|s| s.to_string()).collect();
        let chunks = self
            .get_chunks_for_entities(&entity_id_strings)
            .into_iter()
            .cloned()
            .collect();

        SubgraphResult {
            entities,
            edges,
            chunks,
        }
    }

    /// Search entities by name/alias substring.
    pub fn search_entities(&self, query: &str, limit: usize) -> Vec<&EntityNode> {
        self.tag_index
            .search(query, limit)
            .into_iter()
            .filter_map(|idx| self.graph.node_weight(idx))
            .collect()
    }

    // ── PageRank ──

    /// Compute PageRank scores and store them on EntityNode.rank.
    /// Same algorithm as graph.rs, using CoOccurrenceEdge::strength().
    pub fn compute_pagerank(&mut self) {
        self.compute_pagerank_with_params(20, 0.85);
    }

    fn compute_pagerank_with_params(&mut self, iterations: usize, damping_factor: f64) {
        let node_count = self.graph.node_count();
        if node_count == 0 {
            self.pagerank_dirty = false;
            return;
        }

        let weighted_out_degrees: Vec<f64> = self
            .graph
            .node_indices()
            .map(|idx| {
                self.graph
                    .edges_directed(idx, petgraph::Direction::Outgoing)
                    .map(|e| e.weight().strength())
                    .sum()
            })
            .collect();

        let mut ranks: Vec<f64> = vec![1.0 / node_count as f64; node_count];

        for _ in 0..iterations {
            let mut new_ranks = vec![0.0; node_count];
            for node_idx in self.graph.node_indices() {
                let mut rank_sum = 0.0;
                for edge in self
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                {
                    let source_idx = edge.source();
                    let w_out = weighted_out_degrees[source_idx.index()];
                    if w_out > 0.0 {
                        rank_sum +=
                            ranks[source_idx.index()] * (edge.weight().strength() / w_out);
                    }
                }
                new_ranks[node_idx.index()] =
                    (1.0 - damping_factor) / node_count as f64 + damping_factor * rank_sum;
            }
            ranks = new_ranks;
        }

        for (i, rank) in ranks.iter().enumerate() {
            self.graph[NodeIndex::new(i)].rank = *rank;
        }
        self.pagerank_dirty = false;
    }

    /// Ensure PageRank is up to date before rank-dependent operations.
    pub fn ensure_pagerank_up_to_date(&mut self) {
        if self.pagerank_dirty {
            self.compute_pagerank();
        }
    }

    /// Get top entities by PageRank score.
    pub fn get_top_entities(&mut self, limit: usize) -> Vec<&EntityNode> {
        self.ensure_pagerank_up_to_date();
        let mut ranked: Vec<(NodeIndex, f64)> = self
            .graph
            .node_indices()
            .map(|idx| (idx, self.graph[idx].rank))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(limit);
        ranked
            .into_iter()
            .filter_map(|(idx, _)| self.graph.node_weight(idx))
            .collect()
    }

    // ── Decay ──

    /// Apply exponential decay to access counts based on time since last update.
    pub fn decay_scores(&mut self, half_life_days: f64) {
        let now = chrono::Utc::now().timestamp();
        let half_life_secs = half_life_days * 86400.0;
        let ln2 = std::f64::consts::LN_2;

        for node_idx in self.graph.node_indices() {
            let node = &mut self.graph[node_idx];
            let age_secs = (now - node.updated_at) as f64;
            let decay_factor = (-ln2 * age_secs / half_life_secs).exp();
            node.access_count = (node.access_count as f64 * decay_factor).round() as u32;
        }
    }

    // ── Statistics ──

    pub fn get_statistics(&self) -> KnowledgeGraphStatistics {
        let mut entities_by_type: HashMap<String, usize> = HashMap::new();
        for node in self.graph.node_weights() {
            *entities_by_type
                .entry(node.entity_type.to_string())
                .or_default() += 1;
        }

        let documents_indexed = self.tag_index.document_to_entities.len();

        KnowledgeGraphStatistics {
            entity_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),
            chunk_count: self.chunks.len(),
            entities_by_type,
            documents_indexed,
        }
    }

    // ── Export ──

    /// Export a subgraph containing the specified entities and their interconnecting edges.
    pub fn export_subgraph(&self, entity_ids: &[String]) -> SubgraphResult {
        let idx_set: std::collections::HashSet<NodeIndex> = entity_ids
            .iter()
            .filter_map(|id| self.entity_index.get(id).copied())
            .collect();

        let entities: Vec<EntityNode> = idx_set
            .iter()
            .filter_map(|&idx| self.graph.node_weight(idx).cloned())
            .collect();

        let mut edges = Vec::new();
        for &idx in &idx_set {
            for edge in self
                .graph
                .edges_directed(idx, petgraph::Direction::Outgoing)
            {
                if idx_set.contains(&edge.target()) {
                    let from_id = self.graph[idx].id.clone();
                    let to_id = self.graph[edge.target()].id.clone();
                    edges.push((from_id, to_id, edge.weight().clone()));
                }
            }
        }

        let chunks = self
            .get_chunks_for_entities(entity_ids)
            .into_iter()
            .cloned()
            .collect();

        SubgraphResult {
            entities,
            edges,
            chunks,
        }
    }

    // ── Access to internals (for persistence) ──

    pub fn graph(&self) -> &DiGraph<EntityNode, CoOccurrenceEdge> {
        &self.graph
    }

    pub fn chunks_map(&self) -> &HashMap<String, Chunk> {
        &self.chunks
    }

    pub fn entity_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Rebuild internal indexes from the graph state.
    /// Used after loading from persistence.
    pub fn rebuild_indexes(&mut self) {
        self.entity_index.clear();
        self.tag_index = TagIndex::new();

        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            self.entity_index.insert(node.id.clone(), idx);
            self.tag_index.add_entity(
                &node.canonical_name,
                &node.aliases,
                &node.entity_type,
                &node.source_documents,
                idx,
            );
        }

        self.pagerank_dirty = true;
    }

    /// Direct access to set the graph (used by persistence loading).
    pub fn set_graph(&mut self, graph: DiGraph<EntityNode, CoOccurrenceEdge>) {
        self.graph = graph;
    }

    /// Direct access to set chunks (used by persistence loading).
    pub fn set_chunks(&mut self, chunks: HashMap<String, Chunk>) {
        self.chunks = chunks;
    }
}
