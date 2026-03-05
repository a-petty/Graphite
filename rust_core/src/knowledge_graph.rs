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

/// Result of removing a document and cascading cleanup.
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentRemovalResult {
    pub chunks_removed: usize,
    pub edges_removed: usize,
    pub entities_removed: usize,
    pub entities_updated: usize,
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
    document_hashes: HashMap<String, String>,
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
            document_hashes: HashMap::new(),
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
            keep_node.aliases.push(merge_name.clone());
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

        // Record merge in audit trail
        keep_node.merge_history.push(crate::entity::MergeRecord {
            merged_entity_id: merge_id.to_string(),
            merged_entity_name: merge_name.clone(),
            merged_at: chrono::Utc::now().timestamp(),
            confidence: 0.0,
            method: "direct".to_string(),
        });

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
        if entity_a_id == entity_b_id {
            return Err(format!(
                "Cannot create self-loop: entity '{}' cannot co-occur with itself",
                entity_a_id
            ));
        }

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

    /// Store a chunk in the side-map and update tagged entities' source_chunks.
    pub fn store_chunk(&mut self, chunk: Chunk) {
        let chunk_id = chunk.id.clone();
        let tags = chunk.tags.clone();
        self.chunks.insert(chunk_id.clone(), chunk);

        for entity_id in &tags {
            if let Some(entity) = self.get_entity_mut(entity_id) {
                if !entity.source_chunks.contains(&chunk_id) {
                    entity.source_chunks.push(chunk_id.clone());
                }
            }
        }
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
    /// Uses entity's source_chunks for O(k) lookup when available,
    /// with fallback to full scan for pre-fix graphs.
    pub fn get_temporal_chain(&self, entity_id: &str) -> Vec<&Chunk> {
        let mut chunks: Vec<&Chunk> = if let Some(entity) = self.get_entity(entity_id) {
            if !entity.source_chunks.is_empty() {
                entity.source_chunks.iter()
                    .filter_map(|cid| self.chunks.get(cid))
                    .collect()
            } else {
                self.chunks
                    .values()
                    .filter(|chunk| chunk.tags.contains(&entity_id.to_string()))
                    .collect()
            }
        } else {
            return Vec::new();
        };
        chunks.sort_by_key(|c| c.timestamp.unwrap_or(c.created_at));
        chunks
    }

    // ── Document-level operations ──

    /// Get all chunks belonging to a specific document.
    pub fn get_chunks_by_document(&self, document: &str) -> Vec<&Chunk> {
        self.chunks
            .values()
            .filter(|c| c.source_document == document)
            .collect()
    }

    /// Remove a document and cascade-clean all its artifacts:
    /// chunks, orphaned edges, and orphaned entities.
    pub fn remove_document(&mut self, document: &str) -> DocumentRemovalResult {
        // 1. Collect chunk IDs for this document
        let chunk_ids: std::collections::HashSet<String> = self
            .chunks
            .values()
            .filter(|c| c.source_document == document)
            .map(|c| c.id.clone())
            .collect();

        let chunks_removed = chunk_ids.len();

        // 2. Remove edges whose chunk_id is in the document's chunk set
        let mut edges_to_remove = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge) = self.graph.edge_weight(edge_idx) {
                if chunk_ids.contains(&edge.chunk_id) {
                    edges_to_remove.push(edge_idx);
                }
            }
        }
        // Remove in reverse index order to avoid invalidation
        edges_to_remove.sort_by(|a, b| b.cmp(a));
        for edge_idx in &edges_to_remove {
            self.graph.remove_edge(*edge_idx);
        }
        let edges_removed = edges_to_remove.len();

        // 3. Remove chunks from chunk store
        for cid in &chunk_ids {
            self.chunks.remove(cid);
        }

        // 4. Update entities: trim source_documents and source_chunks.
        //    First pass: mutate nodes. Second pass (read-only): check edges.
        let mut entities_maybe_orphaned: Vec<(NodeIndex, String)> = Vec::new();
        let mut entities_updated = 0usize;

        for idx in self.graph.node_indices() {
            let node = &mut self.graph[idx];
            let had_doc = node.source_documents.contains(&document.to_string());
            if !had_doc {
                continue;
            }

            // Trim this document from source_documents
            node.source_documents.retain(|d| d != document);
            // Trim chunks belonging to this document
            node.source_chunks.retain(|c| !chunk_ids.contains(c));

            if node.source_documents.is_empty() && node.source_chunks.is_empty() {
                entities_maybe_orphaned.push((idx, node.id.clone()));
            } else {
                entities_updated += 1;
            }
        }

        // Second pass: check edges for possibly-orphaned entities (immutable borrow)
        let mut entities_to_remove = Vec::new();
        for (idx, eid) in entities_maybe_orphaned {
            let has_edges = self
                .graph
                .edges_directed(idx, petgraph::Direction::Outgoing)
                .next()
                .is_some()
                || self
                    .graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                    .next()
                    .is_some();
            if !has_edges {
                entities_to_remove.push(eid);
            } else {
                entities_updated += 1;
            }
        }

        // 5. Remove queued entities (collected first, removed second)
        let entities_removed = entities_to_remove.len();
        for eid in entities_to_remove {
            self.remove_entity(&eid);
        }

        // 6. Remove document from tag_index.document_to_entities
        self.tag_index.document_to_entities.remove(document);

        // 7. Mark pagerank dirty
        if chunks_removed > 0 || edges_removed > 0 || entities_removed > 0 {
            self.pagerank_dirty = true;
        }

        DocumentRemovalResult {
            chunks_removed,
            edges_removed,
            entities_removed,
            entities_updated,
        }
    }

    // ── Document hash tracking ──

    /// Get the stored content hash for a document.
    pub fn get_document_hash(&self, document: &str) -> Option<&str> {
        self.document_hashes.get(document).map(|s| s.as_str())
    }

    /// Store a content hash for a document.
    pub fn set_document_hash(&mut self, document: String, hash: String) {
        self.document_hashes.insert(document, hash);
    }

    /// Remove and return the content hash for a document.
    pub fn remove_document_hash(&mut self, document: &str) -> Option<String> {
        self.document_hashes.remove(document)
    }

    /// Get all tracked document paths.
    pub fn tracked_documents(&self) -> Vec<&str> {
        self.document_hashes.keys().map(|s| s.as_str()).collect()
    }

    /// Get reference to document_hashes (for persistence).
    pub fn document_hashes(&self) -> &HashMap<String, String> {
        &self.document_hashes
    }

    /// Set document_hashes (for persistence loading).
    pub fn set_document_hashes(&mut self, hashes: HashMap<String, String>) {
        self.document_hashes = hashes;
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
                    let has_time_filter = time_start.is_some() || time_end.is_some();
                    match edge.weight().timestamp {
                        Some(ts) => {
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
                        None if has_time_filter => continue,
                        None => {}
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

    // ── Reflection utilities ──

    /// Return all entity IDs in the graph.
    pub fn all_entity_ids(&self) -> Vec<String> {
        self.entity_index.keys().cloned().collect()
    }

    /// Return entity IDs with zero edges (no co-occurrences).
    pub fn find_orphan_entities(&self) -> Vec<String> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .edges_directed(idx, petgraph::Direction::Outgoing)
                    .next()
                    .is_none()
                    && self
                        .graph
                        .edges_directed(idx, petgraph::Direction::Incoming)
                        .next()
                        .is_none()
            })
            .filter_map(|idx| self.graph.node_weight(idx).map(|n| n.id.clone()))
            .collect()
    }

    /// Recalculate edge weights based on co-occurrence frequency.
    ///
    /// For each pair of connected entities, counts the number of parallel
    /// edges and sets each edge's weight to that count. Returns the number
    /// of edges updated.
    pub fn recalculate_edge_weights(&mut self) -> usize {
        // Count edges per (source, target) pair
        let mut pair_counts: HashMap<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex), f32> =
            HashMap::new();
        for edge in self.graph.edge_indices() {
            if let Some((src, tgt)) = self.graph.edge_endpoints(edge) {
                *pair_counts.entry((src, tgt)).or_default() += 1.0;
            }
        }

        // Update weights
        let mut updated = 0;
        for edge_idx in self.graph.edge_indices() {
            if let Some((src, tgt)) = self.graph.edge_endpoints(edge_idx) {
                if let Some(&count) = pair_counts.get(&(src, tgt)) {
                    if let Some(edge) = self.graph.edge_weight_mut(edge_idx) {
                        edge.weight = count;
                        updated += 1;
                    }
                }
            }
        }

        self.pagerank_dirty = true;
        updated
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
