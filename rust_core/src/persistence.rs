use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};

use crate::chunk::Chunk;
use crate::cooccurrence::CoOccurrenceEdge;
use crate::entity::EntityNode;
use crate::knowledge_graph::KnowledgeGraph;

/// Serializable intermediate representation of a KnowledgeGraph.
/// petgraph's DiGraph doesn't serialize portably, so we extract nodes/edges
/// into plain vectors, serialize those, then rebuild on load.
#[derive(Serialize, Deserialize)]
struct KnowledgeGraphState {
    entities: Vec<EntityNode>,
    edges: Vec<(usize, usize, CoOccurrenceEdge)>,
    chunks: Vec<Chunk>,
    #[serde(default)]
    document_hashes: HashMap<String, String>,
}

/// Handles saving and loading KnowledgeGraph to/from disk via MessagePack.
pub struct GraphStore {
    path: PathBuf,
}

impl GraphStore {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    fn graph_file(&self) -> PathBuf {
        self.path.join(".cortex").join("graph.msgpack")
    }

    fn backup_file(&self) -> PathBuf {
        self.path.join(".cortex").join("graph.msgpack.bak")
    }

    /// Save the knowledge graph to MessagePack format.
    pub fn save(&self, graph: &KnowledgeGraph) -> Result<(), String> {
        let state = self.graph_to_state(graph);
        let data = rmp_serde::to_vec(&state).map_err(|e| format!("Serialize error: {}", e))?;

        let file_path = self.graph_file();
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }
        fs::write(&file_path, data).map_err(|e| format!("Failed to write graph: {}", e))?;
        Ok(())
    }

    /// Load a knowledge graph from MessagePack format.
    pub fn load(&self, root_path: &Path) -> Result<KnowledgeGraph, String> {
        let file_path = self.graph_file();
        let data =
            fs::read(&file_path).map_err(|e| format!("Failed to read graph file: {}", e))?;
        let state: KnowledgeGraphState =
            rmp_serde::from_slice(&data).map_err(|e| format!("Deserialize error: {}", e))?;
        Ok(self.state_to_graph(state, root_path))
    }

    /// Save with backup: rename existing file to .bak, then save.
    pub fn save_with_backup(&self, graph: &KnowledgeGraph) -> Result<(), String> {
        let file_path = self.graph_file();
        let backup_path = self.backup_file();

        // If primary exists, move to backup
        if file_path.exists() {
            fs::rename(&file_path, &backup_path)
                .map_err(|e| format!("Failed to create backup: {}", e))?;
        }

        self.save(graph)
    }

    /// Try to load from primary file, fall back to backup.
    pub fn recover(&self, root_path: &Path) -> Result<KnowledgeGraph, String> {
        match self.load(root_path) {
            Ok(graph) => Ok(graph),
            Err(primary_err) => {
                let backup_path = self.backup_file();
                if backup_path.exists() {
                    let data = fs::read(&backup_path)
                        .map_err(|e| format!("Failed to read backup: {}", e))?;
                    let state: KnowledgeGraphState = rmp_serde::from_slice(&data)
                        .map_err(|e| format!("Backup deserialize error: {}", e))?;
                    Ok(self.state_to_graph(state, root_path))
                } else {
                    Err(format!(
                        "Primary load failed ({}), no backup available",
                        primary_err
                    ))
                }
            }
        }
    }

    /// Export the graph as JSON for debugging.
    pub fn export_json(&self, graph: &KnowledgeGraph) -> Result<String, String> {
        let state = self.graph_to_state(graph);
        serde_json::to_string_pretty(&state).map_err(|e| format!("JSON serialize error: {}", e))
    }

    // ── Internal helpers ──

    fn graph_to_state(&self, graph: &KnowledgeGraph) -> KnowledgeGraphState {
        let g = graph.graph();
        let entities: Vec<EntityNode> = g.node_weights().cloned().collect();
        let edges: Vec<(usize, usize, CoOccurrenceEdge)> = g
            .edge_indices()
            .map(|e| {
                let (source, target) = g.edge_endpoints(e).unwrap();
                (source.index(), target.index(), g[e].clone())
            })
            .collect();
        let chunks: Vec<Chunk> = graph.chunks_map().values().cloned().collect();

        KnowledgeGraphState {
            entities,
            edges,
            chunks,
            document_hashes: graph.document_hashes().clone(),
        }
    }

    fn state_to_graph(&self, state: KnowledgeGraphState, root_path: &Path) -> KnowledgeGraph {
        let mut petgraph = DiGraph::new();

        // Add nodes
        for entity in &state.entities {
            petgraph.add_node(entity.clone());
        }

        // Add edges
        for (from_idx, to_idx, edge) in state.edges {
            let from = petgraph::graph::NodeIndex::new(from_idx);
            let to = petgraph::graph::NodeIndex::new(to_idx);
            petgraph.add_edge(from, to, edge);
        }

        // Build chunks map
        let chunks: HashMap<String, Chunk> =
            state.chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

        let mut kg = KnowledgeGraph::new(root_path);
        kg.set_graph(petgraph);
        kg.set_chunks(chunks);
        kg.set_document_hashes(state.document_hashes);
        kg.rebuild_indexes();
        kg
    }
}
