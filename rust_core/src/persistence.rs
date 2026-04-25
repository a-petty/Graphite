use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};

use crate::chunk::Chunk;
use crate::cooccurrence::CoOccurrenceEdge;
use crate::entity::EntityNode;
use crate::knowledge_graph::KnowledgeGraph;

/// Minimum file size (bytes) for a file to be considered "non-empty".
/// An empty graph serializes to ~5 bytes of msgpack; use 10 as threshold.
const NON_EMPTY_FILE_THRESHOLD: u64 = 10;
/// Maximum number of timestamped backup copies to retain.
const MAX_BACKUP_COPIES: usize = 3;
/// Suffix appended to the primary filename for the atomic-write staging file.
const TMP_SUFFIX: &str = ".tmp";

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
        self.path.join(".graphite").join("graph.msgpack")
    }

    fn tmp_file(&self) -> PathBuf {
        self.path.join(".graphite").join(format!("graph.msgpack{}", TMP_SUFFIX))
    }

    fn backup_file(&self) -> PathBuf {
        self.path.join(".graphite").join("graph.msgpack.bak")
    }

    fn backup_file_n(&self, n: usize) -> PathBuf {
        self.path.join(".graphite").join(format!("graph.msgpack.bak.{}", n))
    }

    /// Check if saving an empty graph would overwrite a non-empty file.
    /// Returns an error if the graph has 0 entities AND a non-empty primary file exists.
    fn check_empty_overwrite(&self, graph: &KnowledgeGraph) -> Result<(), String> {
        let state = self.graph_to_state(graph);
        if state.entities.is_empty() {
            let file_path = self.graph_file();
            if file_path.exists() {
                if let Ok(metadata) = fs::metadata(&file_path) {
                    if metadata.len() > NON_EMPTY_FILE_THRESHOLD {
                        return Err(
                            "Refusing to save empty graph over non-empty existing file".to_string()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Rotate timestamped backups: delete .bak.3, shift .bak.2→.bak.3,
    /// .bak.1→.bak.2, .bak→.bak.1.
    fn rotate_backups(&self) -> Result<(), String> {
        // Delete oldest backup (.bak.3)
        let bak3 = self.backup_file_n(MAX_BACKUP_COPIES);
        if bak3.exists() {
            fs::remove_file(&bak3)
                .map_err(|e| format!("Failed to delete backup {}: {}", bak3.display(), e))?;
        }

        // Shift .bak.2 → .bak.3
        let bak2 = self.backup_file_n(MAX_BACKUP_COPIES - 1);
        if bak2.exists() {
            fs::rename(&bak2, &bak3)
                .map_err(|e| format!("Failed to rotate backup {}: {}", bak2.display(), e))?;
        }

        // Shift .bak.1 → .bak.2
        let bak1 = self.backup_file_n(MAX_BACKUP_COPIES - 2);
        if bak1.exists() {
            fs::rename(&bak1, &bak2)
                .map_err(|e| format!("Failed to rotate backup {}: {}", bak1.display(), e))?;
        }

        // Shift .bak → .bak.1
        let bak = self.backup_file();
        if bak.exists() {
            fs::rename(&bak, &bak1)
                .map_err(|e| format!("Failed to rotate backup {}: {}", bak.display(), e))?;
        }

        Ok(())
    }

    /// Save the knowledge graph to MessagePack format using an atomic
    /// tmpfile + fsync + rename. Refuses to overwrite a non-empty file
    /// with an empty graph (0 entities).
    pub fn save(&self, graph: &KnowledgeGraph) -> Result<(), String> {
        self.check_empty_overwrite(graph)?;
        self.atomic_write(graph)
    }

    /// Atomic-write the graph to the primary path.
    ///
    /// Steps:
    ///   1. Serialize the graph to a msgpack byte vector.
    ///   2. Create the parent directory if missing.
    ///   3. Remove any stale tmpfile left by a prior crashed save.
    ///   4. Open tmpfile, write all bytes, fsync it.
    ///   5. Rename tmpfile → primary (atomic on all sane filesystems).
    ///   6. Fsync parent directory so the rename is durable across crashes.
    ///
    /// Any crash between steps 4 and 5 leaves the primary untouched. Any
    /// crash between steps 5 and 6 is recoverable as long as the host
    /// filesystem has reasonable ordering; the explicit fsync closes the
    /// window.
    fn atomic_write(&self, graph: &KnowledgeGraph) -> Result<(), String> {
        let state = self.graph_to_state(graph);
        let data = rmp_serde::to_vec(&state).map_err(|e| format!("Serialize error: {}", e))?;

        let file_path = self.graph_file();
        let parent = file_path
            .parent()
            .ok_or_else(|| format!("Graph path has no parent: {}", file_path.display()))?;
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        let tmp_path = self.tmp_file();

        // If a stale tmpfile exists from a prior crashed save, clear it.
        if tmp_path.exists() {
            fs::remove_file(&tmp_path)
                .map_err(|e| format!("Failed to remove stale tmpfile {}: {}", tmp_path.display(), e))?;
        }

        // Write + fsync tmpfile.
        {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp_path)
                .map_err(|e| format!("Failed to open tmpfile {}: {}", tmp_path.display(), e))?;
            f.write_all(&data)
                .map_err(|e| format!("Failed to write tmpfile: {}", e))?;
            f.sync_all()
                .map_err(|e| format!("Failed to fsync tmpfile: {}", e))?;
        }

        // Atomically replace the primary.
        fs::rename(&tmp_path, &file_path)
            .map_err(|e| format!("Failed to rename tmpfile onto primary: {}", e))?;

        // Fsync parent dir so the rename is durable.
        // On Unix, opening a directory as a File and calling sync_all flushes
        // its metadata. Errors here are non-fatal — the rename already landed;
        // we just couldn't confirm durability.
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }

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

    /// Save with backup: rotate timestamped backups, copy the current primary
    /// into the .bak slot, then atomic-write the new state over the primary.
    ///
    /// Unlike the previous implementation, this does NOT rename the primary
    /// away before writing — the atomic write replaces primary in place. The
    /// backup is a copy, so the primary is always present on disk.
    pub fn save_with_backup(&self, graph: &KnowledgeGraph) -> Result<(), String> {
        self.check_empty_overwrite(graph)?;

        let file_path = self.graph_file();

        // Snapshot the current primary into the backup chain before we touch it.
        if file_path.exists() {
            self.rotate_backups()?;
            let bak = self.backup_file();
            fs::copy(&file_path, &bak)
                .map_err(|e| format!("Failed to copy primary into backup slot: {}", e))?;
        }

        self.atomic_write(graph)
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

#[cfg(test)]
mod tests {
    //! Inline tests covering atomic-write invariants that the integration
    //! test suite cannot reach without exposing internals. Public behavior
    //! is also covered in `rust_core/tests/test_persistence.rs`.

    use super::*;
    use crate::chunk::{ChunkType, MemoryCategory};
    use crate::entity::EntityType;

    fn populated_graph(root: &Path) -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new(root);
        let alice = EntityNode::new("Alice".to_string(), EntityType::Person);
        kg.add_entity(alice);
        let mut chunk = Chunk::new(
            "doc.md".to_string(),
            ChunkType::Decision,
            MemoryCategory::Episodic,
            "hello".to_string(),
        );
        chunk.timestamp = Some(1);
        kg.store_chunk(chunk);
        kg
    }

    #[test]
    fn tmpfile_cleaned_up_after_successful_save() {
        let dir = tempfile::tempdir().unwrap();
        let store = GraphStore::new(dir.path());
        let graph = populated_graph(dir.path());

        store.save(&graph).unwrap();

        assert!(!store.tmp_file().exists(), "tmpfile must not persist after a successful save");
        assert!(store.graph_file().exists(), "primary must exist after save");
    }

    #[test]
    fn stale_tmpfile_is_replaced_not_appended() {
        let dir = tempfile::tempdir().unwrap();
        let store = GraphStore::new(dir.path());
        let graph = populated_graph(dir.path());

        // Plant a junk tmpfile as if a prior save had crashed mid-write.
        fs::create_dir_all(store.tmp_file().parent().unwrap()).unwrap();
        fs::write(store.tmp_file(), b"garbage from a crashed save").unwrap();
        assert!(store.tmp_file().exists());

        // A fresh save must clean it up and produce a valid primary.
        store.save(&graph).unwrap();
        assert!(!store.tmp_file().exists());
        let loaded = store.load(dir.path()).unwrap();
        assert_eq!(loaded.entity_count(), 1);
    }

    #[test]
    fn crashed_save_after_primary_leaves_primary_intact() {
        // Simulate a crash between the atomic rename and any subsequent work:
        // after atomic_write returns, the primary is durable. We verify this
        // by loading immediately and confirming the file decodes.
        let dir = tempfile::tempdir().unwrap();
        let store = GraphStore::new(dir.path());
        let graph = populated_graph(dir.path());

        store.save_with_backup(&graph).unwrap();
        assert!(store.graph_file().exists());

        // Simulate a crash *during* the next save by leaving a partial tmpfile
        // and verifying the primary still loads fine.
        fs::write(store.tmp_file(), b"partial write from a killed save").unwrap();
        let loaded = store.load(dir.path()).unwrap();
        assert_eq!(loaded.entity_count(), 1);
    }

    #[test]
    fn backup_is_a_copy_not_a_rename_so_primary_never_disappears() {
        let dir = tempfile::tempdir().unwrap();
        let store = GraphStore::new(dir.path());
        let graph = populated_graph(dir.path());

        store.save(&graph).unwrap();
        let primary_inode_before = fs::metadata(store.graph_file()).unwrap();

        store.save_with_backup(&graph).unwrap();

        // Both primary and .bak exist post-save_with_backup.
        assert!(store.graph_file().exists());
        assert!(store.backup_file().exists());

        // Primary should always be loadable — no window where it is absent.
        let _ = primary_inode_before; // metadata handle dropped; existence checked above
        let loaded = store.load(dir.path()).unwrap();
        assert_eq!(loaded.entity_count(), 1);
    }
}
