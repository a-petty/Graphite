use std::collections::HashMap;
use std::path::{Path, PathBuf};
use petgraph::graph::NodeIndex;

/// A global index of all symbols found in the repository.
#[derive(Debug, Default)]
pub struct SymbolIndex {
    /// Maps a symbol name (e.g., "User") to a list of file paths that define it.
    /// A `Vec` is used because the same symbol can be defined in multiple files (e.g., collisions).
    pub definitions: HashMap<String, Vec<PathBuf>>,
    
    /// Maps a file path to a list of all external symbols it uses.
    pub usages: HashMap<PathBuf, Vec<String>>,

    /// Maps a symbol name to a list of file paths that use it.
    /// This is the reverse of `usages` for efficient lookups.
    pub users: HashMap<String, Vec<PathBuf>>,
}

impl SymbolIndex {
    /// Creates a new, empty `SymbolIndex`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove all references to a specific file
    /// Called when a file is deleted from the graph
    pub fn remove_node_references(&mut self, path: &Path) {
        // Remove from definitions (this file no longer provides symbols)
        for symbol_paths in self.definitions.values_mut() {
            symbol_paths.retain(|p| p != path);
        }
        
        // Remove from users (this file no longer uses symbols)
        for user_paths in self.users.values_mut() {
            user_paths.retain(|p| p != path);
        }
        
        // Remove from usages (this file's usage list is gone)
        self.usages.remove(path);
        
        // Clean up empty entries to prevent memory leaks
        self.definitions.retain(|_, paths| !paths.is_empty());
        self.users.retain(|_, paths| !paths.is_empty());
    }
    
    /// Remap a node index (currently no-op, reserved for future optimization)
    /// 
    /// FUTURE USE: If we add NodeIndex-based caching for O(1) symbol lookups,
    /// this method will update those cached mappings after a swap-remove.
    /// 
    /// Example future field:
    ///   node_to_definitions: HashMap<NodeIndex, Vec<String>>
    ///   node_to_usages: HashMap<NodeIndex, Vec<String>>
    /// 
    /// This would allow: graph.symbol_index.get_definitions(node_idx)
    /// Instead of:       graph.symbol_index.get_definitions(graph[node_idx].path)
    pub fn remap_node_index(&mut self, _old_idx: NodeIndex, _new_idx: NodeIndex) {
        // Currently a no-op because we use PathBuf keys
        // Paths don't change when nodes swap positions
        
        // FUTURE IMPLEMENTATION (when adding NodeIndex caching):
        // if let Some(defs) = self.node_to_definitions.remove(&_old_idx) {
        //     self.node_to_definitions.insert(_new_idx, defs);
        // }
        // if let Some(uses) = self.node_to_usages.remove(&_old_idx) {
        //     self.node_to_usages.insert(_new_idx, uses);
        // }
    }
}
