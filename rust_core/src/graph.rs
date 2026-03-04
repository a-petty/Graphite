use crate::import_resolver::{ImportResolver, NoopImportResolver};
use crate::parser::SupportedLanguage;
use crate::symbol_table::SymbolIndex;
use log::{debug, info};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use thiserror::Error;
use std::sync::Arc;
use lru::LruCache;
use std::num::NonZeroUsize;
use parking_lot::RwLock;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Write;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Represents an error that can occur during graph operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("Node not found for path: {0}")]
    NodeNotFound(PathBuf),
    #[error("File I/O error: {0}")]
    IoError(String),
    #[error("Parsing failed for file: {0}")]
    ParseError(PathBuf),
    #[error("Unsupported language for file: {0}")]
    UnsupportedLanguage(PathBuf),
}

impl From<std::io::Error> for GraphError {
    fn from(err: std::io::Error) -> Self {
        GraphError::IoError(err.to_string())
    }
}

/// Result of a graph update operation
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub struct UpdateResult {
    pub edges_added: usize,
    pub edges_removed: usize,
    pub needs_pagerank_recalc: bool,
}

/// Represents a node in the repository graph, which is a single source file.
#[derive(Debug, Clone)]
pub struct FileNode {
    pub path: PathBuf,
    pub language: SupportedLanguage,
    pub definitions: Vec<String>,
    pub usages: Vec<String>,
    pub rank: f64,

    // Cached data for fast change classification
    pub imports_hash: u64,
    pub definitions_hash: u64,
    pub usages_hash: u64,
    pub content_hash: u64,
}

impl FileNode {
    /// Creates a new FileNode with computed hashes
    pub fn new(
        path: PathBuf,
        language: SupportedLanguage,
        definitions: Vec<String>,
        usages: Vec<String>,
        imports: &HashSet<PathBuf>,
        content: &str,
    ) -> Self {
        Self {
            path,
            language,
            definitions: definitions.clone(),
            usages: usages.clone(),
            rank: 0.0,
            imports_hash: Self::hash_imports(imports),
            definitions_hash: Self::hash_vec(&definitions),
            usages_hash: Self::hash_vec(&usages),
            content_hash: Self::hash_content(content),
        }
    }

    /// Creates an empty node (used when initializing before full parse)
    pub fn empty(path: PathBuf) -> Self {
        let language = SupportedLanguage::from_path(&path);
        Self {
            path,
            language,
            definitions: Vec::new(),
            usages: Vec::new(),
            rank: 0.0,
            imports_hash: 0,
            definitions_hash: 0,
            usages_hash: 0,
            content_hash: 0,
        }
    }

    pub fn hash_imports(imports: &HashSet<PathBuf>) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut sorted: Vec<_> = imports.iter().collect();
        sorted.sort();
        for import in sorted {
            import.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn hash_vec(vec: &[String]) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut sorted = vec.to_vec();
        sorted.sort();
        for item in sorted {
            item.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn hash_content(content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }
}

/// Represents the type of dependency between two files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeKind {
    /// A heuristic dependency based on symbol name matching.
    SymbolUsage,
    /// A structurally confirmed dependency based on an import statement.
    Import,
}

impl EdgeKind {
    pub fn strength(&self) -> f64 {
        match self {
            EdgeKind::Import => 2.0,
            EdgeKind::SymbolUsage => 1.0,
        }
    }
}

/// Represents the magnitude of a change to a file, used for tiered updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateTier {
    Local,
    FileScope,
    GraphScope,
    FullRebuild,
}

/// Statistics about the repository graph
#[derive(Debug)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub import_edges: usize,
    pub symbol_edges: usize,
    pub total_definitions: usize,
    pub total_files_with_usages: usize,
    pub unresolved_import_count: usize,
    pub source_roots: Vec<PathBuf>,
    pub module_index_size: usize,
    pub known_root_modules: Vec<String>,
    pub attempted_imports: usize,
    pub failed_imports: usize,
}

#[derive(Debug)]
struct DirTree {
    files: Vec<(PathBuf, f64)>,
    subdirs: BTreeMap<String, DirTree>,
}

impl DirTree {
    fn new() -> Self {
        Self {
            files: Vec::new(),
            subdirs: BTreeMap::new(),
        }
    }
}

/// The main repository graph, representing the entire codebase's structure.
#[derive(Debug)]
pub struct RepoGraph {
    pub graph: DiGraph<FileNode, EdgeKind>,
    pub path_to_idx: HashMap<PathBuf, NodeIndex>,
    pub symbol_index: SymbolIndex,
    pub import_resolver: Box<dyn ImportResolver>,
    pub pagerank_dirty: bool,
    pub unresolved_imports: HashMap<PathBuf, HashSet<NodeIndex>>,
    pub project_root: PathBuf,
    pub skeleton_cache: RwLock<LruCache<PathBuf, Arc<String>>>,
}

impl RepoGraph {
    pub fn new(project_root: &Path, _language: &str, _ignored_dirs: &[String], _source_roots: Option<&[String]>) -> Self {
        let canonical_root = project_root.canonicalize()
            .unwrap_or_else(|_| project_root.to_path_buf());

        let cache_size = NonZeroUsize::new(500).unwrap();

        Self {
            graph: DiGraph::new(),
            path_to_idx: HashMap::new(),
            symbol_index: SymbolIndex::new(),
            import_resolver: Box::new(NoopImportResolver::new()),
            pagerank_dirty: true,
            unresolved_imports: HashMap::new(),
            project_root: canonical_root,
            skeleton_cache: RwLock::new(LruCache::new(cache_size)),
        }
    }

    /// Get skeleton for a file. With tree-sitter removed, returns source as-is.
    pub fn get_skeleton(&self, path: &Path) -> Result<Arc<String>, GraphError> {
        let canonical_path = self.project_root.join(path);

        // Check cache first
        {
            let mut cache = self.skeleton_cache.write();
            if let Some(skeleton) = cache.get(&canonical_path) {
                return Ok(Arc::clone(skeleton));
            }
        }

        // Read and cache source directly (no tree-sitter skeleton extraction)
        let source = fs::read_to_string(&canonical_path)
            .map_err(|e| GraphError::IoError(e.to_string()))?;

        let skeleton_arc = Arc::new(source);
        let mut cache = self.skeleton_cache.write();
        cache.put(canonical_path, Arc::clone(&skeleton_arc));
        Ok(skeleton_arc)
    }

    /// Helper to ensure an edge exists. Import edges take priority over SymbolUsage edges.
    fn ensure_edge(&mut self, source: NodeIndex, target: NodeIndex, kind: EdgeKind) -> bool {
        if let Some(edge_id) = self.graph.find_edge(source, target) {
            let existing_kind = &mut self.graph[edge_id];
            if *existing_kind == EdgeKind::SymbolUsage && kind == EdgeKind::Import {
                *existing_kind = EdgeKind::Import;
                return true;
            }
            return false;
        } else {
            self.graph.add_edge(source, target, kind);
            return true;
        }
    }

    /// Stub: updates a single file in the graph.
    /// With tree-sitter removed, only updates the content hash.
    pub fn update_file(
        &mut self,
        file_path: &PathBuf,
        source_code: &str,
    ) -> Result<UpdateResult, GraphError> {
        let file_path = &file_path.canonicalize().unwrap_or_else(|_| file_path.clone());
        debug!("Updating file: {}", file_path.display());

        if let Some(&node_idx) = self.path_to_idx.get(file_path) {
            if let Some(node) = self.graph.node_weight_mut(node_idx) {
                node.content_hash = FileNode::hash_content(source_code);
            }
        } else {
            return Err(GraphError::NodeNotFound(file_path.clone()));
        }

        Ok(UpdateResult::default())
    }

    /// Ensures PageRank scores are up-to-date before any rank-dependent operation.
    pub fn ensure_pagerank_up_to_date(&mut self) {
        if self.pagerank_dirty {
            self.calculate_pagerank(20, 0.85);
        }
    }

    /// Simplified build: reads files and creates nodes with no parsing.
    /// Graph has nodes but no edges until the extraction pipeline is built.
    pub fn build(&mut self, paths: &[PathBuf]) {
        info!("Starting simplified build for {} files...", paths.len());

        let canonical_paths: Vec<PathBuf> = paths
            .iter()
            .map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()))
            .collect();

        // Read files in parallel to compute content hashes
        let file_data: Vec<(PathBuf, u64)> = canonical_paths
            .par_iter()
            .filter_map(|path| {
                let lang = SupportedLanguage::from_extension(
                    path.extension().and_then(|s| s.to_str()).unwrap_or(""),
                );
                if lang == SupportedLanguage::Unknown {
                    return None;
                }
                if let Ok(source_code) = fs::read_to_string(path) {
                    let content_hash = FileNode::hash_content(&source_code);
                    Some((path.clone(), content_hash))
                } else {
                    None
                }
            })
            .collect();

        info!("File scanning complete. Found {} supported files.", file_data.len());

        // Create nodes (serial)
        for (path, content_hash) in &file_data {
            let lang = SupportedLanguage::from_path(path);
            let mut file_node = FileNode::empty(path.clone());
            file_node.language = lang;
            file_node.content_hash = *content_hash;

            let node_idx = self.graph.add_node(file_node);
            self.path_to_idx.insert(path.clone(), node_idx);
        }

        info!("Node creation complete. Graph has {} nodes, 0 edges.", self.graph.node_count());
        self.pagerank_dirty = true;
    }

    /// Simplified add_file: creates a node with no parsing.
    pub fn add_file(&mut self, path: PathBuf, content: &str) -> Result<(), GraphError> {
        let path = path.canonicalize().unwrap_or(path);

        if self.path_to_idx.contains_key(&path) {
            self.remove_file(&path)?;
        }

        let lang = SupportedLanguage::from_extension(path.extension().and_then(|s| s.to_str()).unwrap_or(""));
        if lang == SupportedLanguage::Unknown {
            return Ok(());
        }

        let mut file_node = FileNode::empty(path.clone());
        file_node.language = lang;
        file_node.content_hash = FileNode::hash_content(content);

        let new_idx = self.graph.add_node(file_node);
        self.path_to_idx.insert(path.clone(), new_idx);

        // Check if any nodes were waiting for THIS file
        if let Some(waiting_nodes) = self.unresolved_imports.remove(&path) {
            for waiting_idx in waiting_nodes {
                if waiting_idx != new_idx {
                    self.ensure_edge(waiting_idx, new_idx, EdgeKind::Import);
                }
            }
        }

        self.pagerank_dirty = true;
        Ok(())
    }

    pub fn remove_file(&mut self, path: &Path) -> Result<(), GraphError> {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let path = canonical.as_path();

        // Resolve path to index
        let target_idx = *self.path_to_idx.get(path)
            .ok_or_else(|| GraphError::NodeNotFound(path.to_path_buf()))?;

        // Identify swap consequences
        let last_idx = NodeIndex::new(self.graph.node_count() - 1);
        let will_swap = target_idx != last_idx;

        let moved_path = if will_swap {
            Some(self.graph[last_idx].path.clone())
        } else {
            None
        };

        // Downgrade incoming import edges to unresolved
        let incoming_importers: Vec<PathBuf> = self.graph
            .edges_directed(target_idx, petgraph::Direction::Incoming)
            .filter(|e| *e.weight() == EdgeKind::Import)
            .map(|e| self.graph[e.source()].path.clone())
            .collect();

        for importer_path in incoming_importers {
            if let Some(importer_idx) = self.path_to_idx.get(&importer_path).copied() {
                self.unresolved_imports
                    .entry(path.to_path_buf())
                    .or_default()
                    .insert(importer_idx);
            }
        }

        // Symbol cleanup
        let node = self.graph[target_idx].clone();
        for def in &node.definitions {
            if let Some(paths) = self.symbol_index.definitions.get_mut(def) {
                paths.retain(|p| p != path);
                if paths.is_empty() {
                    self.symbol_index.definitions.remove(def);
                }
            }
        }
        for r#use in &node.usages {
            if let Some(paths) = self.symbol_index.users.get_mut(r#use) {
                paths.retain(|p| p != path);
                if paths.is_empty() {
                    self.symbol_index.users.remove(r#use);
                }
            }
        }
        self.symbol_index.usages.remove(path);

        // Graph mutation (swap-remove)
        self.graph.remove_node(target_idx);

        // Path map updates
        self.path_to_idx.remove(path);

        if let Some(moved_path) = moved_path {
            self.path_to_idx.insert(moved_path.clone(), target_idx);
            self.remap_symbol_index(last_idx, target_idx);
            for waiting_set in self.unresolved_imports.values_mut() {
                if waiting_set.remove(&last_idx) {
                    waiting_set.insert(target_idx);
                }
            }
        }

        self.pagerank_dirty = true;
        Ok(())
    }

    fn remap_symbol_index(&mut self, old_idx: NodeIndex, new_idx: NodeIndex) {
        self.symbol_index.remap_node_index(old_idx, new_idx);
    }

    /// Simplified build_complete: creates nodes and runs PageRank.
    /// No import edges or symbol edges without a parsing pipeline.
    pub fn build_complete(&mut self, paths: &[PathBuf], _project_root: &Path) {
        let canonical_paths: Vec<PathBuf> = paths
            .iter()
            .map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()))
            .collect();
        self.build(&canonical_paths);
        self.calculate_pagerank(20, 0.85);
        info!("Build complete. Final graph: {} nodes, {} edges.", self.graph.node_count(), self.graph.edge_count());
    }

    /// Get statistics about the graph
    pub fn get_statistics(&self) -> GraphStatistics {
        let (import_edges, symbol_edges) = self.graph.edge_indices().fold((0, 0), |(i, s), edge_idx| {
            match self.graph[edge_idx] {
                EdgeKind::Import => (i + 1, s),
                EdgeKind::SymbolUsage => (i, s + 1),
            }
        });

        GraphStatistics {
            node_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),
            import_edges,
            symbol_edges,
            total_definitions: self.symbol_index.definitions.len(),
            total_files_with_usages: self.symbol_index.usages.len(),
            unresolved_import_count: self.unresolved_imports.values().map(|s| s.len()).sum(),
            source_roots: self.import_resolver.get_source_roots(),
            module_index_size: self.import_resolver.module_index_size(),
            known_root_modules: self.import_resolver.get_known_root_modules(),
            attempted_imports: self.import_resolver.get_attempted_imports(),
            failed_imports: self.import_resolver.get_failed_imports(),
        }
    }

    pub fn get_unresolved_imports_sample(&self, limit: usize) -> Vec<(PathBuf, usize)> {
        self.unresolved_imports.iter()
            .take(limit)
            .map(|(path, waiters)| (path.clone(), waiters.len()))
            .collect()
    }

    pub fn get_incoming_dependencies(&self, file_path: &Path) -> Vec<(PathBuf, EdgeKind)> {
        let canonical = file_path.canonicalize().unwrap_or_else(|_| file_path.to_path_buf());
        if let Some(&node_idx) = self.path_to_idx.get(canonical.as_path()) {
            self.graph.edges_directed(node_idx, petgraph::Direction::Incoming)
                .map(|edge| (self.graph[edge.source()].path.clone(), edge.weight().clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn has_file(&self, file_path: &Path) -> bool {
        let canonical = file_path.canonicalize().unwrap_or_else(|_| file_path.to_path_buf());
        self.path_to_idx.contains_key(canonical.as_path())
    }

    pub fn get_outgoing_dependencies(&self, file_path: &Path) -> Vec<(PathBuf, EdgeKind)> {
        let canonical = file_path.canonicalize().unwrap_or_else(|_| file_path.to_path_buf());
        if let Some(&node_idx) = self.path_to_idx.get(canonical.as_path()) {
            self.graph.edges_directed(node_idx, petgraph::Direction::Outgoing)
                .map(|edge| (self.graph[edge.target()].path.clone(), edge.weight().clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn calculate_pagerank(&mut self, iterations: usize, damping_factor: f64) {
        let node_count = self.graph.node_count();
        if node_count == 0 { return; }

        let weighted_out_degrees: Vec<f64> = self.graph.node_indices()
            .map(|idx| {
                self.graph.edges_directed(idx, petgraph::Direction::Outgoing)
                    .map(|e| e.weight().strength())
                    .sum()
            })
            .collect();

        let mut ranks: Vec<f64> = vec![1.0 / node_count as f64; node_count];

        for _ in 0..iterations {
            let mut new_ranks = vec![0.0; node_count];
            for node_idx in self.graph.node_indices() {
                let mut rank_sum = 0.0;
                for edge in self.graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let source_idx = edge.source();
                    let w_out = weighted_out_degrees[source_idx.index()];
                    if w_out > 0.0 {
                        rank_sum += ranks[source_idx.index()] * (edge.weight().strength() / w_out);
                    }
                }
                new_ranks[node_idx.index()] = (1.0 - damping_factor) / node_count as f64
                    + damping_factor * rank_sum;
            }
            ranks = new_ranks;
        }

        for (i, rank) in ranks.iter().enumerate() {
            self.graph[NodeIndex::new(i)].rank = *rank;
        }
        self.pagerank_dirty = false;
    }

    pub fn get_top_ranked_files(&mut self, limit: usize) -> Vec<(PathBuf, f64)> {
        self.ensure_pagerank_up_to_date();
        let mut ranked_files: Vec<_> = self.graph.node_weights().map(|node| (node.path.clone(), node.rank)).collect();
        ranked_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked_files.truncate(limit);
        ranked_files
    }

    pub fn get_dependents(&self, file_path: &Path) -> Vec<(PathBuf, EdgeKind)> {
        self.get_incoming_dependencies(file_path)
    }

    pub fn get_dependencies(&self, file_path: &Path) -> Vec<(PathBuf, EdgeKind)> {
        self.get_outgoing_dependencies(file_path)
    }

    fn calculate_stars(&self, rank: f64, max_rank: f64) -> usize {
        if max_rank == 0.0 { return 1; }
        ((rank / max_rank).clamp(0.0, 1.0) * 5.0).ceil().max(1.0).min(5.0) as usize
    }

    fn format_stars(&self, star_count: usize) -> String { "★".repeat(star_count) }

    fn get_max_rank(&self) -> f64 { self.graph.node_weights().map(|n| n.rank).fold(0.0, f64::max) }

    pub fn generate_map(&mut self, max_files: usize) -> String {
        self.ensure_pagerank_up_to_date();

        let mut output = String::new();

        writeln!(output, "Repository Map ({} files)\n", self.graph.node_count()).unwrap();

        let top_files = self.get_top_ranked_files(max_files);
        if !top_files.is_empty() {
            writeln!(output, "TOP RANKED FILES (by architectural importance):").unwrap();
            for (i, (path, rank)) in top_files.iter().enumerate() {
                let dependents_count = self.get_dependents(path).len();
                let display_path = pathdiff::diff_paths(path, &self.project_root)
                    .unwrap_or_else(|| path.clone());
                writeln!(output, "  {}. {} [rank: {:.3}, imported by: {} files]", i + 1, display_path.display(), rank, dependents_count).unwrap();
            }
            writeln!(output).unwrap();
        }

        self.generate_directory_structure(&mut output, Some(&self.project_root));
        output
    }

    fn generate_directory_structure(&self, output: &mut String, project_root: Option<&Path>) {
        writeln!(output, "DIRECTORY STRUCTURE:").unwrap();
        let mut all_files: Vec<_> = self.graph.node_weights().map(|n| {
            let path = project_root.and_then(|root| pathdiff::diff_paths(&n.path, root)).unwrap_or_else(|| n.path.clone());
            (path, n.rank)
        }).collect();
        all_files.sort_by(|a, b| a.0.cmp(&b.0));
        let tree = self.build_directory_tree(&all_files);
        let max_rank = self.get_max_rank();
        self.render_directory_structure_recursive(output, &tree, "", "", max_rank);
    }

    fn build_directory_tree(&self, files: &[(PathBuf, f64)]) -> DirTree {
        let mut root = DirTree::new();
        for (path, rank) in files {
            let mut current = &mut root;
            if let Some(parent) = path.parent() {
                for component in parent.components() {
                    if let std::path::Component::Normal(s) = component {
                        current = current.subdirs.entry(s.to_string_lossy().into_owned()).or_insert_with(DirTree::new);
                    }
                }
            }
            current.files.push((path.clone(), *rank));
        }
        root
    }

    fn render_directory_structure_recursive(&self, output: &mut String, tree: &DirTree, prefix: &str, name: &str, max_rank: f64) {
        if !name.is_empty() { writeln!(output, "{}├── {}/", prefix, name).unwrap(); }
        let child_prefix = if name.is_empty() { String::new() } else { format!("{}│   ", prefix) };
        let mut sorted_subdirs: Vec<_> = tree.subdirs.iter().collect();
        sorted_subdirs.sort_by_key(|(name, _)| *name);

        for (i, (subdir_name, subtree)) in sorted_subdirs.iter().enumerate() {
            let is_last = i == sorted_subdirs.len() - 1 && tree.files.is_empty();
            let new_prefix = if is_last { format!("{}    ", prefix) } else { format!("{}│   ", prefix) };
            self.render_directory_structure_recursive(output, subtree, &new_prefix, subdir_name, max_rank);
        }

        let mut sorted_files: Vec<_> = tree.files.iter().collect();
        sorted_files.sort_by_key(|(path, _)| path.file_name());

        for (i, (file_path, rank)) in sorted_files.iter().enumerate() {
            let is_last = i == sorted_files.len() - 1;
            let connector = if is_last { "└──" } else { "├──" };
            let file_name = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
            let star_str = self.format_stars(self.calculate_stars(*rank, max_rank));
            writeln!(output, "{}{} {} {} (rank: {:.3})", child_prefix, connector, file_name, star_str, rank).unwrap();
        }
    }

    /// Validates the internal consistency of the symbol index and graph.
    pub fn validate_consistency(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.path_to_idx.len() != self.graph.node_count() {
            errors.push(format!(
                "Size mismatch: path_to_idx has {} entries, graph has {} nodes",
                self.path_to_idx.len(),
                self.graph.node_count()
            ));
        }

        for (path, &idx) in &self.path_to_idx {
            if let Some(node) = self.graph.node_weight(idx) {
                if node.path != *path {
                    errors.push(format!(
                        "Mismatch at index {:?}: path_to_idx expects '{}', but node contains '{}'",
                        idx, path.display(), node.path.display()
                    ));
                }
            } else {
                errors.push(format!(
                    "path_to_idx points to invalid index {:?} for path '{}'",
                    idx, path.display()
                ));
            }
        }

        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            if !self.path_to_idx.contains_key(&node.path) {
                errors.push(format!(
                    "Node at index {:?} with path '{}' has no path_to_idx entry",
                    idx, node.path.display()
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
