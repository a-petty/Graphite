#![allow(non_local_definitions)]
#![allow(unsafe_op_in_unsafe_fn)]
pub mod parser;
pub mod import_resolver;
pub mod graph;
pub mod symbol_table;
pub mod watcher;

// Knowledge graph modules (Phase 1)
pub mod entity;
pub mod chunk;
pub mod cooccurrence;
pub mod knowledge_graph;
pub mod tag_index;
pub mod persistence;

use pyo3::prelude::*;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError};
use std::time::Duration;
use std::path::{Path, PathBuf};
use ignore::{WalkBuilder, DirEntry};
use std::collections::HashSet;
use crate::parser::create_skeleton_from_source;
use crate::graph::GraphStatistics;


// Create a Python exception type for our custom error.
create_exception!(semantic_engine, GraphError, PyException);
create_exception!(semantic_engine, ParseError, GraphError);
create_exception!(semantic_engine, NodeNotFoundError, GraphError);


// Implement the conversion from our internal Rust error to the Python exception.
impl From<graph::GraphError> for PyErr {
    fn from(err: graph::GraphError) -> PyErr {
        match err {
            graph::GraphError::ParseError(path) => {
                ParseError::new_err(format!(
                    "Syntax error in {}: unable to parse file",
                    path.display()
                ))
            }
            graph::GraphError::NodeNotFound(path) => {
                NodeNotFoundError::new_err(format!(
                    "File not found in graph: {}",
                    path.display()
                ))
            }
            graph::GraphError::IoError(msg) => {
                GraphError::new_err(format!("I/O error: {}", msg))
            }
            graph::GraphError::UnsupportedLanguage(path) => {
                GraphError::new_err(format!(
                    "Unsupported language for file: {}",
                    path.display()
                ))
            }
        }
    }
}

/// A simple struct to hold the result of a syntax check.
#[pyclass(name="SyntaxCheckResult")]
struct PySyntaxCheckResult {
    #[pyo3(get)]
    is_valid: bool,
    #[pyo3(get)]
    line: usize,
    #[pyo3(get)]
    message: String,
}

/// Stub: check_syntax now only supports Python (via native compile()).
/// Tree-sitter validation was removed in Phase 0b.
#[pyfunction]
fn check_syntax(content: &str, lang_name: &str) -> PyResult<PySyntaxCheckResult> {
    if lang_name == "python" || lang_name == "py" {
        return check_python_syntax_native(content);
    }

    // For non-Python languages, always report valid (no tree-sitter)
    Ok(PySyntaxCheckResult {
        is_valid: true,
        line: 0,
        message: "".to_string(),
    })
}

fn check_python_syntax_native(content: &str) -> PyResult<PySyntaxCheckResult> {
    Python::with_gil(|py| {
        let compile = py.eval("compile", None, None)?;

        match compile.call1((content, "<string>", "exec")) {
            Ok(_) => Ok(PySyntaxCheckResult {
                is_valid: true,
                line: 0,
                message: "".to_string(),
            }),
            Err(e) => {
                let line = if let Ok(syntax_err) = e.value(py).getattr("lineno") {
                    syntax_err.extract::<usize>().unwrap_or(0)
                } else {
                    0
                };

                let message = format!("Syntax error: {}", e.value(py));

                Ok(PySyntaxCheckResult {
                    is_valid: false,
                    line,
                    message,
                })
            }
        }
    })
}


#[pyfunction]
#[pyo3(signature = (path, ignored_dirs = None))]
fn scan_repository(path: &str, ignored_dirs: Option<Vec<String>>) -> PyResult<Vec<String>> {
    let mut files = Vec::new();
    let ignored_set: HashSet<String> = ignored_dirs.unwrap_or_default().into_iter().collect();

    let walker = WalkBuilder::new(path)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .parents(true)
        .filter_entry(move |entry: &DirEntry| {
            if entry.file_name() == ".git" {
                return false;
            }
            if let Some(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    return entry.file_name().to_str()
                        .map(|s| !ignored_set.contains(s))
                        .unwrap_or(true);
                }
            }
            true
        })
        .build();

    for result in walker {
        if let Ok(entry) = result {
            if let Ok(canonical_path) = entry.path().canonicalize() {
                if canonical_path.is_file() {
                    if let Some(path_str) = canonical_path.to_str() {
                        files.push(path_str.to_string());
                    }
                }
            }
        }
    }

    Ok(files)
}

/// Python-facing wrapper for the main repository graph.
#[pyclass(name = "RepoGraph")]
pub struct PyRepoGraph {
    graph: graph::RepoGraph,
}

/// Python-facing wrapper for the result of a graph update operation.
#[pyclass(name = "GraphUpdateResult")]
#[derive(Clone)]
pub struct PyGraphUpdateResult {
    #[pyo3(get)]
    pub edges_added: usize,
    #[pyo3(get)]
    pub edges_removed: usize,
    #[pyo3(get)]
    pub needs_pagerank_recalc: bool,
}

impl From<graph::UpdateResult> for PyGraphUpdateResult {
    fn from(res: graph::UpdateResult) -> Self {
        Self {
            edges_added: res.edges_added,
            edges_removed: res.edges_removed,
            needs_pagerank_recalc: res.needs_pagerank_recalc,
        }
    }
}

#[pyclass(name = "GraphStatistics")]
pub struct PyGraphStatistics {
    #[pyo3(get)]
    pub node_count: usize,
    #[pyo3(get)]
    pub edge_count: usize,
    #[pyo3(get)]
    pub import_edges: usize,
    #[pyo3(get)]
    pub symbol_edges: usize,
    #[pyo3(get)]
    pub total_definitions: usize,
    #[pyo3(get)]
    pub total_files_with_usages: usize,
    #[pyo3(get)]
    pub unresolved_import_count: usize,
    #[pyo3(get)]
    pub source_roots: Vec<String>,
    #[pyo3(get)]
    pub module_index_size: usize,
    #[pyo3(get)]
    pub known_root_modules: Vec<String>,
    #[pyo3(get)]
    pub attempted_imports: usize,
    #[pyo3(get)]
    pub failed_imports: usize,
}

impl From<GraphStatistics> for PyGraphStatistics {
    fn from(stats: GraphStatistics) -> Self {
        Self {
            node_count: stats.node_count,
            edge_count: stats.edge_count,
            import_edges: stats.import_edges,
            symbol_edges: stats.symbol_edges,
            total_definitions: stats.total_definitions,
            total_files_with_usages: stats.total_files_with_usages,
            unresolved_import_count: stats.unresolved_import_count,
            source_roots: stats.source_roots.iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect(),
            module_index_size: stats.module_index_size,
            known_root_modules: stats.known_root_modules,
            attempted_imports: stats.attempted_imports,
            failed_imports: stats.failed_imports,
        }
    }
}


#[pymethods]
impl PyRepoGraph {
    #[new]
    #[pyo3(signature = (project_root, language = "python", ignored_dirs = None, source_roots = None))]
    fn new(project_root: &str, language: &str, ignored_dirs: Option<Vec<String>>, source_roots: Option<Vec<String>>) -> PyResult<Self> {
        let root_path = Path::new(project_root);
        let dirs = ignored_dirs.unwrap_or_default();
        Ok(Self {
            graph: graph::RepoGraph::new(root_path, language, &dirs, source_roots.as_deref()),
        })
    }

    fn build_complete(&mut self, file_paths: Vec<String>) {
        let paths: Vec<PathBuf> = file_paths.into_iter().map(PathBuf::from).collect();
        self.graph.build_complete(&paths, &self.graph.project_root.clone());
    }

    fn add_file(&mut self, path: String, content: String) -> PyResult<()> {
        let path_buf = PathBuf::from(path);
        self.graph.add_file(path_buf, &content)
            .map_err(|e| e.into())
    }

    fn remove_file(&mut self, path: String) -> PyResult<()> {
        let path_buf = PathBuf::from(path);
        self.graph.remove_file(&path_buf)
            .map_err(|e| e.into())
    }

    fn update_file(&mut self, file_path: &str, content: &str) -> PyResult<PyGraphUpdateResult> {
        let path = PathBuf::from(file_path);
        let result = self.graph.update_file(&path, content)?;
        Ok(result.into())
    }

    fn ensure_pagerank_up_to_date(&mut self) {
        self.graph.ensure_pagerank_up_to_date();
    }

    fn generate_map(&mut self, max_files: usize) -> String {
        self.graph.generate_map(max_files)
    }

    fn get_statistics(&self) -> PyGraphStatistics {
        self.graph.get_statistics().into()
    }

    fn get_dependents(&self, file_path: &str) -> PyResult<Vec<(String, String)>> {
        let path = PathBuf::from(file_path);
        let dependencies = self.graph.get_dependents(&path);
        Ok(dependencies.into_iter().map(|(p, k)| (p.to_string_lossy().into_owned(), format!("{:?}", k))).collect())
    }

    fn get_dependencies(&self, file_path: &str) -> PyResult<Vec<(String, String)>> {
        let path = PathBuf::from(file_path);
        let dependencies = self.graph.get_dependencies(&path);
        Ok(dependencies.into_iter().map(|(p, k)| (p.to_string_lossy().into_owned(), format!("{:?}", k))).collect())
    }

    fn has_file(&self, file_path: &str) -> bool {
        let path = PathBuf::from(file_path);
        self.graph.has_file(&path)
    }

    fn get_unresolved_imports(&self, limit: usize) -> Vec<(String, usize)> {
        self.graph.get_unresolved_imports_sample(limit)
            .into_iter()
            .map(|(path, count)| (path.to_string_lossy().into_owned(), count))
            .collect()
    }

    fn get_top_ranked_files(&mut self, limit: usize) -> Vec<(String, f64)> {
        self.graph.get_top_ranked_files(limit)
            .into_iter()
            .map(|(path, rank): (PathBuf, f64)| (path.to_string_lossy().into_owned(), rank))
            .collect()
    }

    #[pyo3(name = "get_skeleton")]
    pub fn get_skeleton(&self, path: String) -> PyResult<String> {
        let skeleton_arc = self.graph.get_skeleton(Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to get skeleton: {}", e)
            ))?;
        Ok(skeleton_arc.as_ref().clone())
    }
}


/// Python-facing file change event
#[pyclass(name = "FileChangeEvent")]
#[derive(Clone)]
pub struct PyFileChangeEvent {
    #[pyo3(get)]
    pub event_type: String,

    #[pyo3(get)]
    pub path: String,
}

#[pymethods]
impl PyFileChangeEvent {
    fn __repr__(&self) -> String {
        format!("FileChangeEvent(type='{}', path='{}')", self.event_type, self.path)
    }

    fn __str__(&self) -> String {
        format!("{}: {}", self.event_type, self.path)
    }
}

impl From<watcher::FileChangeEvent> for PyFileChangeEvent {
    fn from(event: watcher::FileChangeEvent) -> Self {
        match event {
            watcher::FileChangeEvent::Created(path) => PyFileChangeEvent {
                event_type: "created".to_string(),
                path: path.display().to_string(),
            },
            watcher::FileChangeEvent::Modified(path) => PyFileChangeEvent {
                event_type: "modified".to_string(),
                path: path.display().to_string(),
            },
            watcher::FileChangeEvent::Deleted(path) => PyFileChangeEvent {
                event_type: "deleted".to_string(),
                path: path.display().to_string(),
            },
            watcher::FileChangeEvent::Renamed { from, to } => PyFileChangeEvent {
                event_type: "renamed".to_string(),
                path: format!("{} -> {}", from.display(), to.display()),
            },
        }
    }
}

/// Python-facing watcher statistics
#[pyclass(name = "WatcherStats")]
#[derive(Clone)]
pub struct PyWatcherStats {
    #[pyo3(get)]
    pub events_received: usize,

    #[pyo3(get)]
    pub events_filtered: usize,

    #[pyo3(get)]
    pub errors_encountered: usize,
}

#[pymethods]
impl PyWatcherStats {
    fn __repr__(&self) -> String {
        format!(
            "WatcherStats(received={}, filtered={}, errors={})",
            self.events_received, self.events_filtered, self.errors_encountered
        )
    }
}

impl From<watcher::WatcherStats> for PyWatcherStats {
    fn from(stats: watcher::WatcherStats) -> Self {
        Self {
            events_received: stats.events_received,
            events_filtered: stats.events_filtered,
            errors_encountered: stats.errors_encountered,
        }
    }
}

/// Python-facing file watcher
#[pyclass(name = "FileWatcher")]
pub struct PyFileWatcher {
    watcher: Option<watcher::FileWatcher>,
}

#[pymethods]
impl PyFileWatcher {
    #[new]
    #[pyo3(signature = (path, extensions=None, ignored_dirs=None))]
    fn new(
        path: String,
        extensions: Option<Vec<String>>,
        ignored_dirs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut filter = watcher::FileFilter::default();

        if let Some(exts) = extensions {
            filter.extensions = exts;
        }

        if let Some(dirs) = ignored_dirs {
            filter.ignored_dirs.extend(dirs);
        }

        let watcher = watcher::FileWatcher::new(PathBuf::from(path), filter)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start watcher: {}", e)))?;

        Ok(Self {
            watcher: Some(watcher),
        })
    }

    fn poll_events(&self) -> PyResult<Vec<PyFileChangeEvent>> {
        let watcher = self.watcher.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Watcher has been stopped"))?;

        Ok(watcher.poll_events()
            .into_iter()
            .map(PyFileChangeEvent::from)
            .collect())
    }

    fn wait_for_event(&self, timeout_ms: u64) -> PyResult<Option<PyFileChangeEvent>> {
        let watcher = self.watcher.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Watcher has been stopped"))?;

        Ok(watcher.wait_for_event(Duration::from_millis(timeout_ms))
            .map(PyFileChangeEvent::from))
    }

    fn get_stats(&self) -> PyResult<PyWatcherStats> {
        let watcher = self.watcher.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Watcher has been stopped"))?;

        Ok(PyWatcherStats::from(watcher.get_stats()))
    }

    fn stop(&mut self) {
        self.watcher = None;
    }
}


// ── PyKnowledgeGraph: PyO3 wrapper for KnowledgeGraph ──

use crate::knowledge_graph::KnowledgeGraph;
use crate::entity::{EntityNode, EntityType};
use crate::chunk::{Chunk, ChunkType, MemoryCategory};
use crate::cooccurrence::CoOccurrenceEdge;
use crate::persistence::GraphStore;

/// Helper to parse EntityType from a JSON string value.
fn parse_entity_type(s: &str) -> EntityType {
    match s {
        "Person" => EntityType::Person,
        "Project" => EntityType::Project,
        "Technology" => EntityType::Technology,
        "Organization" => EntityType::Organization,
        "Location" => EntityType::Location,
        "Decision" => EntityType::Decision,
        "Concept" => EntityType::Concept,
        "Document" => EntityType::Document,
        other => EntityType::Custom(other.to_string()),
    }
}

/// Helper to parse ChunkType from a JSON string value.
fn parse_chunk_type(s: &str) -> ChunkType {
    match s {
        "Decision" => ChunkType::Decision,
        "Discussion" => ChunkType::Discussion,
        "ActionItem" => ChunkType::ActionItem,
        "StatusUpdate" => ChunkType::StatusUpdate,
        "Preference" => ChunkType::Preference,
        "Background" => ChunkType::Background,
        _ => ChunkType::Discussion,
    }
}

/// Helper to parse MemoryCategory from a JSON string value.
fn parse_memory_category(s: &str) -> MemoryCategory {
    match s {
        "Episodic" => MemoryCategory::Episodic,
        "Semantic" => MemoryCategory::Semantic,
        "Procedural" => MemoryCategory::Procedural,
        _ => MemoryCategory::Episodic,
    }
}

#[pyclass(name = "PyKnowledgeGraph")]
pub struct PyKnowledgeGraph {
    kg: KnowledgeGraph,
}

#[pymethods]
impl PyKnowledgeGraph {
    #[new]
    fn new(root_path: &str) -> Self {
        Self {
            kg: KnowledgeGraph::new(Path::new(root_path)),
        }
    }

    /// Add an entity from a JSON string. Returns the entity ID.
    /// Expected JSON: {"canonical_name": "...", "entity_type": "Person", "aliases": [...]}
    fn add_entity(&mut self, entity_json: &str) -> PyResult<String> {
        let v: serde_json::Value = serde_json::from_str(entity_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON: {}", e)))?;

        let canonical_name = v["canonical_name"]
            .as_str()
            .ok_or_else(|| PyRuntimeError::new_err("Missing canonical_name"))?
            .to_string();

        let entity_type = v["entity_type"]
            .as_str()
            .map(parse_entity_type)
            .unwrap_or(EntityType::Concept);

        let mut node = EntityNode::new(canonical_name, entity_type);

        if let Some(aliases) = v["aliases"].as_array() {
            node.aliases = aliases
                .iter()
                .filter_map(|a| a.as_str().map(String::from))
                .collect();
        }

        if let Some(docs) = v["source_documents"].as_array() {
            node.source_documents = docs
                .iter()
                .filter_map(|d| d.as_str().map(String::from))
                .collect();
        }

        let id = node.id.clone();
        self.kg.add_entity(node);
        Ok(id)
    }

    /// Add a co-occurrence edge between two entities.
    /// Expected edge_json: {"chunk_id": "...", "chunk_type": "Decision", "memory_category": "Episodic", "source_document": "..."}
    fn add_cooccurrence(
        &mut self,
        entity_a_id: &str,
        entity_b_id: &str,
        edge_json: &str,
    ) -> PyResult<()> {
        let v: serde_json::Value = serde_json::from_str(edge_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON: {}", e)))?;

        let chunk_id = v["chunk_id"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let chunk_type = v["chunk_type"]
            .as_str()
            .map(parse_chunk_type)
            .unwrap_or(ChunkType::Discussion);
        let memory_category = v["memory_category"]
            .as_str()
            .map(parse_memory_category)
            .unwrap_or(MemoryCategory::Episodic);
        let timestamp = v["timestamp"].as_i64();
        let source_document = v["source_document"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let edge = CoOccurrenceEdge::new(
            chunk_id,
            chunk_type,
            memory_category,
            timestamp,
            source_document,
        );

        self.kg
            .add_cooccurrence(entity_a_id, entity_b_id, edge)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Store a chunk from JSON.
    /// Expected: {"source_document": "...", "chunk_type": "Decision", "memory_category": "Episodic", "text": "...", ...}
    fn store_chunk(&mut self, chunk_json: &str) -> PyResult<String> {
        let v: serde_json::Value = serde_json::from_str(chunk_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON: {}", e)))?;

        let source_document = v["source_document"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let chunk_type = v["chunk_type"]
            .as_str()
            .map(parse_chunk_type)
            .unwrap_or(ChunkType::Discussion);
        let memory_category = v["memory_category"]
            .as_str()
            .map(parse_memory_category)
            .unwrap_or(MemoryCategory::Episodic);
        let text = v["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let mut chunk = Chunk::new(source_document, chunk_type, memory_category, text);

        if let Some(section) = v["section_name"].as_str() {
            chunk.section_name = Some(section.to_string());
        }
        if let Some(speaker) = v["speaker"].as_str() {
            chunk.speaker = Some(speaker.to_string());
        }
        if let Some(ts) = v["timestamp"].as_i64() {
            chunk.timestamp = Some(ts);
        }
        if let Some(tags) = v["tags"].as_array() {
            chunk.tags = tags
                .iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect();
        }

        let id = chunk.id.clone();
        self.kg.store_chunk(chunk);
        Ok(id)
    }

    /// Get a chunk by ID, returned as JSON.
    fn get_chunk(&self, chunk_id: &str) -> PyResult<Option<String>> {
        match self.kg.get_chunk(chunk_id) {
            Some(chunk) => {
                let json = serde_json::to_string(chunk)
                    .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))?;
                Ok(Some(json))
            }
            None => Ok(None),
        }
    }

    /// Get an entity by ID, returned as JSON.
    fn get_entity(&self, entity_id: &str) -> PyResult<Option<String>> {
        match self.kg.get_entity(entity_id) {
            Some(entity) => {
                let json = serde_json::to_string(entity)
                    .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))?;
                Ok(Some(json))
            }
            None => Ok(None),
        }
    }

    /// Merge two entities. Returns the ID of the kept entity.
    /// Optionally records confidence and method on the merge history record.
    #[pyo3(signature = (keep_id, merge_id, confidence=None, method=None))]
    fn merge_entities(
        &mut self,
        keep_id: &str,
        merge_id: &str,
        confidence: Option<f64>,
        method: Option<&str>,
    ) -> PyResult<String> {
        let result = self.kg
            .merge_entities(keep_id, merge_id)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        if confidence.is_some() || method.is_some() {
            if let Some(entity) = self.kg.get_entity_mut(&result) {
                if let Some(record) = entity.merge_history.last_mut() {
                    if let Some(c) = confidence {
                        record.confidence = c;
                    }
                    if let Some(m) = method {
                        record.method = m.to_string();
                    }
                }
            }
        }
        Ok(result)
    }

    /// Query the neighborhood of an entity via BFS. Returns JSON SubgraphResult.
    #[pyo3(signature = (entity_id, hops, time_start = None, time_end = None))]
    fn query_neighborhood(
        &self,
        entity_id: &str,
        hops: usize,
        time_start: Option<i64>,
        time_end: Option<i64>,
    ) -> PyResult<String> {
        let result = self.kg.query_neighborhood(entity_id, hops, time_start, time_end);
        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get all co-occurrences for an entity. Returns JSON array.
    fn get_cooccurrences(&self, entity_id: &str) -> PyResult<String> {
        let cooccurrences = self.kg.get_cooccurrences(entity_id);
        serde_json::to_string(&cooccurrences)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Search entities by name/alias substring. Returns JSON array of EntityNodes.
    fn search_entities(&self, query: &str, limit: usize) -> PyResult<String> {
        let results = self.kg.search_entities(query, limit);
        serde_json::to_string(&results)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Compute PageRank. Returns JSON array of (id, score) pairs.
    fn compute_pagerank(&mut self) -> PyResult<String> {
        self.kg.compute_pagerank();
        let top = self.kg.get_top_entities(self.kg.entity_count());
        let pairs: Vec<(&str, f64)> = top.iter().map(|e| (e.id.as_str(), e.rank)).collect();
        serde_json::to_string(&pairs)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get temporal chain (chunks sorted by timestamp) for an entity. Returns JSON.
    fn get_temporal_chain(&self, entity_id: &str) -> PyResult<String> {
        let chain = self.kg.get_temporal_chain(entity_id);
        serde_json::to_string(&chain)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get chunks for a list of entity IDs (JSON array input). Returns JSON.
    fn get_chunks_for_entities(&self, entity_ids_json: &str) -> PyResult<String> {
        let ids: Vec<String> = serde_json::from_str(entity_ids_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON: {}", e)))?;
        let chunks = self.kg.get_chunks_for_entities(&ids);
        serde_json::to_string(&chunks)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Apply exponential decay to access counts.
    fn decay_scores(&mut self, half_life_days: f64) -> PyResult<()> {
        self.kg.decay_scores(half_life_days);
        Ok(())
    }

    /// Export a subgraph for given entity IDs (JSON array input). Returns JSON.
    fn export_subgraph(&self, entity_ids_json: &str) -> PyResult<String> {
        let ids: Vec<String> = serde_json::from_str(entity_ids_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON: {}", e)))?;
        let result = self.kg.export_subgraph(&ids);
        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Save the graph to disk at the given path.
    /// Refuses to save if the graph is empty (0 entities) and a non-empty
    /// data file already exists, preventing accidental data loss.
    fn save(&self, path: &str) -> PyResult<()> {
        let store = GraphStore::new(Path::new(path));

        // Extra guard: check entity count before delegating to store
        let entity_count = self.kg.entity_count();
        if entity_count == 0 {
            let graph_file = Path::new(path).join(".graphite").join("graph.msgpack");
            if graph_file.exists() {
                if let Ok(metadata) = std::fs::metadata(&graph_file) {
                    if metadata.len() > 10 {
                        log::warn!(
                            "Refusing to save empty graph (0 entities) over non-empty file ({} bytes): {}",
                            metadata.len(),
                            graph_file.display()
                        );
                        return Err(PyRuntimeError::new_err(
                            "Refusing to save empty graph over non-empty existing file"
                        ));
                    }
                }
            }
        }

        store
            .save_with_backup(&self.kg)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Load a graph from disk into this instance. Replaces current graph data.
    fn load(&mut self, path: &str) -> PyResult<()> {
        let store = GraphStore::new(Path::new(path));
        let kg = store
            .load(Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e))?;
        self.kg = kg;
        Ok(())
    }

    /// Create a new PyKnowledgeGraph by loading from disk. Static constructor.
    /// Use this instead of calling load() as a classmethod.
    #[staticmethod]
    fn from_path(path: &str) -> PyResult<Self> {
        let store = GraphStore::new(Path::new(path));
        let kg = store
            .load(Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(Self { kg })
    }

    /// Get graph statistics as JSON.
    fn get_statistics(&self) -> PyResult<String> {
        let stats = self.kg.get_statistics();
        serde_json::to_string(&stats)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Remove an entity from the graph. Returns true if entity existed.
    fn remove_entity(&mut self, entity_id: &str) -> bool {
        self.kg.remove_entity(entity_id).is_some()
    }

    /// Get top entities by PageRank. Returns JSON array of EntityNodes.
    fn get_top_entities(&mut self, limit: usize) -> PyResult<String> {
        let top = self.kg.get_top_entities(limit);
        serde_json::to_string(&top)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get all entity IDs. Returns JSON array of strings.
    fn all_entity_ids(&self) -> PyResult<String> {
        let ids = self.kg.all_entity_ids();
        serde_json::to_string(&ids)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Find orphan entities (no edges). Returns JSON array of entity IDs.
    fn find_orphan_entities(&self) -> PyResult<String> {
        let orphans = self.kg.find_orphan_entities();
        serde_json::to_string(&orphans)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Recalculate edge weights by co-occurrence frequency. Returns count of edges updated.
    fn recalculate_edge_weights(&mut self) -> usize {
        self.kg.recalculate_edge_weights()
    }

    /// Remove parallel (duplicate) edges between the same entity pair.
    /// Returns the number of duplicate edges removed.
    fn deduplicate_edges(&mut self) -> usize {
        self.kg.deduplicate_edges()
    }

    /// Remove edges whose weight is below the given threshold.
    /// Returns the number of edges pruned.
    fn prune_edges_below_weight(&mut self, threshold: f32) -> usize {
        self.kg.prune_edges_below_weight(threshold)
    }

    /// Remove a document and cascade-clean its chunks, edges, and orphaned entities.
    /// Returns JSON DocumentRemovalResult.
    fn remove_document(&mut self, document: &str) -> PyResult<String> {
        let result = self.kg.remove_document(document);
        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get all chunks belonging to a document. Returns JSON array.
    fn get_chunks_by_document(&self, document: &str) -> PyResult<String> {
        let chunks = self.kg.get_chunks_by_document(document);
        serde_json::to_string(&chunks)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get all chunks whose timestamp falls within `[start, end]` (inclusive).
    /// Pass `None` on either bound for an open-ended window. Chunks without a
    /// timestamp are skipped. Returns JSON array.
    #[pyo3(signature = (start = None, end = None))]
    fn get_chunks_by_time_window(
        &self,
        start: Option<i64>,
        end: Option<i64>,
    ) -> PyResult<String> {
        let chunks = self.kg.get_chunks_by_time_window(start, end);
        serde_json::to_string(&chunks)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get all entities tagged with the given project. Returns JSON array.
    fn get_entities_by_project(&self, project: &str) -> PyResult<String> {
        let entities = self.kg.get_entities_by_project(project);
        serde_json::to_string(&entities)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }

    /// Get the stored content hash for a document.
    fn get_document_hash(&self, document: &str) -> Option<String> {
        self.kg.get_document_hash(document).map(|s| s.to_string())
    }

    /// Store a content hash for a document.
    fn set_document_hash(&mut self, document: &str, hash: &str) {
        self.kg.set_document_hash(document.to_string(), hash.to_string());
    }

    /// Remove the content hash for a document. Returns true if hash existed.
    fn remove_document_hash(&mut self, document: &str) -> bool {
        self.kg.remove_document_hash(document).is_some()
    }

    /// Get all tracked document paths. Returns JSON array.
    fn tracked_documents(&self) -> PyResult<String> {
        let docs = self.kg.tracked_documents();
        serde_json::to_string(&docs)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {}", e)))
    }
}


/// Graphite Semantic Engine
///
/// A knowledge graph engine for LLM memory (transitioning from code analysis).
#[pymodule]
fn semantic_engine(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(scan_repository, m)?)?;
    m.add_function(wrap_pyfunction!(check_syntax, m)?)?;
    m.add_function(wrap_pyfunction!(create_skeleton_from_source, m)?)?;

    m.add_class::<PyRepoGraph>()?;
    m.add_class::<PyGraphStatistics>()?;

    m.add_class::<PyFileWatcher>()?;

    m.add_class::<PyGraphUpdateResult>()?;

    m.add_class::<PyFileChangeEvent>()?;

    m.add_class::<PyWatcherStats>()?;
    m.add_class::<PySyntaxCheckResult>()?;

    // Knowledge graph (Phase 1)
    m.add_class::<PyKnowledgeGraph>()?;

    m.add("GraphError", _py.get_type::<GraphError>())?;
    m.add("ParseError", _py.get_type::<ParseError>())?;
    m.add("NodeNotFoundError", _py.get_type::<NodeNotFoundError>())?;

    Ok(())
}
