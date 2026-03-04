//! Import resolution stubs.
//!
//! Tree-sitter-based import resolution was removed in Phase 0b.
//! This module retains the trait and types needed by graph.rs.
//! It will be replaced by a coreference resolver in Phase 2.

use std::fmt::Debug;
use std::path::PathBuf;

/// A local binding created by an import statement.
#[derive(Debug, Clone)]
pub struct ImportBinding {
    /// The local name used in this file (e.g., "utils", "HttpClient")
    pub local_name: String,
    /// The resolved file path this binding points to
    pub resolved_path: PathBuf,
    /// The specific symbol imported (None for module imports like `import utils`)
    pub imported_symbol: Option<String>,
}

/// Common interface for language-specific import resolution.
///
/// All methods return empty results now that tree-sitter is removed.
/// Retained so graph.rs can compile with the same field type.
pub trait ImportResolver: Debug + Send + Sync {
    fn file_extensions(&self) -> &[&str];
    fn get_source_roots(&self) -> Vec<PathBuf> { Vec::new() }
    fn module_index_size(&self) -> usize { 0 }
    fn get_known_root_modules(&self) -> Vec<String> { Vec::new() }
    fn get_attempted_imports(&self) -> usize { 0 }
    fn get_failed_imports(&self) -> usize { 0 }
}

/// No-op import resolver used while tree-sitter is stripped.
#[derive(Debug)]
pub struct NoopImportResolver;

impl NoopImportResolver {
    pub fn new() -> Self {
        Self
    }
}

impl ImportResolver for NoopImportResolver {
    fn file_extensions(&self) -> &[&str] {
        &[]
    }
}
