use pyo3::prelude::*;
use std::path::Path;

/// Stub: returns source as-is (tree-sitter removed in Phase 0b).
#[pyfunction]
pub fn create_skeleton_from_source(source: &str, _lang_ext: &str) -> PyResult<String> {
    Ok(source.to_string())
}

#[pyclass]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum SupportedLanguage {
    Python,
    Rust,
    JavaScript,
    JavaScriptJsx,
    TypeScript,
    TypeScriptTsx,
    Go,
    Unknown,
}

impl SupportedLanguage {
    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|s| s.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "py" => Self::Python,
            "rs" => Self::Rust,
            "js" | "mjs" | "cjs" => Self::JavaScript,
            "jsx" => Self::JavaScriptJsx,
            "ts" => Self::TypeScript,
            "tsx" => Self::TypeScriptTsx,
            "go" => Self::Go,
            _ => Self::Unknown,
        }
    }

    pub fn all() -> &'static [SupportedLanguage] {
        &[
            SupportedLanguage::Python,
            SupportedLanguage::Rust,
            SupportedLanguage::JavaScript,
            SupportedLanguage::JavaScriptJsx,
            SupportedLanguage::TypeScript,
            SupportedLanguage::TypeScriptTsx,
            SupportedLanguage::Go,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Interface,
    Type,
    Enum,
    Variable,
}

/// Represents a symbol (definition or usage) found in source code.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub start_byte: usize,
    pub end_byte: usize,
    pub is_definition: bool,
}
