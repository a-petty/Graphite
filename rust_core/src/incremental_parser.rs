use std::path::{Path, PathBuf};
use lru::LruCache;
use std::num::NonZeroUsize;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use tree_sitter::{InputEdit, Point, Tree};

use crate::parser::{ParserPool, SupportedLanguage};

/// Represents a single text edit operation.
/// All coordinates are 0-based.
#[pyclass]
#[derive(Debug, Clone)]
pub struct TextEdit {
    #[pyo3(get, set)]
    pub start_line: usize,
    #[pyo3(get, set)]
    pub start_col: usize,
    #[pyo3(get, set)]
    pub end_line: usize,
    #[pyo3(get, set)]
    pub end_col: usize,
    #[pyo3(get, set)]
    pub old_text: String,
    #[pyo3(get, set)]
    pub new_text: String,
}

#[pymethods]
impl TextEdit {
    #[new]
    pub fn new(
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
        old_text: String,
        new_text: String,
    ) -> Self {
        Self {
            start_line,
            start_col,
            end_line,
            end_col,
            old_text,
            new_text,
        }
    }
}

#[pyclass]
pub struct IncrementalParser {
    // A pool to get language-specific parsers.
    parser_pool: ParserPool,

    // Cache of most recent trees for each file.
    trees: LruCache<PathBuf, Tree>,

    // Cache of the source code corresponding to each tree.
    sources: LruCache<PathBuf, String>,
}

impl Clone for IncrementalParser {
    fn clone(&self) -> Self {
        Self {
            parser_pool: ParserPool::new_py(),
            trees: self.trees.clone(),
            sources: self.sources.clone(),
        }
    }
}

#[pymethods]
impl IncrementalParser {
    #[new]
    pub fn new() -> Self {
        let cache_size = NonZeroUsize::new(100).unwrap();
        Self {
            parser_pool: ParserPool::new_py(),
            trees: LruCache::new(cache_size),
            sources: LruCache::new(cache_size),
        }
    }

    #[pyo3(name = "update_file")]
    pub fn py_update_file(&mut self, path_str: String, new_source: String, edit: &TextEdit) -> PyResult<()> {
        let path = PathBuf::from(path_str);
        self.update_file(&path, new_source, edit)
            .map_err(|e| PyRuntimeError::new_err(e))
    }
}

impl IncrementalParser {
    pub fn position_to_byte(source: &str, line: usize, col: usize) -> usize {
        let mut byte_offset = 0;
        for (i, l) in source.lines().enumerate() {
            if i == line {
                byte_offset += l.chars().take(col).map(|c| c.len_utf8()).sum::<usize>();
                break;
            }
            byte_offset += l.len() + 1; // +1 for the newline
        }
        byte_offset
    }

    pub(crate) fn calculate_new_end_position(start_line: usize, start_col: usize, new_text: &str) -> Point {
        let mut new_end_line = start_line;
        let mut new_end_col = start_col;
        for c in new_text.chars() {
            if c == '\n' {
                new_end_line += 1;
                new_end_col = 0;
            } else {
                new_end_col += 1;
            }
        }
        Point::new(new_end_line, new_end_col)
    }

    pub fn text_edit_to_input_edit(edit: &TextEdit, old_source: &str) -> InputEdit {
        let start_byte = Self::position_to_byte(old_source, edit.start_line, edit.start_col);
        let old_end_byte = start_byte + edit.old_text.len();
        let new_end_byte = start_byte + edit.new_text.len();

        let start_position = Point::new(edit.start_line, edit.start_col);

        InputEdit {
            start_byte,
            old_end_byte,
            new_end_byte,
            start_position,
            old_end_position: Point::new(edit.end_line, edit.end_col),
            new_end_position: Self::calculate_new_end_position(
                edit.start_line,
                edit.start_col,
                &edit.new_text,
            ),
        }
    }

    pub fn parse(&mut self, source: &str, language: SupportedLanguage) -> Option<Tree> {
        if language == SupportedLanguage::Unknown {
            return None;
        }

        let parser = self.parser_pool.get(language).expect("Failed to get parser for language");
        parser.parse(source, None)
    }

    pub fn update_file(&mut self, path: &Path, new_source: String, edit: &TextEdit) -> Result<(), String> {
        let lang = SupportedLanguage::from_path(path);
        if lang == SupportedLanguage::Unknown {
            return Err(format!("Unsupported language for file: {:?}", path));
        }

        let parser = self.parser_pool.get(lang).ok_or("Unsupported language")?;

        let old_tree = self.trees.get(path);
        let mut new_tree = None;

        if let Some(tree) = old_tree {
            let old_source = self.sources.get(path).ok_or("Source cache missing")?;
            let mut tree_clone = tree.clone();
            let input_edit = Self::text_edit_to_input_edit(edit, old_source);
            tree_clone.edit(&input_edit);

            new_tree = parser.parse(&new_source, Some(&tree_clone));
        } else {
            // No cached tree, parse from scratch
            new_tree = parser.parse(&new_source, None);
        }

        if let Some(parsed_tree) = new_tree {
            self.trees.put(path.to_path_buf(), parsed_tree);
            self.sources.put(path.to_path_buf(), new_source);
            Ok(())
        } else {
            Err("Failed to parse file".to_string())
        }
    }

    pub fn get_tree(&mut self, path: &Path) -> Option<&Tree> {
        self.trees.get(path)
    }
}
