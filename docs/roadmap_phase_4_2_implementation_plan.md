# Phase 4.2 Implementation Plan: Tree-sitter Incremental Parsing

## Overview

**Goal:** Implement a high-performance incremental parsing service that can update a file's syntax tree in microseconds, rather than re-parsing the entire file from scratch. This is the core engine that will enable sub-second responses to code changes.

**Parent Phase:** Phase 4: Incremental Updates

**Success Criteria:**
- ✅ Incremental parse times should be 10-100x faster than a full re-parse for typical, localized code changes.
- ✅ Target update time < 5ms for a single edit in a large file.
- ✅ Correctly handles single-line, multi-line, insertion, and deletion edits.
- ✅ The system can maintain a cache of parsed trees to facilitate incremental updates.
- ✅ A robust test suite verifies correctness and performance gains.
- ✅ Exposes a clean API to the Python layer for updating files.

**Estimated Time:** 3-4 hours

---

## Technical Approach

We will leverage Tree-sitter's built-in incremental parsing capabilities. The process involves:
1.  Keeping the previous version of a file's source code and its corresponding `Tree` object in memory.
2.  When a change occurs, we describe that change to Tree-sitter using an `InputEdit` struct, which details the byte offsets and position changes.
3.  We apply this `InputEdit` to the old `Tree`.
4.  We then ask Tree-sitter to re-parse the *new* source code, providing the edited old tree as a reference. Tree-sitter will re-use the unchanged portions of the old tree, only re-parsing the sections that were actually affected by the edit.

This avoids the expensive process of lexing and parsing an entire file from top to bottom on every keystroke.

---

## Step 1: Define Core Data Structures

**File:** Create `rust_core/src/incremental_parser.rs`
**Time:** 20 minutes

This new module will encapsulate all logic for incremental parsing.

### 1.1: The `TextEdit` Struct

This struct will be our language-agnostic way of representing a code change, designed to be easily created from Python.

```rust
// in rust_core/src/incremental_parser.rs

use pyo3::prelude::*;
use tree_sitter::{Point, InputEdit};

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
    fn new(start_line: usize, start_col: usize, end_line: usize, end_col: usize, old_text: String, new_text: String) -> Self {
        Self { start_line, start_col, end_line, end_col, old_text, new_text }
    }
}
```

### 1.2: The `IncrementalParser` Struct

This struct will manage the cache of trees and sources, and provide the main `update_file` method.

```rust
// in rust_core/src/incremental_parser.rs

use crate::parser::{ParserPool, SupportedLanguage};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tree_sitter::{Parser, Tree};

#[pyclass]
pub struct IncrementalParser {
    // A pool to get language-specific parsers.
    parser_pool: ParserPool,
    
    // Cache of most recent trees for each file.
    trees: HashMap<PathBuf, Tree>,
    
    // Cache of the source code corresponding to each tree.
    sources: HashMap<PathBuf, String>,
}
```

---

## Step 2: Implement the `IncrementalParser`

**File:** `rust_core/src/incremental_parser.rs`
**Time:** 60 minutes

### 2.1: `IncrementalParser::new()`

The constructor will initialize the parser pool and the empty caches.

```rust
#[pymethods]
impl IncrementalParser {
    #[new]
    pub fn new() -> Self {
        Self {
            parser_pool: ParserPool::new(),
            trees: HashMap::new(),
            sources: HashMap::new(),
        }
    }
    // ... other methods will go here
}
```

### 2.2: Implement `text_edit_to_input_edit`

This is the trickiest part. It converts our user-friendly `TextEdit` into the byte-based `InputEdit` that Tree-sitter needs. An incorrect implementation here will cause Tree-sitter to ignore the old tree and do a full re-parse.

```rust
// Add as a private method on IncrementalParser
impl IncrementalParser {
    fn position_to_byte(source: &str, line: usize, col: usize) -> usize {
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

    fn calculate_new_end_position(start_line: usize, start_col: usize, new_text: &str) -> Point {
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
    
    fn text_edit_to_input_edit(&self, edit: &TextEdit, old_source: &str) -> InputEdit {
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
            new_end_position: Self::calculate_new_end_position(edit.start_line, edit.start_col, &edit.new_text),
        }
    }
}
```

### 2.3: Implement `update_file`

This method orchestrates the incremental parse.

```rust
// In #[pymethods] block for IncrementalParser

    #[pyo3(name = "update_file")]
    pub fn py_update_file(&mut self, path_str: String, new_source: String, edit: &TextEdit) -> PyResult<()> {
        let path = PathBuf::from(path_str);
        self.update_file(&path, new_source, edit).map_err(|e| 
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)
        )
    }

// In impl block for IncrementalParser
impl IncrementalParser {
    pub fn update_file(&mut self, path: &Path, new_source: String, edit: &TextEdit) -> Result<(), String> {
        let lang = SupportedLanguage::from_path(path);
        let parser = self.parser_pool.get(lang).ok_or("Unsupported language")?;
        
        let old_tree = self.trees.get(path);
        let mut new_tree = None;

        if let Some(tree) = old_tree {
            let old_source = self.sources.get(path).ok_or("Source cache missing")?;
            let mut tree_clone = tree.clone();
            let input_edit = self.text_edit_to_input_edit(edit, old_source);
            tree_clone.edit(&input_edit);
            
            new_tree = parser.parse(&new_source, Some(&tree_clone));
        } else {
            // No cached tree, parse from scratch
            new_tree = parser.parse(&new_source, None);
        }

        if let Some(parsed_tree) = new_tree {
            self.trees.insert(path.to_path_buf(), parsed_tree);
            self.sources.insert(path.to_path_buf(), new_source);
            Ok(())
        } else {
            Err("Failed to parse file".to_string())
        }
    }
}
```

**Note:** The `ParserPool` from Phase 1 will need to be updated to expose a `get` method that returns a mutable parser.

---

## Step 3: Integrate with Main Library

**File:** `rust_core/src/lib.rs`
**Time:** 10 minutes

Expose the new `IncrementalParser` and `TextEdit` classes.

```rust
// In rust_core/src/lib.rs

pub mod incremental_parser;

// In #[pymodule] function
fn semantic_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    // ... existing classes
    m.add_class::<incremental_parser::IncrementalParser>()?;
    m.add_class::<incremental_parser::TextEdit>()?;
    Ok(())
}
```

---

## Step 4: Testing

**File:** Create `rust_core/tests/test_incremental_parsing.rs`
**Time:** 60 minutes

### 4.1: Unit Test for `text_edit_to_input_edit`

Verify that byte offsets and new end positions are calculated correctly for various scenarios.

```rust
#[test]
fn test_edit_conversion_single_line_insertion() {
    let source = "hello world";
    let edit = TextEdit {
        start_line: 0, start_col: 6,
        end_line: 0, end_col: 6,
        old_text: "".to_string(),
        new_text: "rust ".to_string(),
    };
    
    let parser = IncrementalParser::new();
    let input_edit = parser.text_edit_to_input_edit(&edit, source);

    assert_eq!(input_edit.start_byte, 6);
    assert_eq!(input_edit.old_end_byte, 6);
    assert_eq!(input_edit.new_end_byte, 11); // 6 + "rust ".len()
    assert_eq!(input_edit.new_end_position, Point::new(0, 11));
}

// ... add more tests for deletion, replacement, multi-line edits etc.
```

### 4.2: Integration Test for `update_file`

Verify that an incremental parse produces a valid tree that reflects the edit.

```rust
#[test]
fn test_incremental_update_produces_valid_tree() {
    let mut parser = IncrementalParser::new();
    let path = PathBuf::from("test.py");
    let source1 = "x = 1\ny = 2\n";
    let source2 = "x = 100\ny = 2\n";

    // 1. Initial parse
    parser.update_file(&path, source1.to_string(), &TextEdit::new_empty()).unwrap();

    // 2. Create an edit
    let edit = TextEdit {
        start_line: 0, start_col: 4,
        end_line: 0, end_col: 5,
        old_text: "1".to_string(),
        new_text: "100".to_string(),
    };

    // 3. Incremental update
    parser.update_file(&path, source2.to_string(), &edit).unwrap();

    // 4. Verify the new tree
    let new_tree = parser.trees.get(&path).unwrap();
    // Check for syntax errors
    assert!(!new_tree.root_node().has_error()); 
    // Check that the change is reflected
    let number_node = new_tree.root_node().descendant_for_byte_range(6, 7).unwrap();
    assert_eq!(number_node.utf8_text(source2.as_bytes()).unwrap(), "100");
}
```

### 4.3: Benchmark Test

Implement the benchmark from the roadmap to prove the performance benefits.

**File:** `rust_core/benches/incremental_benchmark.rs`

```rust
use criterion::{criterion_group, criterion_main, Bencher, Criterion};
// ... imports

fn bench_incremental_vs_full_parse(c: &mut Criterion) {
    let large_file = std::fs::read_to_string("path/to/large_file.py").unwrap();
    let mut parser = IncrementalParser::new();
    let path = PathBuf::from("large_file.py");

    // Initial parse
    parser.update_file(&path, large_file.clone(), &TextEdit::new_empty()).unwrap();

    c.bench_function("full parse", |b| b.iter(|| {
        parser.parser_pool.get(SupportedLanguage::Python).unwrap().parse(&large_file, None);
    }));

    let mut new_source = large_file.clone();
    new_source.insert_str(1000, " # a comment");
    let edit = TextEdit {
        start_line: /* calculate line */, start_col: /* calculate col */,
        end_line: /* ... */, end_col: /* ... */,
        old_text: "".to_string(),
        new_text: " # a comment".to_string(),
    };

    c.bench_function("incremental parse", |b| b.iter(|| {
        parser.update_file(&path, new_source.clone(), &edit);
    }));
}

criterion_group!(benches, bench_incremental_vs_full_parse);
criterion_main!(benches);
```
*(Note: requires adding `criterion` as a dev-dependency and configuring `Cargo.toml`)*

---

## Troubleshooting

-   **Issue: Incremental parse is slow (same as full parse).**
    -   **Cause:** The `InputEdit` struct is likely incorrect. Tree-sitter is silently rejecting the old tree and parsing from scratch.
    -   **Solution:** Triple-check the byte offset and `Point` calculations in `text_edit_to_input_edit`. Add extensive logging to compare the generated `InputEdit` with what's expected.

-   **Issue: `has_error()` returns true after an incremental parse.**
    -   **Cause:** The edit has resulted in invalid syntax, and the tree now contains `ERROR` nodes.
    -   **Solution:** This is expected behavior. The system needs to handle this gracefully, potentially by flagging the file as "broken" and falling back to a full parse on the next edit until the error is resolved.

-   **Issue: Memory usage grows over time.**
    -   **Cause:** The `trees` and `sources` `HashMap`s are growing without bounds.
    -   **Solution:** For now, this is acceptable for a proof-of-concept. The full solution is an LRU cache (slated for Phase 6) to evict the least-recently-used trees from memory.

---

## Next Phase

**Phase 4.3**: Graph Update Propagation
-   Use the new, efficiently parsed trees to re-extract symbols and imports.
-   Compare the new symbols/imports with the old ones.
-   Update the `RepoGraph` by adding or removing nodes and edges.
-   Trigger a lazy PageRank recalculation if the graph structure has changed.

---

## Implementation Notes
*   **Line Endings:** The proposed `position_to_byte` function relies on `source.lines()` and assumes LF (`\n`) line endings when calculating byte offsets. This may be inaccurate for files with CRLF (`\r\n`) line endings, which could cause incremental parsing to fail silently and fall back to a full re-parse. A more robust implementation should be considered if this becomes an issue.

```