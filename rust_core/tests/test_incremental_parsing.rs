use std::path::PathBuf;

use semantic_engine::incremental_parser::{IncrementalParser, TextEdit};
use semantic_engine::parser::SupportedLanguage;

// Helper for initial parse without an edit
fn new_empty_text_edit() -> TextEdit {
    TextEdit::new(
        0, 0, 0, 0,
        "".to_string(),
        "".to_string(),
    )
}

#[test]
fn test_edit_conversion_single_line_insertion() {
    let source = "hello world";
    let edit = TextEdit::new(
        0, 6, 0, 6,
        "".to_string(),
        "rust ".to_string(),
    );
    
    let input_edit = IncrementalParser::text_edit_to_input_edit(&edit, source);

    assert_eq!(input_edit.start_byte, 6);
    assert_eq!(input_edit.old_end_byte, 6);
    assert_eq!(input_edit.new_end_byte, 11); // 6 + "rust ".len()
    assert_eq!(input_edit.new_end_position, tree_sitter::Point::new(0, 11));
}

// TODO: Add more unit tests for TextEdit conversion (deletion, replacement, multi-line, etc.)

#[test]
fn test_incremental_update_produces_valid_tree() {
    let mut parser = IncrementalParser::new();
    let path = PathBuf::from("test.py");
    let source1 = "x = 1\ny = 2\n";
    let source2 = "x = 100\ny = 2\n";

    // 1. Initial parse
    parser.update_file(&path, source1.to_string(), &new_empty_text_edit()).unwrap();

    // 2. Create an edit
    let edit = TextEdit::new(
        0, 4, 0, 5,
        "1".to_string(),
        "100".to_string(),
    );

    // 3. Incremental update
    parser.update_file(&path, source2.to_string(), &edit).unwrap();

    // 4. Verify the new tree
    let new_tree = parser.get_tree(&path).unwrap();
    // Check for syntax errors
    assert!(!new_tree.root_node().has_error()); 
    // Check that the change is reflected
    let number_node = new_tree.root_node().descendant_for_byte_range(4, 7).unwrap(); // '100' is at byte range 4-7
    assert_eq!(number_node.utf8_text(source2.as_bytes()).unwrap(), "100");
    assert_eq!(number_node.kind(), "integer");
}

// TODO: Add more integration tests for various edit types.
