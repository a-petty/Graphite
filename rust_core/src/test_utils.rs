use tree_sitter;
use std::io::Write;

pub fn print_tree(node: tree_sitter::Node, source: &str, depth: usize) {
    let indent = "  ".repeat(depth);
    let node_text = node.utf8_text(source.as_bytes()).unwrap_or("");
    let preview = if node_text.len() > 50 {
        format!("{}...", &node_text[..50])
    } else {
        node_text.to_string()
    };
    
    let is_named = node.is_named();
    let name_suffix = if is_named { "" } else { " (ANON)" };

    println!("{}({}){}: {:?}", indent, node.kind(), name_suffix, preview.replace('\n', "\\n"));
    
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if let Some(field_name) = node.field_name_for_child(i as u32) { // Cast i to u32
                print!("{}  {}: ", indent, field_name);
            } else {
                print!("{}  - ", indent); // No field name
            }
            print_tree(child, source, depth + 1);
        }
    }
}

pub fn dump_tree_to_file(tree: &tree_sitter::Tree, source: &str, filename: &str) {
    use std::fs::File;
    
    let mut file = File::create(filename).unwrap();
    dump_node(&mut file, tree.root_node(), source, 0);
}

fn dump_node(file: &mut std::fs::File, node: tree_sitter::Node, source: &str, depth: usize) {
    
    let indent = "  ".repeat(depth);
    let node_text = node.utf8_text(source.as_bytes()).unwrap_or("");
    let preview = if node_text.len() > 50 {
        format!("{}...", &node_text[..50])
    } else {
        node_text.to_string()
    };
    
    let is_named = node.is_named();
    let name_suffix = if is_named { "" } else { " (ANON)" };
    
    writeln!(file, "{}({}){}: {:?}", indent, node.kind(), name_suffix, preview.replace('\n', "\\n")).unwrap();
    
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if let Some(field_name) = node.field_name_for_child(i as u32) { // Cast i to u32
                write!(file, "{}  {}: ", indent, field_name).unwrap();
            } else {
                write!(file, "{}  - ", indent).unwrap(); // No field name
            }
            dump_node(file, child, source, depth + 1);
        }
    }
}
