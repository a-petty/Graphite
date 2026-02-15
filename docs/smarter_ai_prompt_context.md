# Comprehensive Debugging Context for the `SymbolUsage` Edge Failure

## 1. The Goal & The Test Case

The objective is to create a "smart" code graph that distinguishes between a simple `import` and an actual `SymbolUsage`. The test `test_symbol_usage_edge_creation` in `rust_core/tests/test_graph_construction.rs` is designed to validate this.

**Test Setup:**
- **`models.py`**: Defines a class `class User: ...`
- **`app.py`**: Imports and instantiates the class: `from models import User` followed by `user = User()`.

**Expected Outcome:**
The graph should contain a single, strong **`EdgeKind::SymbolUsage`** edge from the `app.py` node to the `models.py` node.

**Actual Outcome:**
The test fails because the edge is of kind **`EdgeKind::Import`**.

## 2. The Build Process & Debugging Timeline

Based on extensive logging, I have traced the exact sequence of operations during the test run.

### Step A: `build()` Function - Initial Population

1.  **Parsing & Symbol Harvest**: The system correctly parses both `app.py` and `models.py`.
2.  **Symbol Index Populated**: The `SymbolIndex` is correctly populated.
    - `definitions`: Knows that the symbol `User` is defined in `models.py`.
    - `usages`: Knows that the symbol `User` is used in `app.py`.
3.  **Import Edge Creation**: The `build` function iterates through the parsed files and creates the initial edges based *only* on import statements.
    - A weak `EdgeKind::Import` is created from `app.py` to `models.py`.
    - **Log Confirmation**: `  - Adding import edge from "app.py" to "models.py"`

At this point, the graph is valid but semantically incomplete, which is expected.

### Step B: `build_semantic_edges()` Function - The Upgrade Attempt

1.  **Semantic Analysis**: This function runs *after* the initial `build()` is complete. Its purpose is to upgrade the weak `Import` edges.
2.  **Logic**: It queries the `SymbolIndex`, sees that `app.py` uses a symbol (`User`) defined in `models.py`, and correctly decides to create a `SymbolUsage` edge.
3.  **Upgrade Attempt**: It calls the `ensure_edge` function to perform this upgrade.
    - **Log Confirmation**: `  - Adding symbol usage edge from "app.py" to "models.py"`

### Step C: The Failure Point inside `ensure_edge`

This is the core of the mystery. The `ensure_edge` function is explicitly designed to handle this upgrade.

```rust
// in rust_core/src/graph.rs

fn ensure_edge(&mut self, source: NodeIndex, target: NodeIndex, kind: EdgeKind) -> bool {
    // 1. It finds the edge successfully.
    if let Some(edge_id) = self.graph.find_edge(source, target) {
        
        // 2. It gets a mutable reference to the existing edge's kind.
        let existing_kind = &mut self.graph[edge_id]; 
        
        // 3. This condition should evaluate to TRUE.
        //    - *existing_kind is `Import`
        //    - kind (the parameter) is `SymbolUsage`
        if *existing_kind == EdgeKind::Import && kind == EdgeKind::SymbolUsage {
            
            // 4. This line should update the edge kind in the graph.
            *existing_kind = EdgeKind::SymbolUsage; 
            return true; // It reports that it upgraded the edge.
        }
        return false; 
    } else {
        // This branch is not taken, as the import edge already exists.
        self.graph.add_edge(source, target, kind);
        return true;
    }
}
```

Despite the logic appearing flawless and the logs confirming the correct sequence of events, the update `*existing_kind = EdgeKind::SymbolUsage;` is **failing silently**. When the test inspects the edge after this process, its kind is still `Import`.

## 3. Current Hypothesis

The logic within `ensure_edge` is sound, but the state change is not persisting. This could be due to a subtle ownership or borrowing issue, a misunderstanding of `petgraph`'s behavior, or a race condition that isn't immediately apparent. The next logical step is to add verbose logging *inside* `ensure_edge` to trace the value of `existing_kind` before and after the attempted update.
