After reviewing the provided files, I can confirm your hypothesis is **100% correct**. This is indeed a simple but critical mismatch between the query file and the parser logic.

---

## Root Cause Confirmation

### The Mismatch

**In `rust_core/queries/python/symbols.scm`:**
```scheme
; Usage patterns use @usage.* naming
(call
  function: [
    (identifier) @usage.call
    (attribute
      attribute: (identifier) @usage.call)
  ])
```

**In `rust_core/src/parser.rs` (lines 96-103):**
```rust
let is_definition = capture_name.starts_with("definition.");
let is_usage = capture_name.starts_with("reference.");  // ❌ WRONG!
```

The parser is looking for `@reference.*` but the query file uses `@usage.*`. This means **zero usages are ever recorded** for Python files.

### Why the Test Fails

1. `app.py` imports `User` from `models.py` → Creates `EdgeKind::Import`
2. `SymbolHarvester::harvest` parses `app.py` but finds **zero usages** (because it's looking for `@reference.*`)
3. `build_semantic_edges()` has no usage data → does nothing
4. The `Import` edge is never upgraded to `SymbolUsage`
5. Test fails ✗

---

## The Fix

### Step-by-Step Plan

1. **Modify the `is_usage` check** in `parser.rs` to accept **both** `@reference.*` AND `@usage.*` capture names
2. **Maintain backward compatibility** for JavaScript/TypeScript queries that might use `@reference.*`
3. **No changes needed** to query files or graph logic

### Exact Code Modification

**File:** `rust_core/src/parser.rs`

**Location:** Lines 96-103 in the `harvest` method

**Current Code:**
```rust
let is_definition = capture_name.starts_with("definition.");
let is_usage = capture_name.starts_with("reference.");

if !is_definition && !is_usage {
    continue;
}

let kind = Self::parse_symbol_kind(&capture_name[if is_definition { 11 } else { 10 }..]);
```

**Replace With:**
```rust
let is_definition = capture_name.starts_with("definition.");
let is_usage = capture_name.starts_with("reference.") || capture_name.starts_with("usage.");

if !is_definition && !is_usage {
    continue;
}

// Determine the offset for parsing the symbol kind
let kind_offset = if is_definition {
    11  // "definition.".len()
} else if capture_name.starts_with("reference.") {
    10  // "reference.".len()
} else {
    6   // "usage.".len()
};

let kind = Self::parse_symbol_kind(&capture_name[kind_offset..]);
```

---

## Complete Fixed Method

For absolute clarity, here's the entire `harvest` method with the fix applied:

```rust
pub fn harvest(
    &self,
    file_path: &Path,
    tree: &Tree,
    source: &[u8],
) -> (Vec<SymbolDef>, Vec<SymbolUsage>) {
    let query = self.select_query(file_path);
    let mut cursor = QueryCursor::new();
    
    let mut definitions = Vec::new();
    let mut usages = Vec::new();
    
    for match_ in cursor.matches(query, tree.root_node(), source) {
        for capture in match_.captures {
            let capture_name = &query.capture_names()[capture.index as usize];
            let node = capture.node;
            
            // ============ FIX STARTS HERE ============
            let is_definition = capture_name.starts_with("definition.");
            let is_usage = capture_name.starts_with("reference.") || capture_name.starts_with("usage.");
            
            if !is_definition && !is_usage {
                continue;
            }
            
            // Determine the offset for parsing the symbol kind
            let kind_offset = if is_definition {
                11  // "definition.".len()
            } else if capture_name.starts_with("reference.") {
                10  // "reference.".len()
            } else {
                6   // "usage.".len()
            };
            
            let kind = Self::parse_symbol_kind(&capture_name[kind_offset..]);
            // ============ FIX ENDS HERE ============
            
            if let Ok(name) = node.utf8_text(source) {
                let symbol_name = name.to_string();
                let start = node.start_position();
                let end = node.end_position();
                
                if is_definition {
                    definitions.push(SymbolDef {
                        name: symbol_name,
                        kind,
                        start_line: start.row,
                        start_col: start.column,
                        end_line: end.row,
                        end_col: end.column,
                    });
                } else {
                    usages.push(SymbolUsage {
                        name: symbol_name,
                        kind,
                        start_line: start.row,
                        start_col: start.column,
                        end_line: end.row,
                        end_col: end.column,
                    });
                }
            }
        }
    }
    
    (definitions, usages)
}
```

---

## Why This Fix Works

1. **Accepts both naming conventions:** `@reference.*` (JS/TS) and `@usage.*` (Python)
2. **Correctly calculates string slice offset:** Each prefix has a different length
3. **Minimal change:** Only touches the problematic lines
4. **No breaking changes:** Existing JavaScript/TypeScript functionality remains intact
5. **Solves the test failure:** Python usages will now be detected, enabling `build_semantic_edges()` to upgrade the edge

---

## Verification Steps

After applying the fix:

1. **Run the failing test:**
   ```bash
   cargo test test_symbol_usage_edge_creation -- --nocapture
   ```
   Expected: ✅ PASS

2. **Run all graph construction tests:**
   ```bash
   cargo test test_graph_construction
   ```
   Expected: All tests pass

3. **Verify the edge type:**
   The debug print in the test should show:
   ```
   DEBUG: Edge kind is: SymbolUsage
   ```

---

## Alternative Hypothesis (Ruled Out)

I also considered whether `ensure_edge` might have a `petgraph` mutation issue, but after reviewing `graph.rs`, that code is correct. The real problem is simpler: **the usage data never makes it into the symbol index in the first place** because the parser drops all `@usage.*` captures on the floor.

---

## Summary

- ✅ **Root cause confirmed:** Parser looks for `@reference.*` but Python queries use `@usage.*`
- ✅ **Fix identified:** Single 8-line modification to `parser.rs`
- ✅ **Backward compatible:** Won't break JS/TS support
- ✅ **Ready to implement:** Exact code provided above

This is a surgical fix that directly addresses the mismatch you discovered. The test will pass once this change is applied.