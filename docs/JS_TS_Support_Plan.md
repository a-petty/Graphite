# Implementation Plan: JavaScript/TypeScript Support for Atlas

## Overview

This plan refactors the Python-only `ImportResolver` into a trait-based system and adds full JavaScript/TypeScript support. The work is divided into 6 major phases executed sequentially.

---

## Phase 1: Refactor to Trait-Based Architecture

### Step 1.1: Define the `ImportResolver` Trait

**File:** `rust_core/src/import_resolver.rs`

1. **Add the trait definition at the top of the file (after imports):**

```rust
/// Common interface for language-specific import resolution.
pub trait ImportResolver: std::fmt::Debug {
    /// Resolves an import/require statement to an absolute file path.
    /// Returns None if the import cannot be resolved or should be filtered out.
    fn resolve<'a>(
        &self,
        import_node: Node<'a>,
        current_file: &Path,
        source: &'a [u8],
    ) -> Option<PathBuf>;
    
    /// Returns the Tree-sitter language for this resolver.
    fn language(&self) -> tree_sitter::Language;
    
    /// Returns the file extensions this resolver handles.
    fn file_extensions(&self) -> &[&str];
}
```

### Step 1.2: Rename and Refactor Existing Struct

**File:** `rust_core/src/import_resolver.rs`

1. **Rename `ImportResolver` struct to `PythonImportResolver`:**
   - Change `pub struct ImportResolver` to `pub struct PythonImportResolver`
   - Update the implementation block: `impl ImportResolver` → `impl PythonImportResolver`

2. **Implement the `ImportResolver` trait for `PythonImportResolver`:**

```rust
impl ImportResolver for PythonImportResolver {
    fn resolve<'a>(
        &self,
        import_node: Node<'a>,
        current_file: &Path,
        source: &'a [u8],
    ) -> Option<PathBuf> {
        // Move the existing resolve logic here (keep it unchanged)
        let mut query_cursor = QueryCursor::new();
        // ... (existing code)
    }
    
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_python::language()
    }
    
    fn file_extensions(&self) -> &[&str] {
        &["py", "pyi"]
    }
}
```

3. **Keep the private helper methods** (`resolve_absolute`, `resolve_relative`) as-is on `PythonImportResolver`.

### Step 1.3: Update Graph to Use Trait Object

**File:** `rust_core/src/graph.rs`

1. **Locate the `RepoGraph` struct definition and modify it:**

```rust
pub struct RepoGraph {
    pub graph: Graph<NodeData, EdgeKind, Directed>,
    pub path_to_node: HashMap<PathBuf, NodeIndex>,
    pub project_root: PathBuf,
    // Change from concrete type to trait object
    import_resolver: Box<dyn ImportResolver>,
    symbol_harvester: SymbolHarvester,
    // ... other fields
}
```

2. **Update the `new()` constructor:**

```rust
impl RepoGraph {
    pub fn new(project_root: PathBuf, language: &str) -> Self {
        let import_resolver: Box<dyn ImportResolver> = match language.to_lowercase().as_str() {
            "python" => Box::new(PythonImportResolver::new(&project_root)),
            "javascript" | "typescript" | "js" | "ts" => {
                Box::new(JsTsImportResolver::new(&project_root))
            }
            _ => panic!("Unsupported language: {}", language),
        };
        
        Self {
            graph: Graph::new(),
            path_to_node: HashMap::new(),
            project_root: project_root.clone(),
            import_resolver,
            symbol_harvester: SymbolHarvester::new(),
            // ... initialize other fields
        }
    }
}
```

3. **Update all usages of `self.import_resolver`** in methods like `process_file_imports` to use the trait methods.

---

## Phase 2: Create JavaScript/TypeScript Tree-sitter Queries

### Step 2.1: Create JavaScript Symbol Query

**File:** `rust_core/queries/javascript/symbols.scm` (new file)

```scheme
; Function declarations
(function_declaration
  name: (identifier) @definition.function)

; Arrow functions assigned to variables
(variable_declarator
  name: (identifier) @definition.function
  value: (arrow_function))

; Class declarations
(class_declaration
  name: (identifier) @definition.class)

; Method definitions
(method_definition
  name: (property_identifier) @definition.method)

; Variable declarations (const, let, var)
(variable_declarator
  name: (identifier) @definition.variable)

; Function calls
(call_expression
  function: [
    (identifier) @reference.call
    (member_expression property: (property_identifier) @reference.call)
  ])

; New expressions
(new_expression
  constructor: (identifier) @reference.class)

; Identifiers (general usage)
(identifier) @reference.identifier
```

### Step 2.2: Create TypeScript Symbol Query

**File:** `rust_core/queries/typescript/symbols.scm` (new file)

```scheme
; Include all JavaScript patterns
(function_declaration
  name: (identifier) @definition.function)

(variable_declarator
  name: (identifier) @definition.function
  value: (arrow_function))

(class_declaration
  name: (type_identifier) @definition.class)

(method_definition
  name: (property_identifier) @definition.method)

(variable_declarator
  name: (identifier) @definition.variable)

; TypeScript-specific: Interface declarations
(interface_declaration
  name: (type_identifier) @definition.interface)

; TypeScript-specific: Type alias
(type_alias_declaration
  name: (type_identifier) @definition.type)

; TypeScript-specific: Enum declarations
(enum_declaration
  name: (identifier) @definition.enum)

; Function calls
(call_expression
  function: [
    (identifier) @reference.call
    (member_expression property: (property_identifier) @reference.call)
  ])

; New expressions
(new_expression
  constructor: (identifier) @reference.class)

(identifier) @reference.identifier
```

### Step 2.3: Create JavaScript Import/Export Query

**File:** `rust_core/queries/javascript/imports.scm` (new file)

```scheme
; ES6 import statements
(import_statement
  source: (string) @import.source)

; CommonJS require
(call_expression
  function: (identifier) @_require (#eq? @_require "require")
  arguments: (arguments (string) @import.source))

; Dynamic imports
(call_expression
  function: (import) @_import
  arguments: (arguments (string) @import.source))
```

### Step 2.4: Create TypeScript Import Query

**File:** `rust_core/queries/typescript/imports.scm` (new file)

Copy the JavaScript imports query (TypeScript uses same import syntax):

```scheme
; Same as javascript/imports.scm
(import_statement
  source: (string) @import.source)

(call_expression
  function: (identifier) @_require (#eq? @_require "require")
  arguments: (arguments (string) @import.source))

(call_expression
  function: (import) @_import
  arguments: (arguments (string) @import.source))
```

---

## Phase 3: Implement `JsTsImportResolver`

### Step 3.1: Add Dependencies

**File:** `rust_core/Cargo.toml`

Add these dependencies:

```toml
serde_json = "1.0"  # For parsing tsconfig.json
glob = "0.3"        # For glob pattern matching in tsconfig paths
```

### Step 3.2: Create Node.js Built-ins List

**File:** `rust_core/src/import_resolver.rs`

Add to the `lazy_static!` block:

```rust
lazy_static! {
    // ... existing Python sets ...
    
    static ref NODE_BUILTIN_MODULES: HashSet<&'static str> = [
        "assert", "buffer", "child_process", "cluster", "crypto", "dgram",
        "dns", "domain", "events", "fs", "http", "https", "net", "os",
        "path", "punycode", "querystring", "readline", "stream", "string_decoder",
        "timers", "tls", "tty", "url", "util", "v8", "vm", "zlib",
        // Node 12+ built-ins
        "worker_threads", "perf_hooks", "async_hooks", "inspector",
    ].iter().cloned().collect();
}
```

### Step 3.3: Define Path Alias Config Struct

**File:** `rust_core/src/import_resolver.rs`

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TsConfig {
    #[serde(rename = "compilerOptions")]
    compiler_options: Option<CompilerOptions>,
}

#[derive(Debug, Deserialize)]
struct CompilerOptions {
    #[serde(rename = "baseUrl")]
    base_url: Option<String>,
    paths: Option<HashMap<String, Vec<String>>>,
}
```

### Step 3.4: Implement `JsTsImportResolver` Struct

**File:** `rust_core/src/import_resolver.rs`

```rust
#[derive(Debug)]
pub struct JsTsImportResolver {
    project_root: PathBuf,
    /// Maps module specifiers to file paths (e.g., "components/Button" -> "src/components/Button.tsx")
    module_index: HashMap<String, PathBuf>,
    /// Compiled Tree-sitter query for import statements
    import_query_js: Query,
    import_query_ts: Query,
    /// Path aliases from tsconfig.json/jsconfig.json
    path_aliases: HashMap<String, Vec<String>>,
    /// Base URL from tsconfig
    base_url: Option<PathBuf>,
}

impl JsTsImportResolver {
    pub fn new(project_root: &Path) -> Self {
        let mut resolver = Self {
            project_root: project_root.to_path_buf(),
            module_index: HashMap::new(),
            import_query_js: Query::new(
                tree_sitter_javascript::language(),
                include_str!("../queries/javascript/imports.scm")
            ).expect("Failed to create JS import query"),
            import_query_ts: Query::new(
                tree_sitter_typescript::language_typescript(),
                include_str!("../queries/typescript/imports.scm")
            ).expect("Failed to create TS import query"),
            path_aliases: HashMap::new(),
            base_url: None,
        };
        
        resolver.load_tsconfig();
        resolver.index_modules();
        resolver
    }
    
    fn load_tsconfig(&mut self) {
        // Try tsconfig.json first, then jsconfig.json
        for config_name in &["tsconfig.json", "jsconfig.json"] {
            let config_path = self.project_root.join(config_name);
            if config_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_json::from_str::<TsConfig>(&content) {
                        if let Some(opts) = config.compiler_options {
                            if let Some(base) = opts.base_url {
                                self.base_url = Some(self.project_root.join(base));
                            }
                            if let Some(paths) = opts.paths {
                                self.path_aliases = paths;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
    
    fn index_modules(&mut self) {
        let extensions = ["js", "jsx", "ts", "tsx", "mjs", "cjs"];
        
        for entry in WalkDir::new(&self.project_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
        {
            let path = entry.path();
            let ext = path.extension().and_then(|s| s.to_str());
            
            if ext.map_or(false, |e| extensions.contains(&e)) {
                if let Ok(rel_path) = path.strip_prefix(&self.project_root) {
                    // Index by path without extension
                    let path_str = rel_path.to_string_lossy();
                    let without_ext = path_str.trim_end_matches(ext.unwrap())
                                              .trim_end_matches('.');
                    self.module_index.insert(without_ext.to_string(), path.to_path_buf());
                    
                    // Also index /index files without the /index part
                    if path.file_stem().and_then(|s| s.to_str()) == Some("index") {
                        if let Some(parent_str) = Path::new(without_ext).parent() {
                            self.module_index.insert(
                                parent_str.to_string_lossy().to_string(),
                                path.to_path_buf()
                            );
                        }
                    }
                }
            }
        }
    }
    
    fn resolve_import_specifier(&self, specifier: &str, current_file: &Path) -> Option<PathBuf> {
        // 1. Filter out Node.js built-ins
        if NODE_BUILTIN_MODULES.contains(specifier) {
            return None;
        }
        
        // 2. Filter out node_modules (third-party packages)
        if !specifier.starts_with('.') && !specifier.starts_with('/') {
            // Check if it's a path alias
            if let Some(resolved) = self.resolve_path_alias(specifier) {
                return Some(resolved);
            }
            // Otherwise it's a node_modules import, ignore it
            return None;
        }
        
        // 3. Resolve relative/absolute imports
        let base_path = if specifier.starts_with('/') {
            self.project_root.clone()
        } else {
            current_file.parent()?.to_path_buf()
        };
        
        let target_path = base_path.join(specifier.trim_start_matches('/'));
        
        // Try with various extensions
        for ext in &["", ".js", ".jsx", ".ts", ".tsx", "/index.js", "/index.jsx", "/index.ts", "/index.tsx"] {
            let candidate = target_path.with_extension("").to_string_lossy().to_string() + ext;
            let candidate_path = PathBuf::from(&candidate);
            
            if candidate_path.exists() {
                return Some(candidate_path);
            }
        }
        
        None
    }
    
    fn resolve_path_alias(&self, specifier: &str) -> Option<PathBuf> {
        for (alias_pattern, alias_paths) in &self.path_aliases {
            // Handle patterns like "@/*" -> ["src/*"]
            if alias_pattern.ends_with("/*") {
                let prefix = alias_pattern.trim_end_matches("/*");
                if specifier.starts_with(prefix) {
                    let remainder = specifier.trim_start_matches(prefix).trim_start_matches('/');
                    
                    for path_pattern in alias_paths {
                        let resolved = path_pattern.replace('*', remainder);
                        let base = self.base_url.as_ref().unwrap_or(&self.project_root);
                        let full_path = base.join(&resolved);
                        
                        // Try to resolve with extensions
                        if let Some(path) = self.try_resolve_with_extensions(&full_path) {
                            return Some(path);
                        }
                    }
                }
            }
        }
        None
    }
    
    fn try_resolve_with_extensions(&self, base: &Path) -> Option<PathBuf> {
        for ext in &["", ".js", ".jsx", ".ts", ".tsx", "/index.js", "/index.jsx", "/index.ts", "/index.tsx"] {
            let candidate = base.to_string_lossy().to_string() + ext;
            let candidate_path = PathBuf::from(&candidate);
            if candidate_path.exists() {
                return Some(candidate_path);
            }
        }
        None
    }
}
```

### Step 3.5: Implement ImportResolver Trait for JsTsImportResolver

**File:** `rust_core/src/import_resolver.rs`

```rust
impl ImportResolver for JsTsImportResolver {
    fn resolve<'a>(
        &self,
        import_node: Node<'a>,
        current_file: &Path,
        source: &'a [u8],
    ) -> Option<PathBuf> {
        // Determine which query to use based on file extension
        let query = if current_file.extension().and_then(|s| s.to_str()) == Some("ts") 
                    || current_file.extension().and_then(|s| s.to_str()) == Some("tsx") {
            &self.import_query_ts
        } else {
            &self.import_query_js
        };
        
        let mut cursor = QueryCursor::new();
        let matches = cursor.matches(query, import_node, source);
        
        for match_ in matches {
            for capture in match_.captures {
                let capture_name = &query.capture_names()[capture.index as usize];
                if capture_name == "import.source" {
                    let import_text = capture.node.utf8_text(source).ok()?;
                    // Remove quotes
                    let specifier = import_text.trim_matches(|c| c == '"' || c == '\'' || c == '`');
                    return self.resolve_import_specifier(specifier, current_file);
                }
            }
        }
        
        None
    }
    
    fn language(&self) -> tree_sitter::Language {
        tree_sitter_javascript::language()
    }
    
    fn file_extensions(&self) -> &[&str] {
        &["js", "jsx", "ts", "tsx", "mjs", "cjs"]
    }
}
```

---

## Phase 4: Update Symbol Harvester

### Step 4.1: Make SymbolHarvester Language-Aware

**File:** `rust_core/src/parser.rs`

1. **Modify the `SymbolHarvester` struct:**

```rust
pub struct SymbolHarvester {
    // Store queries for each language
    python_query: Query,
    javascript_query: Query,
    typescript_query: Query,
}

impl SymbolHarvester {
    pub fn new() -> Self {
        Self {
            python_query: Query::new(
                tree_sitter_python::language(),
                include_str!("../queries/python/symbols.scm")
            ).expect("Failed to load Python symbols query"),
            javascript_query: Query::new(
                tree_sitter_javascript::language(),
                include_str!("../queries/javascript/symbols.scm")
            ).expect("Failed to load JavaScript symbols query"),
            typescript_query: Query::new(
                tree_sitter_typescript::language_typescript(),
                include_str!("../queries/typescript/symbols.scm")
            ).expect("Failed to load TypeScript symbols query"),
        }
    }
    
    pub fn harvest(&self, file_path: &Path, tree: &Tree, source: &[u8]) -> Vec<Symbol> {
        let query = self.select_query(file_path);
        let mut symbols = Vec::new();
        let mut cursor = QueryCursor::new();
        
        // ... rest of harvesting logic using the selected query
    }
    
    fn select_query(&self, file_path: &Path) -> &Query {
        match file_path.extension().and_then(|s| s.to_str()) {
            Some("py") | Some("pyi") => &self.python_query,
            Some("ts") | Some("tsx") => &self.typescript_query,
            Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => &self.javascript_query,
            _ => &self.python_query, // fallback
        }
    }
}
```

### Step 4.2: Update Parser Selection in RepoGraph

**File:** `rust_core/src/graph.rs`

1. **In the `process_file` or similar method, select the correct parser:**

```rust
fn get_parser_for_file(&self, file_path: &Path) -> Parser {
    let mut parser = Parser::new();
    
    let language = match file_path.extension().and_then(|s| s.to_str()) {
        Some("py") | Some("pyi") => tree_sitter_python::language(),
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => tree_sitter_javascript::language(),
        Some("ts") => tree_sitter_typescript::language_typescript(),
        Some("tsx") => tree_sitter_typescript::language_tsx(),
        _ => tree_sitter_python::language(), // fallback
    };
    
    parser.set_language(language).expect("Failed to set language");
    parser
}
```

---

## Phase 5: Update PyO3 Bindings

### Step 5.1: Modify Python API

**File:** `rust_core/src/lib.rs`

1. **Update `PyRepoGraph::new` to accept a language parameter:**

```rust
#[pymethods]
impl PyRepoGraph {
    #[new]
    fn new(project_root: String, language: Option<String>) -> PyResult<Self> {
        let lang = language.unwrap_or_else(|| "python".to_string());
        let graph = RepoGraph::new(PathBuf::from(project_root), &lang);
        Ok(PyRepoGraph { inner: Arc::new(Mutex::new(graph)) })
    }
    
    // ... rest of methods
}
```

2. **Update the module documentation:**

```rust
/// Atlas Semantic Engine
/// 
/// A multi-language code analysis engine supporting Python, JavaScript, and TypeScript.
#[pymodule]
fn semantic_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRepoGraph>()?;
    Ok(())
}
```

---

## Phase 6: Testing and Validation

### Step 6.1: Create Test Fixtures

**Directory:** `rust_core/tests/fixtures/js_test_repo/`

Create a minimal JavaScript test repository:

```
js_test_repo/
├── tsconfig.json
├── src/
│   ├── index.js
│   ├── utils/
│   │   ├── helpers.js
│   │   └── index.js
│   └── components/
│       └── Button.tsx
```

**File:** `js_test_repo/tsconfig.json`
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

**File:** `js_test_repo/src/index.js`
```javascript
import { helper } from './utils';
import Button from '@/components/Button';

function main() {
    helper();
    return Button;
}
```

**File:** `js_test_repo/src/utils/helpers.js`
```javascript
export function helper() {
    return 'help';
}
```

### Step 6.2: Write Integration Tests

**File:** `rust_core/tests/test_js_import_resolution.rs` (new file)

```rust
use semantic_engine::graph::RepoGraph;
use std::path::PathBuf;

#[test]
fn test_js_import_resolution() {
    let test_repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/js_test_repo");
    
    let mut graph = RepoGraph::new(test_repo.clone(), "javascript");
    graph.scan_repository();
    graph.build_complete();
    
    // Verify files were indexed
    assert!(graph.path_to_node.len() > 0);
    
    // Verify edges were created
    let edge_count = graph.graph.edge_count();
    assert!(edge_count > 0, "Expected edges but got zero");
}

#[test]
fn test_ts_path_alias_resolution() {
    // Test that @/* aliases resolve correctly
    // ... similar test structure
}
```

### Step 6.3: Update Existing Tests

**File:** `rust_core/src/import_resolver.rs`

Update all existing tests in the `#[cfg(test)]` module to use `PythonImportResolver` instead of `ImportResolver`.

### Step 6.4: Manual Testing Script

**File:** `python_shell/test_js_graph.py` (new file)

```python
from semantic_engine import PyRepoGraph

def test_javascript_repo():
    graph = PyRepoGraph("./test_repos/js_example", language="javascript")
    graph.scan_repository()
    graph.build_complete()
    
    map_output = graph.generate_map(max_tokens=2000)
    print(map_output)
    
    # Verify non-zero PageRanks
    assert "PageRank: 0.000" not in map_output, "All files have zero PageRank!"
    print("✓ JavaScript graph built successfully")

if __name__ == "__main__":
    test_javascript_repo()
```

---

## Phase 7: Documentation and Cleanup

### Step 7.1: Update README/Documentation

Document the new multi-language support:
- How to specify language when creating a graph
- Supported languages and their limitations
- How tsconfig path aliases work

### Step 7.2: Add Cargo Feature Flags (Optional)

If you want to make language support optional:

**File:** `Cargo.toml`

```toml
[features]
default = ["python", "javascript"]
python = ["tree-sitter-python"]
javascript = ["tree-sitter-javascript", "tree-sitter-typescript"]
```

---

## Summary Checklist

- [ ] Phase 1: Refactor to trait-based architecture
  - [ ] Define `ImportResolver` trait
  - [ ] Rename to `PythonImportResolver` and implement trait
  - [ ] Update `RepoGraph` to use `Box<dyn ImportResolver>`
  
- [ ] Phase 2: Create Tree-sitter queries
  - [ ] JavaScript symbols.scm and imports.scm
  - [ ] TypeScript symbols.scm and imports.scm
  
- [ ] Phase 3: Implement `JsTsImportResolver`
  - [ ] Add dependencies (serde_json, glob)
  - [ ] Create Node.js builtins list
  - [ ] Implement tsconfig parsing
  - [ ] Implement module indexing
  - [ ] Implement import resolution logic
  - [ ] Implement trait methods
  
- [ ] Phase 4: Update Symbol Harvester
  - [ ] Make language-aware with multiple queries
  - [ ] Add parser selection logic
  
- [ ] Phase 5: Update PyO3 bindings
  - [ ] Add language parameter to API
  
- [ ] Phase 6: Testing
  - [ ] Create JS test fixtures
  - [ ] Write integration tests
  - [ ] Update existing tests
  - [ ] Manual testing
  
- [ ] Phase 7: Documentation and cleanup

This plan provides a complete path from the current Python-only implementation to full JavaScript/TypeScript support while maintaining backwards compatibility and following Rust best practices.