use lazy_static::lazy_static;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use tree_sitter::{Query, QueryCursor, Tree};
use walkdir::WalkDir;
use crate::parser::SupportedLanguage;

lazy_static! {
    static ref PYTHON_STDLIB_MODULES: HashSet<&'static str> = [
        "os", "sys", "math", "json", "re", "collections", "datetime", "time", "random",
        "logging", "argparse", "io", "abc", "typing", "functools", "itertools",
        "heapq", "queue", "threading", "multiprocessing", "subprocess", "socket",
        "select", "ssl", "http", "urllib", "email", "csv", "xml", "html",
        "dataclasses", "enum", "decimal", "fractions", "pathlib", "shutil", "tempfile",
        "zipfile", "tarfile", "gzip", "bz2", "lzma", "hashlib", "hmac", "secrets",
        "array", "mmap", "struct", "warnings", "contextlib", "locale", "gettext",
        "unittest", "doctest", "pdb", "cProfile", "profile", "timeit", "trace",
        "linecache", "faulthandler", "inspect", "dis", "gc", "sysconfig", "builtins",
    ].iter().cloned().collect();

    static ref COMMON_THIRD_PARTY_MODULES: HashSet<&'static str> = [
        "numpy", "pandas", "django", "flask", "requests", "sqlalchemy", "werkzeug",
        "tensorflow", "torch", "matplotlib", "scipy", "sklearn", "pytz", "pytest",
        "PIL", "cv2", "fastapi", "pydantic", "starlette", "uvicorn", "jinja2", "itsdangerous"
    ].iter().cloned().collect();

    static ref NODE_BUILTIN_MODULES: HashSet<&'static str> = [
        "assert", "buffer", "child_process", "cluster", "crypto", "dgram",
        "dns", "domain", "events", "fs", "http", "https", "net", "os",
        "path", "punycode", "querystring", "readline", "stream", "string_decoder",
        "timers", "tls", "tty", "url", "util", "v8", "vm", "zlib",
        "worker_threads", "perf_hooks", "async_hooks", "inspector",
    ].iter().cloned().collect();
}

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

/// The result of an import resolution attempt, providing more context than Option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportResolutionResult {
    /// The import was successfully resolved to a file path within the project.
    Resolved(PathBuf),
    /// The import is identified as a standard library module.
    Stdlib(String),
    /// The import is identified as a likely third-party package.
    External(String),
    /// The import could not be resolved to any known file or category.
    Unresolved(String),
}

/// Common interface for language-specific import resolution.
pub trait ImportResolver: Debug + Send + Sync {
    /// Finds all resolvable, project-local imports in a file's AST.
    fn find_imports<'a>(
        &self,
        tree: &'a Tree,
        current_file: &Path,
        source: &'a [u8],
    ) -> HashSet<PathBuf>;

    /// Returns the file extensions this resolver handles.
    fn file_extensions(&self) -> &[&str];
}

/// Resolves Python import statements to absolute file paths.
#[derive(Debug)]
pub struct PythonImportResolver {
    pub project_root: PathBuf,
    /// A map of module names (e.g., `my_package.utils`) to their file paths.
    module_index: HashMap<String, PathBuf>,
    /// Compiled query for import statements.
    import_query: Query,
}

impl PythonImportResolver {
    /// Creates a new `PythonImportResolver` and indexes all Python modules in the project.
    pub fn new(project_root: &Path) -> Self {
        let mut resolver = Self {
            project_root: project_root.to_path_buf(),
            module_index: HashMap::new(),
            // Pre-compile the query for performance
            import_query: Query::new(
                tree_sitter_python::language(),
                r#"
                (import_statement name: (dotted_name) @module)
                (import_from_statement module_name: (dotted_name) @module)
                (import_from_statement module_name: (relative_import) @relative_import (dotted_name)? @module)
                (import_from_statement (wildcard_import))
                "#,
            )
            .expect("Failed to create import query"),
        };
        resolver.index_modules();
        resolver
    }

    /// Scans the project directory for Python files (.py) and builds the module index.
    fn index_modules(&mut self) {
        for entry in WalkDir::new(&self.project_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
        {
            let path = entry.path();
            let extension = path.extension().and_then(|s| s.to_str());

            if extension == Some("py") || extension == Some("pyi") {
                if let Ok(relative_path) = path.strip_prefix(&self.project_root) {
                    let mut components: Vec<String> = relative_path
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().to_string())
                        .collect();

                    let module_path_str =
                        if components.last() == Some(&"__init__.py".to_string())
                            || components.last() == Some(&"__init__.pyi".to_string())
                        {
                            components.pop(); // Remove `__init__.py`
                            components.join(".")
                        } else {
                            if let Some(last) = components.last_mut() {
                                if let Some(stem) = Path::new(last).file_stem() {
                                    *last = stem.to_string_lossy().to_string();
                                }
                            }
                            components.join(".")
                        };

                    if !module_path_str.is_empty() {
                        self.module_index
                            .insert(module_path_str, path.to_path_buf());
                    }
                }
            }
        }
    }

    fn resolve_absolute(&self, module_path: &str) -> Option<PathBuf> {
        // 1. Direct match (e.g., `my_package.utils` -> `my_package/utils.py`)
        if let Some(path) = self.module_index.get(module_path) {
            return Some(path.clone());
        }

        // 2. Package match (e.g., `my_package.utils` -> `my_package/utils/__init__.py`)
        let package_path_str = format!("{}.__init__", module_path);
        if let Some(path) = self.module_index.get(&package_path_str) {
            return Some(path.clone());
        }

        None
    }

    fn resolve_relative(
        &self,
        current_file: &Path,
        dots: usize,
        relative_module: Option<&str>,
    ) -> Option<PathBuf> {
        // Start at file's directory
        let file_dir = current_file.parent()?;
        let mut base_path = file_dir.to_path_buf();

        // Go up (dots - 1) levels.
        // For `.` (dots=1), loop doesn't run, base_path is current dir.
        // For `..` (dots=2), loop runs once, base_path is parent dir.
        for _ in 1..dots {
            base_path = base_path.parent()?.to_path_buf();
        }

        // If the current file is NOT in a package (no __init__.py), a `..` import
        // should look relative to the parent package, so we go up one more level.
        if !file_dir.join("__init__.py").exists() && dots > 1 {
            base_path = base_path.parent()?.to_path_buf();
        }

        // Convert base_path to a module string from project root.
        let rel_path = match base_path.strip_prefix(&self.project_root) {
            Ok(p) => p,
            Err(_) => return None,
        };

        let mut components: Vec<String> = rel_path
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();

        if let Some(module) = relative_module {
            if !module.is_empty() {
                components.extend(module.split('.').map(String::from));
            }
        }

        let module_str = components.join(".");
        self.resolve_absolute(&module_str)
    }
}

impl ImportResolver for PythonImportResolver {
    fn find_imports<'a>(
        &self,
        tree: &'a Tree,
        current_file: &Path,
        source: &'a [u8],
    ) -> HashSet<PathBuf> {
        let mut imports = HashSet::new();
        let mut query_cursor = QueryCursor::new();
        let matches = query_cursor.matches(&self.import_query, tree.root_node(), source);

        for match_ in matches {
            // The last pattern in the query is for wildcard imports, which we ignore.
            if match_.pattern_index == 3 {
                continue;
            }

            let mut relative_import_text: Option<&str> = None;
            let mut module_text: Option<&str> = None;

            for capture in match_.captures {
                let capture_name = &self.import_query.capture_names()[capture.index as usize];
                let text = capture.node.utf8_text(source).unwrap_or("");
                match capture_name.as_str() {
                    "relative_import" => relative_import_text = Some(text),
                    "module" => module_text = Some(text),
                    _ => (),
                }
            }
            
            let resolved_path = if let Some(relative_text) = relative_import_text {
                let dots = relative_text.chars().filter(|&c| c == '.').count();
                self.resolve_relative(current_file, dots, module_text)
            } else if let Some(module_text_str) = module_text {
                let root_module = module_text_str.split('.').next().unwrap_or("");
                if PYTHON_STDLIB_MODULES.contains(root_module)
                    || COMMON_THIRD_PARTY_MODULES.contains(root_module)
                {
                    None
                } else {
                    self.resolve_absolute(module_text_str)
                }
            } else {
                None
            };
            
            if let Some(path) = resolved_path {
                imports.insert(path);
            }
        }
        imports
    }

    fn file_extensions(&self) -> &[&str] {
        &["py", "pyi"]
    }
}

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
                include_str!("../queries/javascript/imports.scm"),
            )
            .expect("Failed to create JS import query"),
            import_query_ts: Query::new(
                tree_sitter_typescript::language_typescript(),
                include_str!("../queries/typescript/imports.scm"),
            )
            .expect("Failed to create TS import query"),
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
                    let without_ext =
                        path_str.trim_end_matches(ext.unwrap()).trim_end_matches('.');
                    self.module_index
                        .insert(without_ext.to_string(), path.to_path_buf());

                    // Also index /index files without the /index part
                    if path.file_stem().and_then(|s| s.to_str()) == Some("index") {
                        if let Some(parent_str) = Path::new(without_ext).parent() {
                            self.module_index.insert(
                                parent_str.to_string_lossy().to_string(),
                                path.to_path_buf(),
                            );
                        }
                    }
                }
            }
        }
    }

    fn resolve_import_specifier(
        &self,
        specifier: &str,
        current_file: &Path,
    ) -> Option<PathBuf> {
        // 1. Filter out Node.js built-ins
        if NODE_BUILTIN_MODULES.contains(specifier) {
            return None;
        }

        // 2. Handle Path Aliases or Node Modules
        if !specifier.starts_with('.') && !specifier.starts_with('/') {
            // Try to resolve as a path alias first
            if let Some(resolved) = self.resolve_path_alias(specifier) {
                return Some(resolved);
            }
            // If not an alias, assume it's a node_modules package and ignore it
            return None;
        }

        // 3. Resolve relative or absolute imports from the file system
        let base_path = if specifier.starts_with('/') {
            // Absolute path from project root
            self.project_root.clone()
        } else {
            // Relative path from current file's directory
            current_file.parent()?.to_path_buf()
        };

        // trim '/' to prevent path.join from treating it as an absolute path on its own
        let target_path = base_path.join(specifier.trim_start_matches('/'));

        // Use the robust, unified extension resolution logic
        self.try_resolve_with_extensions(&target_path)
    }

    fn resolve_path_alias(&self, specifier: &str) -> Option<PathBuf> {
        for (alias_pattern, alias_paths) in &self.path_aliases {
            // Handle patterns like "@/*" -> ["src/*"]
            if let Some(prefix) = alias_pattern.strip_suffix("/*") {
                if let Some(remainder) = specifier.strip_prefix(prefix) {
                    let remainder = remainder.trim_start_matches('/');
                    for path_pattern in alias_paths {
                        if let Some(path_prefix) = path_pattern.strip_suffix("/*") {
                            let base = self.base_url.as_ref().unwrap_or(&self.project_root);
                            let resolved_base = base.join(path_prefix);
                            let full_path = resolved_base.join(remainder);
                            if let Some(path) = self.try_resolve_with_extensions(&full_path) {
                                return Some(path);
                            }
                        }
                    }
                }
            }
            // Handle exact matches, e.g., "react" -> "./node_modules/react"
            else if alias_pattern == specifier {
                for path_pattern in alias_paths {
                    let base = self.base_url.as_ref().unwrap_or(&self.project_root);
                    let full_path = base.join(path_pattern);
                    if let Some(path) = self.try_resolve_with_extensions(&full_path) {
                        return Some(path);
                    }
                }
            }
        }
        None
    }

    fn try_resolve_with_extensions(&self, base: &Path) -> Option<PathBuf> {
        let extensions = ["ts", "tsx", "js", "jsx", "mjs", "cjs"];

        // Case 1: The base path with an extension exists (e.g., .../Button -> .../Button.tsx)
        for ext in &extensions {
            let candidate = base.with_extension(ext);
            if candidate.is_file() {
                return Some(candidate);
            }
        }

        // Case 2: The base path is a directory with an index file (e.g., .../Button -> .../Button/index.tsx)
        for ext in &extensions {
            let candidate = base.join(format!("index.{}", ext));
            if candidate.is_file() {
                return Some(candidate);
            }
        }

        // Case 3: The base path itself is a file with a non-standard extension (e.g. import './styles.css')
        if base.is_file() {
            return Some(base.to_path_buf());
        }

        None
    }
}

impl ImportResolver for JsTsImportResolver {
    fn find_imports<'a>(
        &self,
        tree: &'a Tree,
        current_file: &Path,
        source: &'a [u8],
    ) -> HashSet<PathBuf> {
        let mut imports = HashSet::new();
        // Determine which query to use based on file extension
        let lang = SupportedLanguage::from_path(current_file);
        let query = match lang {
            SupportedLanguage::TypeScript | SupportedLanguage::TypeScriptTsx => &self.import_query_ts,
            _ => &self.import_query_js,
        };

        let mut cursor = QueryCursor::new();
        let matches = cursor.matches(query, tree.root_node(), source);

        for match_ in matches {
            for capture in match_.captures {
                let capture_name = &query.capture_names()[capture.index as usize];
                if capture_name == "import" {
                    if let Ok(import_text) = capture.node.utf8_text(source) {
                        let specifier = import_text.trim_matches(|c| c == '"' || c == '\'' || c == '`');
                        if let Some(path) = self.resolve_import_specifier(specifier, current_file) {
                             imports.insert(path);
                        }
                    }
                }
            }
        }
        imports
    }

    fn file_extensions(&self) -> &[&str] {
        &["js", "jsx", "ts", "tsx", "mjs", "cjs"]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;
    use tree_sitter::Parser;

    struct TestRepo {
        _root: tempfile::TempDir, // The TempDir is owned by this struct, so it gets cleaned up when dropped
        pub root_path: PathBuf,
        pub resolver: PythonImportResolver,
        pub parser: Parser,
    }

    fn setup() -> TestRepo {
        let root = tempdir().unwrap();
        let root_path = root.path().to_path_buf();

        // Create a dummy project structure
        create_dummy_file(&root_path, "app.py", "");
        create_dummy_file(&root_path, "src/utils.py", "");
        create_dummy_file(&root_path, "src/api/__init__.py", "");
        create_dummy_file(&root_path, "src/api/v1/endpoints.py", "");
        create_dummy_file(&root_path, "src/api/v1/helpers.py", "");
        create_dummy_file(&root_path, "README.md", ""); // Should be ignored

        let resolver = PythonImportResolver::new(&root_path);

        let mut parser = Parser::new();
        parser
            .set_language(tree_sitter_python::language())
            .expect("Failed to load python grammar");

        TestRepo {
            _root: root,
            root_path,
            resolver,
            parser,
        }
    }

    // Helper to create a dummy file and its parent directories.
    fn create_dummy_file(root: &Path, path: &str, content: &str) {
        let file_path = root.join(path);
        fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        fs::write(file_path, content).unwrap();
    }
    
    fn run_find_imports(repo: &mut TestRepo, source: &str, file_name: &str) -> HashSet<PathBuf> {
        let tree = repo.parser.parse(source, None).unwrap();
        repo.resolver.find_imports(&tree, &repo.root_path.join(file_name), source.as_bytes())
    }

    #[test]
    fn test_module_indexing() {
        let repo = setup();
        let index = &repo.resolver.module_index;
        assert_eq!(index.len(), 5);
        assert_eq!(index.get("app"), Some(&repo.root_path.join("app.py")));
        assert_eq!(
            index.get("src.utils"),
            Some(&repo.root_path.join("src/utils.py"))
        );
        assert_eq!(
            index.get("src.api"),
            Some(&repo.root_path.join("src/api/__init__.py"))
        );
        assert_eq!(
            index.get("src.api.v1.endpoints"),
            Some(&repo.root_path.join("src/api/v1/endpoints.py"))
        );
    }

    #[test]
    fn test_resolve_absolute() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "import src.utils", "app.py");
        assert_eq!(imports.len(), 1);
        assert!(imports.contains(&repo.root_path.join("src/utils.py")));
    }

    #[test]
    fn test_resolve_package() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "import src.api", "app.py");
        assert_eq!(imports.len(), 1);
        assert!(imports.contains(&repo.root_path.join("src/api/__init__.py")));
    }

    #[test]
    fn test_resolve_relative_single_dot() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "from . import helpers", "src/api/v1/endpoints.py");
        assert_eq!(imports.len(), 1);
        assert!(imports.contains(&repo.root_path.join("src/api/v1/helpers.py")));
    }

    #[test]
    fn test_resolve_relative_double_dot() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "from .. import utils", "src/api/v1/endpoints.py");
        assert_eq!(imports.len(), 1);
        assert!(imports.contains(&repo.root_path.join("src/utils.py")));
    }

    #[test]
    fn test_resolve_stdlib() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "import os", "app.py");
        assert!(imports.is_empty());
    }

    #[test]
    fn test_resolve_third_party() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "import numpy as np", "app.py");
        assert!(imports.is_empty());
    }

    #[test]
    fn test_resolve_non_existent() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "import project.non_existent", "app.py");
        assert!(imports.is_empty());
    }

    #[test]
    fn test_resolve_package_deep() {
        let mut repo = setup();
        create_dummy_file(&repo.root_path, "src/api/v1/__init__.py", "");
        repo.resolver.index_modules(); // Re-index after adding a new file

        let imports = run_find_imports(&mut repo, "import src.api.v1", "app.py");
        assert_eq!(imports.len(), 1);
        assert!(imports.contains(&repo.root_path.join("src/api/v1/__init__.py")));
    }

    #[test]
    fn test_resolve_star_import() {
        let mut repo = setup();
        let imports = run_find_imports(&mut repo, "from . import *", "src/api/v1/endpoints.py");
        assert!(imports.is_empty());
    }
}