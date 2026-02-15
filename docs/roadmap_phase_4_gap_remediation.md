Here is a detailed outline of the changes required to fix the issues in the Phase 4 implementation and align it with the
  roadmap.

  The plan is divided into three parts, addressing each major shortcoming.

  ---

  Part 1: Fix Memory Leak in Incremental Parser (`rust_core/src/incremental_parser.rs`)

  The current IncrementalParser caches every parsed tree and source file indefinitely, which is a memory leak. The fix is to use a Least
  Recently Used (LRU) cache to bound memory usage.

   1. Add `lru` Crate Dependency:
       * Modify rust_core/Cargo.toml to include lru = "0.12" under [dependencies].

   2. Update the `IncrementalParser` Struct:
       * In rust_core/src/incremental_parser.rs, replace the HashMap fields with LruCache.
       * From:

   1         // Cache of most recent trees for each file.
   2         trees: HashMap<PathBuf, Tree>,
   3
   4         // Cache of the source code corresponding to each tree.
   5         sources: HashMap<PathBuf, String>,
       * To:

    1         use lru::LruCache;
    2         use std::num::NonZeroUsize;
    3
    4         // ...
    5
    6         // Cache of most recent trees for each file.
    7         trees: LruCache<PathBuf, Tree>,
    8
    9         // Cache of the source code corresponding to each tree.
   10         sources: LruCache<PathBuf, String>,

   3. Update `IncrementalParser::new()`:
       * Initialize the LruCache with a fixed size (e.g., 100 entries).
       * Change:

   1         pub fn new() -> Self {
   2             let cache_size = NonZeroUsize::new(100).unwrap();
   3             Self {
   4                 parser_pool: ParserPool::new_py(),
   5                 trees: LruCache::new(cache_size),
   6                 sources: LruCache::new(cache_size),
   7             }
   8         }

   4. Update Cache Access Logic:
       * Replace .get() and .insert() calls with the LruCache equivalents (.get(), .put()). The API is similar, so this is a
         straightforward change in the update_file method.

  Part 2: Implement Correct Incremental Graph Updates (`rust_core/src/graph.rs`)

  The current update_file function is inefficient and incorrect. It must be refactored to be truly incremental.

   1. Implement True Symbol Index Cleanup:
       * Before harvesting new symbols in update_file, get the file's old symbols from the FileNode being updated.
       * Iterate through the old definitions and remove the current file's path from the global symbol_index.definitions.
       * Remove the file's entry completely from symbol_index.usages.

   2. Implement Targeted Edge Updates:
       * Remove the call to `self.build_semantic_edges()` within update_file.
       * Step 1: Edge Removal. Before updating symbols, get the NodeIndex of the changed file. Find all incoming and outgoing SymbolUsage
         edges connected to it and remove them. This isolates the node cleanly.
       * Step 2: Edge Recreation. After the SymbolIndex has been cleaned and updated with new symbols:
           * Recreate Outgoing Edges: For each symbol in the file's new_usages, look up its definition(s) in the SymbolIndex and create
             edges from the current file's node to the definition nodes.
           * Recreate Incoming Edges: For each symbol in the file's new_definitions, find all files that use that symbol (via SymbolIndex)
             and create edges from those user nodes to the current file's node.

  This ensures only the affected portion of the graph is rewired, rather than rebuilding all semantic links.

  Part 3: Implement the Missing Multi-Tier Update Strategy

  This involves creating the classification logic in Rust and exposing it to the Python orchestration layer.

   1. Define the `UpdateTier` Enum:
       * In rust_core/src/graph.rs, create a public enum to represent the magnitude of a change.

   1         pub enum UpdateTier {
   2             // Only the function body changed. No change to symbols or imports.
   3             Local,
   4             // The file's symbols changed (e.g., a function was added/removed).
   5             FileScope,
   6             // An import statement was changed, affecting graph structure.
   7             GraphScope,
   8         }

   2. Implement a `classify_change` Function:
       * Create a new public function in RepoGraph that takes a file path and its new content.
       * It will compare the file's old imports and symbols (stored in the graph) with the new ones (by parsing the new content).
       * Logic:
           * If imports have changed -> return UpdateTier::GraphScope.
           * If symbols (definitions) have changed -> return UpdateTier::FileScope.
           * Otherwise -> return UpdateTier::Local.

   3. Refactor `update_file` into Tiered Functions:
       * Break the monolithic update_file into smaller, tiered functions.
       * update_local(path, new_content): The lightest update. Re-parses the file to update its skeleton for context, but performs no
         symbol or graph updates.
       * update_file_scope(path, new_content): Performs a Local update and the full incremental symbol/edge rewiring from Part 2.
       * The Python agent will be responsible for calling the correct function based on the classification.

   4. Expose to Python:
       * Use #[pymethods] to expose the classify_change function and the new tiered update functions to the Python shell.

  By implementing these changes, the agent will have a fast, efficient, and correct incremental update system that aligns with the
  project's architectural goals.