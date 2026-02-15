 Here is a specific, step-by-step plan to implement Step 3.2: Add Context Management, which corresponds to the initial parts of Phase 5
  from the roadmap.md.

  Our goal is to implement the "Anchor & Expand" strategy. This involves creating two new components: an EmbeddingManager to handle vector
  search ("Anchor") and a ContextManager to use those results with our existing RepoGraph ("Expand").

  ---

  Implementation Plan: Context Management

  Step 1: Implement the Embedding Service

  We'll start by creating the service responsible for generating and searching vector embeddings.

   * File to Create: python_shell/atlas/embeddings.py
   * Class to Implement: EmbeddingManager
   * Key Responsibilities:
       1. Initialization: In __init__, load the fastembed model (BAAI/bge-small-en-v1.5) and initialize an in-memory dictionary to serve
          as a simple cache for file embeddings.
       2. Embedding Generation: Create a method to generate embeddings for a list of file contents. This will be the core interaction with
          the fastembed library.
       3. Vector Search: Create a find_relevant_files(query: str, files: List[Path]) method. This method will:
           * Generate an embedding for the user's query.
           * Ensure all files in the repository have their embeddings generated and cached.
           * Perform a cosine similarity search between the query embedding and all file embeddings.
           * Return the top 3-5 file paths that are most similar to the query. These will be our "anchor" files.

  Step 2: Expose Dependency Methods in Rust Core

  Our ContextManager will need to ask the RepoGraph for the dependencies of the anchor files. We must expose this function from Rust to
  Python.

   * File to Modify: rust_core/src/graph.rs
       * Action: Implement get_dependencies(path) and get_dependents(path) methods. These will traverse the graph from a given file node
         and return a list of its incoming and outgoing neighbors.
   * File to Modify: rust_core/src/lib.rs
       * Action: Add the new methods to the #[pymethods] block for PyRepoGraph, making them callable from Python.

  Step 3: Implement the Context Orchestrator

  This is the "brain" of our context strategy, combining the results from the previous steps.

   * File to Create: python_shell/atlas/context.py
   * Class to Implement: ContextManager
   * Key Responsibilities:
       1. Initialization: In __init__, it will take the RepoGraph and EmbeddingManager instances as arguments. It will also initialize the
          tiktoken encoder for managing the token budget.
       2. Orchestration (`assemble_context` method): This main method will execute the "Anchor & Expand" strategy:
           * Anchor: Call the EmbeddingManager to find the semantically relevant "anchor" files based on the user's query.
           * Expand: Use the newly exposed get_dependencies and get_dependents methods on the RepoGraph instance to build a "neighborhood"
             of files surrounding the anchors.
           * Assemble & Budget: Create the final prompt text. It will intelligently add content in order of importance (Repo Map -> Full
             content of "neighborhood" files -> Skeletons of other high-PageRank files) until the token budget is reached.

  Step 4: Integrate into the Main Agent

  Finally, we'll wire the new ContextManager into the AtlasAgent.

   * File to Modify: python_shell/atlas/agent.py
   * Action:
       1. In __init__, instantiate EmbeddingManager and ContextManager.
       2. In the query() method, replace the placeholder logic with a call to self.context_manager.assemble_context(user_input).

  This plan provides a clear path forward, starting with the foundational embedding service and progressively building up to the full
  context assembly logic, ensuring each component can be developed and tested in a logical order.