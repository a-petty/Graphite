
import os
from pathlib import Path
from cortex.semantic_engine import RepoGraph, scan_repository

def generate_repo_map():
    """
    Initializes the RepoGraph, generates the repository map,
    and saves it to a file.
    """
    project_root = Path(__file__).parent.parent
    test_repo_path = project_root / "test_repo"
    output_file = project_root / "repo_map.log"

    print(f"Project Root: {project_root}")
    print(f"Analyzing repository: {test_repo_path}")

    if not test_repo_path.exists():
        print(f"Error: Test repository not found at {test_repo_path}")
        return

    try:
        # Initialize the graph from the Rust core
        print("Initializing RepoGraph...")
        repo_graph = RepoGraph(str(test_repo_path))

        # Scan the repository to get file paths
        print("Scanning repository for files...")
        file_paths = scan_repository(str(test_repo_path))
        print(f"Found {len(file_paths)} files.")

        # Build the graph from the file paths
        print("Building complete graph from file paths...")
        repo_graph.build_complete(file_paths)

        # Ensure PageRank is calculated
        print("Calculating PageRank...")
        repo_graph.ensure_pagerank_up_to_date()

        # Generate the map
        print("Generating repository map...")
        # The roadmap suggests generate_map takes a max_files argument.
        # Let's use a reasonable default.
        repo_map_content = repo_graph.generate_map(20)

        # Save the map to a file
        with open(output_file, "w") as f:
            f.write(repo_map_content)

        print(f"Successfully generated and saved repository map to: {output_file}")
        print("""
--- REPO MAP PREVIEW ---""")
        print(repo_map_content)
        print("""--- END PREVIEW ---
""")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_repo_map()
