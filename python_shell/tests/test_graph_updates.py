# python_shell/tests/test_graph_updates.py

import pytest
from pathlib import Path
from graphite.semantic_engine import RepoGraph, GraphUpdateResult, scan_repository

@pytest.fixture
def dummy_repo(tmp_path):
    """Creates a temporary dummy repository for testing."""
    repo_root = tmp_path / "test_repo"
    repo_root.mkdir()
    
    (repo_root / "src").mkdir()
    (repo_root / "src" / "models").mkdir()
    
    (repo_root / "src" / "main.py").write_text("import src.auth\nimport src.utils")
    (repo_root / "src" / "auth.py").write_text("from src.models import user")
    (repo_root / "src" / "utils.py").write_text("# No imports")
    (repo_root / "src" / "models" / "user.py").write_text("class User: pass")
    
    return repo_root

def test_python_update_file(dummy_repo):
    """Tests the full flow of building a graph and then updating a single file.

    With tree-sitter removed (Phase 0b), update_file only updates the content
    hash — no edge changes or pagerank recalculation.
    """
    # 1. Initial Scan and Build
    scanned_files = scan_repository(str(dummy_repo))

    graph = RepoGraph(str(dummy_repo))
    graph.build_complete(scanned_files)

    main_path = str(dummy_repo / "src" / "main.py")

    # 2. Verify initial state
    initial_map = graph.generate_map(10)
    print(f"Initial Map:\n{initial_map}")
    assert "main.py" in initial_map

    # 3. Modify a file
    with open(main_path, "a") as f:
        f.write("\n# updated content")

    # 4. Perform incremental update
    content = Path(main_path).read_text()
    update_result = graph.update_file(main_path, content)

    assert isinstance(update_result, GraphUpdateResult)
    # Without parsing, no edge changes occur
    assert update_result.edges_added == 0
    assert update_result.needs_pagerank_recalc is False

    # 5. Verify graph still works after update
    updated_map = graph.generate_map(10)
    print(f"Updated Map:\n{updated_map}")
    assert "main.py" in updated_map


def test_python_lazy_pagerank(dummy_repo):
    """Tests that PageRank is not recalculated if there are no structural changes."""

    scanned_files = scan_repository(str(dummy_repo))
    graph = RepoGraph(str(dummy_repo))
    graph.build_complete(scanned_files)

    main_path = str(dummy_repo / "src" / "main.py")

    # Get initial ranks
    initial_ranks = graph.get_top_ranked_files(5)
    
    # Modify a file with a non-structural change (add a comment)
    with open(main_path, "a") as f:
        f.write("\n# A comment")
    
    # Read the new content and pass it to the update function
    content = Path(main_path).read_text()
    update_result = graph.update_file(main_path, content)
    assert update_result.needs_pagerank_recalc is False
    
    # Get ranks again, they should be identical
    # Note: `get_top_ranked_files` calls `ensure_pagerank_up_to_date`, which will NOT
    # run the calculation because the `pagerank_dirty` flag was not set.
    new_ranks = graph.get_top_ranked_files(5)
    
    # Convert to dict for easier comparison, ignoring minor float differences
    initial_ranks_dict = {path: f"{rank:.5f}" for path, rank in initial_ranks}
    new_ranks_dict = {path: f"{rank:.5f}" for path, rank in new_ranks}

    assert initial_ranks_dict == new_ranks_dict
