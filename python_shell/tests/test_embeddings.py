"""Tests for EmbeddingManager: skeleton input, chunk splitting, and max-chunk similarity."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from atlas.embeddings import EmbeddingManager, FileEmbedding, _DEFINITION_PREFIXES


# ---------------------------------------------------------------------------
# Chunk splitting tests
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    """Tests for _split_into_chunks method."""

    def _make_manager(self):
        """Create an EmbeddingManager with mocked FastEmbed model."""
        with patch("atlas.embeddings.TextEmbedding"):
            return EmbeddingManager()

    def test_small_text_single_chunk(self):
        """Text shorter than the word limit stays as a single chunk."""
        mgr = self._make_manager()
        text = "import os\nimport sys\n\ndef hello():\n    pass\n"
        chunks = mgr._split_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_python_def(self):
        """Multiple Python function definitions create separate chunks."""
        mgr = self._make_manager()
        # Build text long enough to exceed _MAX_CHUNK_WORDS
        lines = ["import os", "import sys", ""]
        for i in range(20):
            lines.append(f"def function_{i}(x, y):")
            lines.append(f'    """Docstring for function {i}."""')
            # Add enough words to push past the threshold
            lines.append(f"    result = x + y + {i}")
            for j in range(15):
                lines.append(f"    value_{j} = compute_something(x, y, {j})")
            lines.append("")
        text = "\n".join(lines)

        chunks = mgr._split_into_chunks(text)
        assert len(chunks) > 1
        # First chunk should contain imports (header)
        assert "import os" in chunks[0]
        # Each subsequent chunk should start with a def
        for chunk in chunks[1:]:
            assert chunk.strip().startswith("def ")

    def test_splits_on_python_class(self):
        """Class definitions create chunk boundaries."""
        mgr = self._make_manager()
        lines = []
        for i in range(10):
            lines.append(f"class MyClass{i}:")
            lines.append(f'    """Class {i} docstring."""')
            for j in range(20):
                lines.append(f"    attr_{j} = {j}")
            lines.append("")
        text = "\n".join(lines)

        chunks = mgr._split_into_chunks(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.strip().startswith("class ")

    def test_splits_on_async_def(self):
        """async def is recognized as a chunk boundary."""
        mgr = self._make_manager()
        lines = []
        for i in range(15):
            lines.append(f"async def handler_{i}(request):")
            lines.append(f'    """Handle request {i}."""')
            for j in range(20):
                lines.append(f"    data_{j} = await fetch_data({j})")
            lines.append("")
        text = "\n".join(lines)

        chunks = mgr._split_into_chunks(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.strip().startswith("async def ")

    def test_splits_on_rust_fn(self):
        """Rust fn/pub fn are recognized as chunk boundaries."""
        mgr = self._make_manager()
        lines = []
        for i in range(15):
            lines.append(f"pub fn process_{i}(input: &str) -> String {{")
            for j in range(20):
                lines.append(f"    let var_{j} = input.to_string();")
            lines.append("}")
            lines.append("")
        text = "\n".join(lines)

        chunks = mgr._split_into_chunks(text)
        assert len(chunks) > 1

    def test_splits_on_js_function(self):
        """JS function/export function are recognized as chunk boundaries."""
        mgr = self._make_manager()
        lines = []
        for i in range(15):
            lines.append(f"export function handler{i}(req, res) {{")
            for j in range(20):
                lines.append(f"    const val{j} = req.params['{j}'];")
            lines.append("}")
            lines.append("")
        text = "\n".join(lines)

        chunks = mgr._split_into_chunks(text)
        assert len(chunks) > 1

    def test_empty_text_returns_single_chunk(self):
        """Empty text returns itself as a single chunk."""
        mgr = self._make_manager()
        chunks = mgr._split_into_chunks("")
        assert chunks == [""]

    def test_no_definitions_returns_single_chunk(self):
        """Text with no definition keywords but over the word limit stays as one chunk."""
        mgr = self._make_manager()
        text = " ".join(["word"] * 500)
        chunks = mgr._split_into_chunks(text)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Max-chunk similarity tests
# ---------------------------------------------------------------------------

class TestFileSimilarity:
    """Tests for _file_similarity max-chunk scoring."""

    def _make_manager(self):
        with patch("atlas.embeddings.TextEmbedding"):
            return EmbeddingManager()

    def test_max_chunk_used(self):
        """Similarity should be the maximum across chunks, not the average."""
        mgr = self._make_manager()
        query = np.array([1.0, 0.0, 0.0])
        entry = FileEmbedding(chunk_embeddings=[
            np.array([0.1, 0.9, 0.0]),   # low similarity to query
            np.array([0.95, 0.05, 0.0]),  # high similarity to query
            np.array([0.0, 0.0, 1.0]),    # orthogonal to query
        ])

        sim = mgr._file_similarity(query, entry)
        # Should use the best chunk (second one), not average
        individual_sims = [mgr._cosine_similarity(query, e) for e in entry.chunk_embeddings]
        assert sim == max(individual_sims)
        assert sim == pytest.approx(individual_sims[1])

    def test_empty_chunks_returns_zero(self):
        """A FileEmbedding with no chunks should return 0.0."""
        mgr = self._make_manager()
        query = np.array([1.0, 0.0, 0.0])
        entry = FileEmbedding(chunk_embeddings=[])
        assert mgr._file_similarity(query, entry) == 0.0

    def test_single_chunk_matches_direct_cosine(self):
        """With one chunk, file similarity equals direct cosine similarity."""
        mgr = self._make_manager()
        query = np.array([1.0, 0.0, 0.0])
        chunk = np.array([0.5, 0.5, 0.0])
        entry = FileEmbedding(chunk_embeddings=[chunk])

        assert mgr._file_similarity(query, entry) == pytest.approx(
            mgr._cosine_similarity(query, chunk)
        )


# ---------------------------------------------------------------------------
# Cosine similarity edge cases
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Edge cases for _cosine_similarity."""

    def _make_manager(self):
        with patch("atlas.embeddings.TextEmbedding"):
            return EmbeddingManager()

    def test_zero_vector_returns_zero(self):
        """A zero vector should return 0.0, not NaN."""
        mgr = self._make_manager()
        assert mgr._cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
        assert mgr._cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 0.0])) == 0.0

    def test_identical_vectors_return_one(self):
        mgr = self._make_manager()
        v = np.array([0.5, 0.3, 0.8])
        assert mgr._cosine_similarity(v, v) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Skeleton fallback tests
# ---------------------------------------------------------------------------

class TestGetEmbeddingText:
    """Tests for _get_embedding_text skeleton-or-fallback logic."""

    def _make_manager_with_graph(self, skeleton_return="def foo(): ..."):
        """Create an EmbeddingManager with a mock repo_graph."""
        with patch("atlas.embeddings.TextEmbedding"):
            mock_graph = MagicMock()
            mock_graph.get_skeleton.return_value = skeleton_return
            mgr = EmbeddingManager(repo_graph=mock_graph)
            return mgr, mock_graph

    def test_uses_skeleton_when_available(self, tmp_path):
        """Should use skeleton from repo_graph instead of raw file content."""
        mgr, mock_graph = self._make_manager_with_graph("def foo(): ...")
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")

        result = mgr._get_embedding_text(test_file)
        assert result == "def foo(): ..."
        mock_graph.get_skeleton.assert_called_once()

    def test_falls_back_to_raw_when_skeleton_empty(self, tmp_path):
        """Should fall back to raw content when skeleton is empty."""
        mgr, _ = self._make_manager_with_graph("")
        test_file = tmp_path / "test.py"
        test_file.write_text("raw content here")

        result = mgr._get_embedding_text(test_file)
        assert result == "raw content here"

    def test_falls_back_to_raw_when_skeleton_raises(self, tmp_path):
        """Should fall back to raw content when get_skeleton raises."""
        with patch("atlas.embeddings.TextEmbedding"):
            mock_graph = MagicMock()
            mock_graph.get_skeleton.side_effect = RuntimeError("not found")
            mgr = EmbeddingManager(repo_graph=mock_graph)

        test_file = tmp_path / "test.py"
        test_file.write_text("fallback content")

        result = mgr._get_embedding_text(test_file)
        assert result == "fallback content"

    def test_falls_back_when_no_repo_graph(self, tmp_path):
        """Should read raw file when repo_graph is None."""
        with patch("atlas.embeddings.TextEmbedding"):
            mgr = EmbeddingManager(repo_graph=None)

        test_file = tmp_path / "test.py"
        test_file.write_text("no graph content")

        result = mgr._get_embedding_text(test_file)
        assert result == "no graph content"


# ---------------------------------------------------------------------------
# Module prefix tests
# ---------------------------------------------------------------------------

class TestFilePathToModulePrefix:
    """Tests for _file_path_to_module_prefix method."""

    def _make_manager(self, project_root):
        with patch("atlas.embeddings.TextEmbedding"):
            return EmbeddingManager(project_root=project_root)

    def test_basic_python_file(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr._file_path_to_module_prefix(tmp_path / "mypackage" / "models" / "user.py")
        assert result == "mypackage.models.user"

    def test_strips_src_prefix(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr._file_path_to_module_prefix(tmp_path / "src" / "airflow" / "models" / "pool.py")
        assert result == "airflow.models.pool"

    def test_strips_lib_prefix(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr._file_path_to_module_prefix(tmp_path / "lib" / "utils" / "helpers.py")
        assert result == "utils.helpers"

    def test_init_file_stripped(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr._file_path_to_module_prefix(tmp_path / "mypackage" / "__init__.py")
        assert result == "mypackage"

    def test_top_level_file(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr._file_path_to_module_prefix(tmp_path / "setup.py")
        assert result == "setup"

    def test_no_project_root_returns_empty(self):
        with patch("atlas.embeddings.TextEmbedding"):
            mgr = EmbeddingManager(project_root=None)
        result = mgr._file_path_to_module_prefix(Path("/some/file.py"))
        assert result == ""

    def test_file_outside_project_returns_empty(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr._file_path_to_module_prefix(Path("/completely/different/path.py"))
        assert result == ""

    def test_nested_src_dir(self, tmp_path):
        """Handles project structures like airflow-core/src/airflow/...

        The src/ directory is stripped, but the workspace package name
        (airflow-core) is preserved — it's not a source directory.
        """
        mgr = self._make_manager(tmp_path)
        result = mgr._file_path_to_module_prefix(
            tmp_path / "airflow-core" / "src" / "airflow" / "models" / "pool.py"
        )
        assert result == "airflow-core.airflow.models.pool"

    def test_prefix_prepended_to_embedding_text(self, tmp_path):
        """_get_embedding_text should prepend the module prefix."""
        with patch("atlas.embeddings.TextEmbedding"):
            mgr = EmbeddingManager(project_root=tmp_path)
        test_file = tmp_path / "mypackage" / "core.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("def hello(): pass")

        result = mgr._get_embedding_text(test_file)
        assert result.startswith("mypackage.core\n")
        assert "def hello(): pass" in result


# ---------------------------------------------------------------------------
# Integration: find_relevant_files with chunk pipeline
# ---------------------------------------------------------------------------

class TestFindRelevantFilesChunked:
    """Test the full find_relevant_files flow with chunked embeddings."""

    def test_caches_file_embeddings(self, tmp_path):
        """Files should be cached after first embedding and reused on second call."""
        with patch("atlas.embeddings.TextEmbedding") as MockModel:
            mock_instance = MagicMock()
            # Return 384-dim embeddings
            dim = 384
            mock_instance.embed.side_effect = lambda texts: [
                np.random.randn(dim) for _ in texts
            ]
            MockModel.return_value = mock_instance

            mgr = EmbeddingManager()
            f1 = tmp_path / "a.py"
            f1.write_text("def hello(): pass")

            # First call: should embed
            mgr.find_relevant_files("hello", [f1], top_n=1)
            assert f1 in mgr.embeddings_cache

            call_count = mock_instance.embed.call_count

            # Second call: should use cache (no new embed call for file)
            mgr.find_relevant_files("hello", [f1], top_n=1)
            # Only one new call for the query embedding, no file re-embedding
            assert mock_instance.embed.call_count == call_count + 1

    def test_returns_correct_number_of_results(self, tmp_path):
        """Should return at most top_n results."""
        with patch("atlas.embeddings.TextEmbedding") as MockModel:
            mock_instance = MagicMock()
            dim = 384
            mock_instance.embed.side_effect = lambda texts: [
                np.random.randn(dim) for _ in texts
            ]
            MockModel.return_value = mock_instance

            mgr = EmbeddingManager()
            files = []
            for i in range(10):
                f = tmp_path / f"file_{i}.py"
                f.write_text(f"def func_{i}(): pass")
                files.append(f)

            result = mgr.find_relevant_files("test query", files, top_n=3)
            assert len(result) == 3
            assert all(isinstance(p, Path) for p in result)

    def test_scored_returns_tuples_with_scores(self, tmp_path):
        """find_relevant_files_scored should return (Path, float) tuples."""
        with patch("atlas.embeddings.TextEmbedding") as MockModel:
            mock_instance = MagicMock()
            dim = 384
            mock_instance.embed.side_effect = lambda texts: [
                np.random.randn(dim) for _ in texts
            ]
            MockModel.return_value = mock_instance

            mgr = EmbeddingManager()
            files = []
            for i in range(5):
                f = tmp_path / f"file_{i}.py"
                f.write_text(f"def func_{i}(): pass")
                files.append(f)

            result = mgr.find_relevant_files_scored("test query", files, top_n=3)
            assert len(result) == 3
            for path, score in result:
                assert isinstance(path, Path)
                assert isinstance(score, float)
            # Scores should be sorted descending
            scores = [s for _, s in result]
            assert scores == sorted(scores, reverse=True)

    def test_scored_consistent_with_unscored(self, tmp_path):
        """find_relevant_files and find_relevant_files_scored should return same ordering."""
        with patch("atlas.embeddings.TextEmbedding") as MockModel:
            mock_instance = MagicMock()
            dim = 384
            # Use deterministic embeddings
            np.random.seed(42)
            mock_instance.embed.side_effect = lambda texts: [
                np.random.randn(dim) for _ in texts
            ]
            MockModel.return_value = mock_instance

            mgr = EmbeddingManager()
            files = []
            for i in range(5):
                f = tmp_path / f"file_{i}.py"
                f.write_text(f"def func_{i}(): pass")
                files.append(f)

            scored = mgr.find_relevant_files_scored("test", files, top_n=3)
            scored_paths = [p for p, _ in scored]

            # Reset seed and create a fresh manager for consistent comparison
            np.random.seed(42)
            mock_instance.embed.side_effect = lambda texts: [
                np.random.randn(dim) for _ in texts
            ]
            mgr2 = EmbeddingManager()
            unscored = mgr2.find_relevant_files("test", files, top_n=3)

            assert scored_paths == unscored
