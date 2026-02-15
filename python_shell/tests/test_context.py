"""Tests for ContextManager._get_dependency_neighborhood multi-hop BFS."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class FakeRepoGraph:
    """Minimal mock of RepoGraph for testing neighborhood traversal."""

    def __init__(self):
        # dep_map: {file_str: [(neighbor_str, edge_kind), ...]}
        self.dep_map = {}
        # dependent_map: {file_str: [(dependent_str, edge_kind), ...]}
        self.dependent_map = {}

    def add_edge(self, source: str, target: str, edge_kind: str):
        """Add a directed edge: source depends on target."""
        self.dep_map.setdefault(source, []).append((target, edge_kind))
        self.dependent_map.setdefault(target, []).append((source, edge_kind))

    def get_dependencies(self, path: str):
        return self.dep_map.get(path, [])

    def get_dependents(self, path: str):
        return self.dependent_map.get(path, [])

    def get_top_ranked_files(self, n):
        return []

    def generate_map(self, max_files=50):
        return ""


def _make_context_manager(graph):
    """Construct a ContextManager with mocked embedding manager and tiktoken."""
    with patch("atlas.context.tiktoken") as mock_tiktoken:
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = []
        mock_tiktoken.encoding_for_model.return_value = mock_encoder

        mock_embed = MagicMock()
        from atlas.context import ContextManager
        cm = ContextManager(graph, mock_embed, model="gpt-4", max_tokens=100000)
        return cm


class TestBugFix:
    """Verify the original bug (edge_kind not in neighborhood) is fixed."""

    def test_import_edge_dedup_uses_dep_path(self):
        """Import edges should not always be added — only if dep_path is new."""
        graph = FakeRepoGraph()
        # A -> B via Import, A -> B via SymbolUsage (same target, two edges)
        a = str(Path("/repo/a.py").resolve())
        b = str(Path("/repo/b.py").resolve())
        graph.add_edge(a, b, "Import")
        graph.add_edge(a, b, "SymbolUsage")

        cm = _make_context_manager(graph)
        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=1,
            max_files=30,
        )

        # B should appear exactly once
        paths = [p for p, _ in result]
        assert paths.count(Path(b)) == 1


class TestMultiHopBFS:
    """Test multi-hop traversal depth and edge-type rules."""

    def _build_chain_graph(self):
        """Build a linear chain: A -> B -> C -> D via SymbolUsage."""
        graph = FakeRepoGraph()
        a = str(Path("/repo/a.py").resolve())
        b = str(Path("/repo/b.py").resolve())
        c = str(Path("/repo/c.py").resolve())
        d = str(Path("/repo/d.py").resolve())
        graph.add_edge(a, b, "SymbolUsage")
        graph.add_edge(b, c, "SymbolUsage")
        graph.add_edge(c, d, "SymbolUsage")
        return graph, a, b, c, d

    def test_single_hop(self):
        """With max_hops=1, only direct neighbors are returned."""
        graph, a, b, c, d = self._build_chain_graph()
        cm = _make_context_manager(graph)

        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=1,
            max_files=30,
        )
        paths = {p for p, _ in result}

        assert Path(b) in paths
        assert Path(c) not in paths
        assert Path(d) not in paths

    def test_two_hop(self):
        """With max_hops=2, two levels of neighbors are returned."""
        graph, a, b, c, d = self._build_chain_graph()
        cm = _make_context_manager(graph)

        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=2,
            max_files=30,
        )
        paths = {p for p, _ in result}

        assert Path(b) in paths
        assert Path(c) in paths
        assert Path(d) not in paths

    def test_three_hop(self):
        """With max_hops=3, three levels are returned."""
        graph, a, b, c, d = self._build_chain_graph()
        cm = _make_context_manager(graph)

        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=3,
            max_files=30,
        )
        paths = {p for p, _ in result}

        assert Path(b) in paths
        assert Path(c) in paths
        assert Path(d) in paths


class TestImportEdgeCapping:
    """Import edges should only be followed for 1 hop."""

    def test_import_edges_capped_at_one_hop(self):
        """A chain via Import edges should only reach 1 hop deep."""
        graph = FakeRepoGraph()
        a = str(Path("/repo/a.py").resolve())
        b = str(Path("/repo/b.py").resolve())
        c = str(Path("/repo/c.py").resolve())
        graph.add_edge(a, b, "Import")
        graph.add_edge(b, c, "Import")

        cm = _make_context_manager(graph)
        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=3,
            max_files=30,
        )
        paths = {p for p, _ in result}

        assert Path(b) in paths
        # C should NOT be reached — Import edges stop after 1 hop
        assert Path(c) not in paths

    def test_symbol_usage_continues_past_import_limit(self):
        """SymbolUsage edges should continue even when Import edges stop."""
        graph = FakeRepoGraph()
        a = str(Path("/repo/a.py").resolve())
        b = str(Path("/repo/b.py").resolve())
        c = str(Path("/repo/c.py").resolve())
        graph.add_edge(a, b, "SymbolUsage")
        graph.add_edge(b, c, "SymbolUsage")

        cm = _make_context_manager(graph)
        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=2,
            max_files=30,
        )
        paths = {p for p, _ in result}

        assert Path(b) in paths
        assert Path(c) in paths


class TestDistanceDecay:
    """Verify distance-decay weighting."""

    def test_hop_weights(self):
        """Hop 1 = 1.0, hop 2 = 0.5."""
        graph = FakeRepoGraph()
        a = str(Path("/repo/a.py").resolve())
        b = str(Path("/repo/b.py").resolve())
        c = str(Path("/repo/c.py").resolve())
        graph.add_edge(a, b, "SymbolUsage")
        graph.add_edge(b, c, "SymbolUsage")

        cm = _make_context_manager(graph)
        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=2,
            max_files=30,
        )

        weight_map = {p: w for p, w in result}
        assert weight_map[Path(b)] == 1.0
        assert weight_map[Path(c)] == 0.5

    def test_hop3_weight(self):
        """Hop 3 = 0.25."""
        graph = FakeRepoGraph()
        a = str(Path("/repo/a.py").resolve())
        b = str(Path("/repo/b.py").resolve())
        c = str(Path("/repo/c.py").resolve())
        d = str(Path("/repo/d.py").resolve())
        graph.add_edge(a, b, "SymbolUsage")
        graph.add_edge(b, c, "SymbolUsage")
        graph.add_edge(c, d, "SymbolUsage")

        cm = _make_context_manager(graph)
        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=3,
            max_files=30,
        )

        weight_map = {p: w for p, w in result}
        assert weight_map[Path(d)] == 0.25


class TestMaxFilesTruncation:
    """Verify max_files cap works."""

    def test_max_files_limits_results(self):
        """Should return at most max_files results."""
        graph = FakeRepoGraph()
        a = str(Path("/repo/a.py").resolve())
        # Create 10 direct deps
        for i in range(10):
            dep = str(Path(f"/repo/dep_{i}.py").resolve())
            graph.add_edge(a, dep, "SymbolUsage")

        cm = _make_context_manager(graph)
        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed=set(),
            max_hops=1,
            max_files=5,
        )

        assert len(result) == 5

    def test_already_processed_excluded(self):
        """Files in already_processed should not appear in results."""
        graph = FakeRepoGraph()
        a = str(Path("/repo/a.py").resolve())
        b = str(Path("/repo/b.py").resolve())
        c = str(Path("/repo/c.py").resolve())
        graph.add_edge(a, b, "SymbolUsage")
        graph.add_edge(a, c, "SymbolUsage")

        cm = _make_context_manager(graph)
        result = cm._get_dependency_neighborhood(
            [Path("/repo/a.py")],
            already_processed={Path(b)},
            max_hops=1,
            max_files=30,
        )
        paths = {p for p, _ in result}

        assert Path(b) not in paths
        assert Path(c) in paths
