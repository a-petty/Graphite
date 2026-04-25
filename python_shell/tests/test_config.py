"""Tests for GraphiteConfig — focusing on the PR 4 graph_root addition."""

from pathlib import Path

import pytest

from graphite.config import GraphiteConfig


class TestGraphRootDefault:
    def test_default_graph_root_is_home(self):
        config = GraphiteConfig()
        assert config.graph_root == Path.home()

    def test_graph_root_overrideable_at_construction(self, tmp_path: Path):
        config = GraphiteConfig(graph_root=tmp_path)
        assert config.graph_root == tmp_path


class TestGraphRootFromToml:
    def test_toml_paths_graph_root_expands_tilde(self, tmp_path: Path):
        toml = tmp_path / "graphite.toml"
        toml.write_text('[paths]\ngraph_root = "~/custom_graphite"\n')

        config = GraphiteConfig.from_toml(toml)
        expected = Path.home() / "custom_graphite"
        assert config.graph_root == expected

    def test_toml_paths_graph_root_accepts_absolute(self, tmp_path: Path):
        target = tmp_path / "elsewhere"
        toml = tmp_path / "graphite.toml"
        toml.write_text(f'[paths]\ngraph_root = "{target}"\n')

        config = GraphiteConfig.from_toml(toml)
        assert config.graph_root == target

    def test_toml_without_graph_root_falls_back_to_default(self, tmp_path: Path):
        toml = tmp_path / "graphite.toml"
        toml.write_text('[llm]\nmodel = "foo"\n')

        config = GraphiteConfig.from_toml(toml)
        assert config.graph_root == Path.home()
