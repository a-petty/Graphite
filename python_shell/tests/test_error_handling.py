# python_shell/tests/test_error_handling.py
#
# Error handling tests for Graphite agent.

from pathlib import Path
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from graphite.agent import GraphiteAgent


def test_agent_initializes_with_empty_graph():
    """
    Tests that the agent initializes cleanly even when no knowledge graph exists yet.
    No exceptions should be raised during construction.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        # Should not raise — agent is created lazily, initialize() loads graph
        agent = GraphiteAgent(project_root=temp_dir)
        assert agent.project_root == temp_dir.resolve()
        assert agent.kg is None  # Not loaded until initialize()


def test_agent_query_without_initialize_works():
    """
    Tests that calling query() before initialize() works if context_manager
    is mocked (the agent design allows query without full init for testing).
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        agent = GraphiteAgent(project_root=temp_dir)
        agent.context_manager = MagicMock()
        agent.context_manager.assemble_context.return_value = ""

        # Should not raise — stub LLM returns text, no crash
        agent.query("hello")
        assert len(agent.conversation_history) == 0  # query() doesn't append to history