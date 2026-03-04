# python_shell/tests/test_error_handling.py
#
# Error handling tests for Cortex agent.
# Tree-sitter-dependent tests (ParseError from syntax checking) were removed
# in Phase 0b. New error handling tests will be added as the extraction
# pipeline is built in Phase 2.

from pathlib import Path
import pytest
import tempfile

from cortex.agent import CortexAgent


def test_agent_handles_missing_file_gracefully():
    """
    Tests that the agent handles a modification event for a file
    that doesn't exist on disk without crashing.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        agent = CortexAgent(project_root=temp_dir)

        # Simulate modification event for a non-existent file
        nonexistent = temp_dir / "does_not_exist.py"
        # Should not raise — _safe_read_file returns None for missing files
        agent._handle_file_modified(nonexistent)