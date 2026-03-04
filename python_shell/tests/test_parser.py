"""Tests for CortexAgent._parse_response and related parser methods."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


def _make_agent():
    """Create a CortexAgent with mocked dependencies for parser testing."""
    with tempfile.TemporaryDirectory() as tmp:
        with patch("cortex.agent.RepoGraph"), \
             patch("cortex.agent.EmbeddingManager"), \
             patch("cortex.agent.ContextManager"):
            from cortex.agent import CortexAgent
            agent = CortexAgent(project_root=Path(tmp))
            return agent


class TestWellFormedParsing:
    """Test parsing of well-formed <think> and <action> responses."""

    def test_basic_think_and_action(self):
        agent = _make_agent()
        response = "<think>I need to read the file</think><action>read_file('src/main.py')</action>"
        thoughts, actions, plain = agent._parse_response(response)

        assert thoughts == "I need to read the file"
        assert len(actions) == 1
        assert actions[0]['tool'] == 'read_file'
        assert actions[0]['args'] == ['src/main.py']

    def test_multiple_actions(self):
        agent = _make_agent()
        response = (
            "<think>Read two files</think>"
            "<action>read_file('a.py')</action>"
            "<action>read_file('b.py')</action>"
        )
        thoughts, actions, plain = agent._parse_response(response)

        assert len(actions) == 2
        assert actions[0]['args'] == ['a.py']
        assert actions[1]['args'] == ['b.py']

    def test_write_file_two_args(self):
        agent = _make_agent()
        response = "<think>Writing</think><action>write_file('out.py', 'print(1)')</action>"
        thoughts, actions, plain = agent._parse_response(response)

        assert actions[0]['tool'] == 'write_file'
        assert actions[0]['args'] == ['out.py', 'print(1)']


class TestUnclosedTags:
    """Test handling of unclosed tags (common with local models)."""

    def test_unclosed_think(self):
        agent = _make_agent()
        response = "<think>I need to think about this"
        thoughts, actions, plain = agent._parse_response(response)

        assert thoughts == "I need to think about this"

    def test_unclosed_action(self):
        agent = _make_agent()
        response = "<action>read_file('main.py')"
        thoughts, actions, plain = agent._parse_response(response)

        assert len(actions) == 1
        assert actions[0]['tool'] == 'read_file'


class TestWhitespaceVariants:
    """Test normalization of whitespace in tags."""

    def test_spaces_in_tags(self):
        agent = _make_agent()
        response = "< think >Reasoning< /think >< action >read_file('x.py')< /action >"
        thoughts, actions, plain = agent._parse_response(response)

        assert thoughts == "Reasoning"
        assert len(actions) == 1


class TestMalformedArgs:
    """Test handling of malformed arguments."""

    def test_unparseable_args_falls_back_to_regex(self):
        agent = _make_agent()
        # Malformed: unbalanced quotes that ast.literal_eval can't handle,
        # but regex can extract the quoted string
        response = "<action>read_file('some/path.py'  extra_junk)</action>"
        thoughts, actions, plain = agent._parse_response(response)

        # The regex fallback should extract the quoted path
        assert len(actions) == 1
        assert actions[0]['args'] == ['some/path.py']

    def test_no_args(self):
        agent = _make_agent()
        response = "<action>list_directory()</action>"
        thoughts, actions, plain = agent._parse_response(response)

        assert len(actions) == 1
        assert actions[0]['tool'] == 'list_directory'
        assert actions[0]['args'] == []


class TestFallbackParsing:
    """Test fallback parsing when no <action> tags are present."""

    def test_known_tool_in_plain_text(self):
        agent = _make_agent()
        response = "I think you should read_file('config.py') to understand the config."
        thoughts, actions, plain = agent._parse_response(response)

        assert len(actions) == 1
        assert actions[0]['tool'] == 'read_file'
        assert actions[0]['args'] == ['config.py']

    def test_no_tools_in_plain_text(self):
        agent = _make_agent()
        response = "I don't think any tool calls are needed here."
        thoughts, actions, plain = agent._parse_response(response)

        assert len(actions) == 0
        assert "tool calls" in plain


class TestPlainText:
    """Test plain text extraction."""

    def test_text_outside_tags(self):
        agent = _make_agent()
        response = "Hello world <think>thinking</think> more text <action>read_file('x.py')</action> final"
        thoughts, actions, plain = agent._parse_response(response)

        assert thoughts == "thinking"
        assert len(actions) == 1
        assert "Hello world" in plain
        assert "final" in plain

    def test_empty_response(self):
        agent = _make_agent()
        response = ""
        thoughts, actions, plain = agent._parse_response(response)

        assert thoughts is None
        assert len(actions) == 0
        assert plain == ""


class TestParseSingleAction:
    """Test _parse_single_action directly."""

    def test_valid_action(self):
        agent = _make_agent()
        result = agent._parse_single_action("read_file('test.py')")
        assert result == {'tool': 'read_file', 'args': ['test.py']}

    def test_no_match(self):
        agent = _make_agent()
        result = agent._parse_single_action("just some text")
        assert result is None

    def test_empty_args(self):
        agent = _make_agent()
        result = agent._parse_single_action("generate_repository_map()")
        assert result == {'tool': 'generate_repository_map', 'args': []}
