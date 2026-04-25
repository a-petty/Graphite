"""Tests for LLM client implementations."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from graphite.llm import LLMClient, StubClient


class TestMLXClientImportGuard:
    """MLXClient should raise a helpful ImportError when mlx-lm is missing."""

    def test_import_error_when_mlx_lm_missing(self):
        """Attempting to create MLXClient without mlx-lm raises ImportError."""
        # Ensure mlx_lm is not importable
        with patch.dict("sys.modules", {"mlx_lm": None, "mlx_lm.sample_utils": None}):
            from graphite.llm import MLXClient
            with pytest.raises(ImportError) as exc_info:
                MLXClient()
            assert "mlx-lm is not installed" in str(exc_info.value)

    def test_import_error_mentions_install_command(self):
        """Error message tells the user how to install."""
        with patch.dict("sys.modules", {"mlx_lm": None, "mlx_lm.sample_utils": None}):
            from graphite.llm import MLXClient
            with pytest.raises(ImportError) as exc_info:
                MLXClient()
            assert "pip install 'graphite[mlx]'" in str(exc_info.value)


class TestMLXClientInit:
    """MLXClient should load model and tokenizer via mlx_lm.load()."""

    def test_load_called_with_model_id(self):
        mock_mlx_lm = MagicMock()
        mock_sample_utils = MagicMock()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
        mock_mlx_lm.generate = MagicMock()
        mock_sample_utils.make_sampler.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "mlx_lm": mock_mlx_lm,
            "mlx_lm.sample_utils": mock_sample_utils,
        }):
            from graphite.llm import MLXClient
            client = MLXClient(model="mlx-community/test-model")

        mock_mlx_lm.load.assert_called_once_with("mlx-community/test-model")
        assert client.model is mock_model
        assert client.tokenizer is mock_tokenizer


class TestMLXClientChat:
    """MLXClient.chat() should format messages and call generate()."""

    def _make_client(self):
        """Create an MLXClient with fully mocked mlx_lm."""
        mock_mlx_lm = MagicMock()
        mock_sample_utils = MagicMock()

        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_sampler = MagicMock()
        mock_mlx_lm.load.return_value = (self.mock_model, self.mock_tokenizer)
        mock_mlx_lm.generate = MagicMock()
        mock_sample_utils.make_sampler.return_value = self.mock_sampler

        with patch.dict("sys.modules", {
            "mlx_lm": mock_mlx_lm,
            "mlx_lm.sample_utils": mock_sample_utils,
        }):
            from graphite.llm import MLXClient
            client = MLXClient()

        self.mock_generate = mock_mlx_lm.generate
        return client

    def test_apply_chat_template_called_correctly(self):
        client = self._make_client()
        messages = [{"role": "user", "content": "Hello"}]
        self.mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        self.mock_generate.return_value = "response"

        client.chat(messages)

        # ``enable_thinking=False`` is required for Qwen3 to emit clean
        # JSON without <think>...</think> blocks. Older models ignore it.
        self.mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def test_thinking_mode_disabled_for_qwen3_compatibility(self):
        """Regression guard: removing ``enable_thinking=False`` would
        break the Graphite tagger pipeline on Qwen3 models because
        <think>...</think> blocks would land inside the JSON output
        the tagger expects to parse."""
        client = self._make_client()
        self.mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        self.mock_generate.return_value = "[]"

        client.chat([{"role": "user", "content": "tag entities"}])

        kwargs = self.mock_tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs.get("enable_thinking") is False

    def test_generate_called_with_correct_args(self):
        client = self._make_client()
        messages = [{"role": "user", "content": "Hello"}]
        self.mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        self.mock_generate.return_value = "response"

        client.chat(messages)

        self.mock_generate.assert_called_once_with(
            self.mock_model, self.mock_tokenizer,
            prompt="<formatted>", max_tokens=4096, sampler=self.mock_sampler,
        )

    def test_returns_generate_output(self):
        client = self._make_client()
        messages = [{"role": "user", "content": "Hello"}]
        self.mock_tokenizer.apply_chat_template.return_value = "<formatted>"
        self.mock_generate.return_value = "the generated text"

        result = client.chat(messages)

        assert result == "the generated text"


class TestMLXClientConformsToABC:
    """MLXClient should be a proper LLMClient subclass."""

    def test_isinstance_check(self):
        mock_mlx_lm = MagicMock()
        mock_sample_utils = MagicMock()
        mock_mlx_lm.load.return_value = (MagicMock(), MagicMock())
        mock_mlx_lm.generate = MagicMock()
        mock_sample_utils.make_sampler.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "mlx_lm": mock_mlx_lm,
            "mlx_lm.sample_utils": mock_sample_utils,
        }):
            from graphite.llm import MLXClient
            client = MLXClient()

        assert isinstance(client, LLMClient)


class TestStubClient:
    """Regression tests for StubClient."""

    def test_call_count_increments(self):
        client = StubClient()
        assert client.call_count == 0
        client.chat([{"role": "user", "content": "hi"}])
        assert client.call_count == 1
        client.chat([{"role": "user", "content": "hi again"}])
        assert client.call_count == 2

    def test_returns_string(self):
        client = StubClient()
        result = client.chat([{"role": "user", "content": "test"}])
        assert isinstance(result, str)
