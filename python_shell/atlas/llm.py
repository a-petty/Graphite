import ollama
from typing import List, Dict
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages and get response."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama models."""
    
    def __init__(self, model: str = "deepseek-coder"):
        self.model = model
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat messages to Ollama and return the content.
        """
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': 0.1,  # Lower for more deterministic code-gen
                'num_predict': 4096, # Max tokens to generate
            }
        )
        return response['message']['content']


class StubClient(LLMClient):
    """Stub client for testing without a real LLM."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Return a canned response for testing. Supports multi-turn."""
        self.call_count += 1
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # Detect tool-result follow-ups and return a final answer
        if "## Tool Results" in user_msg:
            return "<think>I received the tool results. Let me summarize.</think>\n\nHere is the information you requested based on the tool results."

        if "architecture" in user_msg.lower():
            return "<think>The user wants to know about the architecture. I will read the repo map and provide a summary.</think><action>read_file('roadmap.md')</action>"

        return "I'm a stub LLM client. Real integration coming soon!"