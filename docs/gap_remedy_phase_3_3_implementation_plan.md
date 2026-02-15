# Phase 3.3 Implementation Plan: Stub LLM Integration

**Goal:** Implement the necessary interfaces and stub clients to prepare for LLM integration, fulfilling the requirements of Phase 3.3 from `1_29_26_implementation_plan.md`.

---

## 1. LLM Client Abstraction

The first step is to define a generic interface for interacting with Large Language Models. This ensures that we can easily swap different models or services in the future.

### 1.1. Create LLM Interface File

**File to Create:** `python_shell/atlas/llm.py`

### 1.2. Implement `LLMClient` Abstract Base Class

This class will define the contract for all LLM clients.

**File:** `python_shell/atlas/llm.py`

**Content:**
```python
from typing import List, Dict
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages and get response."""
        pass
```

---

## 2. Stub and Skeleton LLM Clients

With the interface defined, we can create clients for testing and future use.

### 2.1. Implement `StubClient` for Testing

This client will return canned responses for end-to-end testing without needing a real LLM.

**File:** `python_shell/atlas/llm.py`

**Action:** Add the `StubClient` class.

```python
class StubClient(LLMClient):
    """Stub client for testing without a real LLM."""
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Return a canned response for testing."""
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        
        if "architecture" in user_msg.lower():
            return "Based on the repository map, this appears to be a Python project with a core module and several utilities."
        
        return "I'm a stub LLM client. Real integration coming soon!"
```

### 2.2. Implement `OllamaClient` Skeleton

This class will be the placeholder for the actual Ollama integration.

**File:** `python_shell/atlas/llm.py`

**Action:** Add the `OllamaClient` class.

```python
class OllamaClient(LLMClient):
    """Client for local Ollama models."""
    
    def __init__(self, model: str = "deepseek-r1:32b-qwen-distill-q4_K_M"):
        self.model = model
        # TODO: Add ollama library when ready
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat messages to Ollama.
        
        For now, this is a stub that returns a placeholder.
        """
        # TODO: Implement actual Ollama integration
        return "[LLM Response - Not Yet Implemented]"
```

---

## 3. Agent Integration

Finally, we'll integrate the new LLM client structure into the `AtlasAgent`.

### 3.1. Update `AtlasAgent` Initialization

**File:** `python_shell/atlas/agent.py`

**Action:** Modify the `__init__` method to include the LLM client.

```python
# Add to imports
from .llm import StubClient, OllamaClient

# In AtlasAgent class
def __init__(self, project_root: Path, use_real_llm: bool = False):
    # ... existing init ...
    
    # LLM client (stub by default)
    if use_real_llm:
        self.llm = OllamaClient()
    else:
        self.llm = StubClient()
```

### 3.2. Update `AtlasAgent` Query Method

**File:** `python_shell/atlas/agent.py`

**Action:** Update the `query` method to use the `ContextManager` and the LLM client.

```python
def query(self, user_input: str) -> str:
    """Process a user query with LLM."""
    
    # Build context
    context = self.context_manager.build_context(
        anchor_files=[],  # TODO: Extract from query
        repo_graph=self.repo_graph,
        user_query=user_input,
        include_map=True
    )
    
    # Create messages
    messages = [
        {
            "role": "system",
            "content": f"You are Atlas, an AI coding agent with deep knowledge of this repository.\n\n{context}"
        },
        {
            "role": "user",
            "content": user_input
        }
    ]
    
    # Get LLM response
    response = self.llm.chat(messages)
    return response
```

---

## 4. Execution and Verification

1.  Create the `python_shell/atlas/llm.py` file with the content from steps 1 and 2.
2.  Apply the code changes to `python_shell/atlas/agent.py` from step 3.
3.  Run the agent and verify that the `query` method now returns a stubbed LLM response.
4.  Ensure that the `use_real_llm` flag can be used to switch between the `StubClient` and the `OllamaClient` skeleton.

This plan provides a clear path to setting up the LLM integration framework, allowing for end-to-end testing of the context assembly and query flow without requiring a live LLM connection.