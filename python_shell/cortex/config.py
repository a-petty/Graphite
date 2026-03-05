"""Cortex configuration management.

Provides CortexConfig dataclass with sensible defaults for all pipeline
settings: LLM provider/model, classification categories, tagging thresholds,
chunking parameters, and directory paths.

TOML loading deferred to Phase 4 — for now, instantiate with overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Default valid chunk types for classification (Pass 2)
DEFAULT_CHUNK_TYPES = [
    "decision",
    "discussion",
    "action_item",
    "status_update",
    "preference",
    "background",
    "filler",
]

# Default valid entity types for tagging (Pass 3)
DEFAULT_ENTITY_TYPES = [
    "person",
    "project",
    "technology",
    "organization",
    "location",
    "decision",
    "concept",
]


@dataclass
class CortexConfig:
    """Configuration for the Cortex extraction pipeline.

    All fields have sensible defaults. Override individual fields
    as needed — no TOML file required (Phase 4 adds TOML loading).
    """

    # ── LLM ──
    llm_provider: str = "ollama"
    llm_model: str = "llama3.3:70b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Classification (Pass 2) ──
    valid_chunk_types: List[str] = field(default_factory=lambda: list(DEFAULT_CHUNK_TYPES))
    default_chunk_type: str = "background"

    # ── Tagging (Pass 3) ──
    valid_entity_types: List[str] = field(default_factory=lambda: list(DEFAULT_ENTITY_TYPES))
    disambiguation_auto_merge_threshold: float = 0.85
    disambiguation_review_threshold: float = 0.70
    max_tag_retries: int = 1
    circuit_breaker_failure_rate: float = 0.50

    # ── Chunking (Pass 1) ──
    max_chunk_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # ── Context Assembly ──
    tier1_budget_pct: float = 0.10   # Knowledge Map
    tier2_budget_pct: float = 0.60   # Evidence Chunks
    # Tier 3 gets the remaining budget

    anchor_count_min: int = 3
    anchor_count_max: int = 10

    neighborhood_max_hops_sparse: int = 3   # graph density < 3.0
    neighborhood_max_hops_dense: int = 2    # graph density >= 3.0
    neighborhood_max_entities: int = 40

    chunk_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "Decision": 1.0,
        "ActionItem": 0.9,
        "StatusUpdate": 0.7,
        "Preference": 0.6,
        "Discussion": 0.5,
        "Background": 0.3,
    })

    similarity_weight: float = 0.80
    pagerank_weight: float = 0.20

    # ── Paths ──
    memory_root: Path = field(default_factory=lambda: Path("memory"))
    prompts_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "extraction" / "prompts"
    )

    def get_prompt(self, name: str) -> str:
        """Load a prompt template by name (without extension).

        Args:
            name: Prompt filename stem, e.g. "classify" or "tag".

        Returns:
            The prompt template text.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """
        path = self.prompts_dir / f"{name}.txt"
        return path.read_text()
