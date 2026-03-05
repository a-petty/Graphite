"""Cortex configuration management.

Provides CortexConfig dataclass with sensible defaults for all pipeline
settings: LLM provider/model, classification categories, tagging thresholds,
chunking parameters, and directory paths.

Use CortexConfig.from_toml(path) to load from a .cortex.toml file,
or instantiate directly with keyword overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


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
    as needed, or load from a .cortex.toml file via from_toml().
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

    # ── Reflection (Phase 5) ──
    decay_half_life_days: float = 30.0
    decay_archival_threshold: int = 5
    orphan_max_age_days: int = 7
    merge_embedding_threshold: float = 0.90
    merge_alias_overlap_threshold: float = 0.80
    synthesis_max_chunks_per_entity: int = 20
    lightweight_reflection_on_ingest: bool = True

    # ── Paths ──
    memory_root: Path = field(default_factory=lambda: Path("memory"))
    prompts_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "extraction" / "prompts"
    )

    def __post_init__(self):
        """Validate configuration values."""
        errors = []

        # Float fields in [0.0, 1.0]
        unit_fields = {
            "tier1_budget_pct": self.tier1_budget_pct,
            "tier2_budget_pct": self.tier2_budget_pct,
            "similarity_weight": self.similarity_weight,
            "pagerank_weight": self.pagerank_weight,
            "disambiguation_auto_merge_threshold": self.disambiguation_auto_merge_threshold,
            "disambiguation_review_threshold": self.disambiguation_review_threshold,
            "circuit_breaker_failure_rate": self.circuit_breaker_failure_rate,
            "merge_embedding_threshold": self.merge_embedding_threshold,
            "merge_alias_overlap_threshold": self.merge_alias_overlap_threshold,
        }
        for name, value in unit_fields.items():
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name} must be in [0.0, 1.0], got {value}")

        # Budget sum
        if self.tier1_budget_pct + self.tier2_budget_pct > 1.0:
            errors.append(
                f"tier1_budget_pct + tier2_budget_pct must be <= 1.0, "
                f"got {self.tier1_budget_pct} + {self.tier2_budget_pct} = "
                f"{self.tier1_budget_pct + self.tier2_budget_pct}"
            )

        # Positive integers (must be > 0)
        pos_int_fields = {
            "llm_max_tokens": self.llm_max_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "anchor_count_min": self.anchor_count_min,
            "anchor_count_max": self.anchor_count_max,
            "neighborhood_max_hops_sparse": self.neighborhood_max_hops_sparse,
            "neighborhood_max_hops_dense": self.neighborhood_max_hops_dense,
            "neighborhood_max_entities": self.neighborhood_max_entities,
            "synthesis_max_chunks_per_entity": self.synthesis_max_chunks_per_entity,
        }
        for name, value in pos_int_fields.items():
            if value <= 0:
                errors.append(f"{name} must be a positive integer, got {value}")

        # Non-negative integers (0 is valid — e.g., orphan_max_age_days=0 means no grace period)
        nonneg_int_fields = {
            "orphan_max_age_days": self.orphan_max_age_days,
        }
        for name, value in nonneg_int_fields.items():
            if value < 0:
                errors.append(f"{name} must be a non-negative integer, got {value}")

        # Positive floats
        if self.decay_half_life_days <= 0:
            errors.append(
                f"decay_half_life_days must be positive, got {self.decay_half_life_days}"
            )

        # Logical constraints
        if self.anchor_count_min > self.anchor_count_max:
            errors.append(
                f"anchor_count_min ({self.anchor_count_min}) must be "
                f"<= anchor_count_max ({self.anchor_count_max})"
            )
        if self.disambiguation_review_threshold > self.disambiguation_auto_merge_threshold:
            errors.append(
                f"disambiguation_review_threshold ({self.disambiguation_review_threshold}) "
                f"must be <= disambiguation_auto_merge_threshold "
                f"({self.disambiguation_auto_merge_threshold})"
            )

        if errors:
            raise ValueError(
                "Invalid CortexConfig:\n  " + "\n  ".join(errors)
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

    @classmethod
    def from_toml(cls, toml_path: Path) -> "CortexConfig":
        """Load configuration from a .cortex.toml file.

        Reads known keys from the TOML file and uses defaults for anything
        not specified. Unknown keys are silently ignored.

        Args:
            toml_path: Path to the .cortex.toml file.

        Returns:
            A CortexConfig populated from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                logger.warning(
                    "Neither tomllib nor tomli available; using default config."
                )
                return cls()

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        kwargs: Dict = {}

        # [llm] section
        llm = data.get("llm", {})
        if "provider" in llm:
            kwargs["llm_provider"] = llm["provider"]
        if "model" in llm:
            kwargs["llm_model"] = llm["model"]
        if "temperature" in llm:
            kwargs["llm_temperature"] = float(llm["temperature"])
        if "max_tokens" in llm:
            kwargs["llm_max_tokens"] = int(llm["max_tokens"])

        # [extraction] section
        extraction = data.get("extraction", {})
        if "auto_merge_threshold" in extraction:
            kwargs["disambiguation_auto_merge_threshold"] = float(
                extraction["auto_merge_threshold"]
            )
        if "review_threshold" in extraction:
            kwargs["disambiguation_review_threshold"] = float(
                extraction["review_threshold"]
            )
        if "max_chunk_tokens" in extraction:
            kwargs["max_chunk_tokens"] = int(extraction["max_chunk_tokens"])

        # [context] section
        context = data.get("context", {})
        if "tier1_budget_pct" in context:
            kwargs["tier1_budget_pct"] = float(context["tier1_budget_pct"])
        if "tier2_budget_pct" in context:
            kwargs["tier2_budget_pct"] = float(context["tier2_budget_pct"])
        if "similarity_weight" in context:
            kwargs["similarity_weight"] = float(context["similarity_weight"])
        if "pagerank_weight" in context:
            kwargs["pagerank_weight"] = float(context["pagerank_weight"])

        # [reflection] section
        reflection = data.get("reflection", {})
        if "decay_half_life_days" in reflection:
            kwargs["decay_half_life_days"] = float(reflection["decay_half_life_days"])
        if "decay_archival_threshold" in reflection:
            kwargs["decay_archival_threshold"] = int(reflection["decay_archival_threshold"])
        if "orphan_max_age_days" in reflection:
            kwargs["orphan_max_age_days"] = int(reflection["orphan_max_age_days"])
        if "merge_embedding_threshold" in reflection:
            kwargs["merge_embedding_threshold"] = float(reflection["merge_embedding_threshold"])
        if "merge_alias_overlap_threshold" in reflection:
            kwargs["merge_alias_overlap_threshold"] = float(reflection["merge_alias_overlap_threshold"])
        if "synthesis_max_chunks_per_entity" in reflection:
            kwargs["synthesis_max_chunks_per_entity"] = int(reflection["synthesis_max_chunks_per_entity"])
        if "lightweight_reflection_on_ingest" in reflection:
            kwargs["lightweight_reflection_on_ingest"] = bool(reflection["lightweight_reflection_on_ingest"])

        # [paths] section
        paths = data.get("paths", {})
        if "memory_root" in paths:
            kwargs["memory_root"] = Path(paths["memory_root"])

        return cls(**kwargs)
