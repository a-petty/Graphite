"""Graphite configuration management.

Provides GraphiteConfig dataclass with sensible defaults for all pipeline
settings: LLM provider/model, classification categories, tagging thresholds,
chunking parameters, and directory paths.

Use GraphiteConfig.from_toml(path) to load from a .graphite.toml file,
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
    # Conversation-specific types
    "debugging",
    "code_review",
    "architecture",
    "learning",
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
    # Conversation-specific types
    "preference",
    "goal",
    "pattern",
    "skill",
]


@dataclass
class GraphiteConfig:
    """Configuration for the Graphite extraction pipeline.

    All fields have sensible defaults. Override individual fields
    as needed, or load from a .graphite.toml file via from_toml().
    """

    # ── LLM ──
    # Supported providers: "ollama", "mlx", "openai", "anthropic"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.3:70b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Classification (Pass 2) ──
    valid_chunk_types: List[str] = field(default_factory=lambda: list(DEFAULT_CHUNK_TYPES))
    default_chunk_type: str = "background"
    classify_batch_size: int = 10

    # ── Tagging (Pass 3) ──
    valid_entity_types: List[str] = field(default_factory=lambda: list(DEFAULT_ENTITY_TYPES))
    disambiguation_auto_merge_threshold: float = 0.85
    disambiguation_review_threshold: float = 0.70
    max_tag_retries: int = 1
    circuit_breaker_failure_rate: float = 0.50
    tag_batch_size: int = 5

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

    # ── Agent Context ──
    agent_brief_max_entities: int = 8
    agent_brief_max_tokens: int = 500
    agent_full_max_entities: int = 20
    agent_full_max_events: int = 15
    agent_full_max_tokens: int = 5000
    agent_name_match_bonus: float = 0.3
    agent_pending_chunk_types: List[str] = field(
        default_factory=lambda: ["Decision", "ActionItem"]
    )

    # ── Reflection (Phase 5) ──
    decay_half_life_days: float = 30.0
    decay_archival_threshold: int = 5
    orphan_max_age_days: int = 7
    orphan_grace_period_days: int = 3  # Newly-orphaned entities younger than this are not removed
    prune_edges_below: float = 0.01  # Edges below this weight after decay are pruned
    merge_embedding_threshold: float = 0.90
    merge_alias_overlap_threshold: float = 0.80
    synthesis_max_chunks_per_entity: int = 20
    lightweight_reflection_on_ingest: bool = True

    # ── Conversation Ingestion ──
    claude_data_dir: Path = field(default_factory=lambda: Path.home() / ".claude")
    conversation_max_exchange_tokens: int = 1200
    conversation_include_tool_summaries: bool = True
    conversation_skip_tool_output: bool = True

    # ── Daemon ──
    # When the graphited daemon starts, scan the session archive and enqueue
    # any session not already in the graph. Disable for headless / CI runs.
    reconcile_on_startup: bool = True

    # When the spool's pending count reaches this many fragments, the daemon
    # auto-enqueues a batch extraction. Set to 0 to disable auto-triggering
    # (manual `flush_spool` is then the only way to drain).
    spool_size_threshold: int = 50

    # How long extracted spool fragments stick around before cleanup.
    # Failed fragments are retained indefinitely until manually triaged.
    spool_retain_days: int = 30

    # ── Paths ──
    # graph_root is the directory in which the `.graphite/graph.msgpack`
    # file is stored. Defaults to the user's home directory, giving Graphite
    # a single global graph at ~/.graphite/graph.msgpack. Overrideable via
    # the [paths] TOML section or the --graph-root CLI flag for testing.
    graph_root: Path = field(default_factory=lambda: Path.home())
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
            "agent_name_match_bonus": self.agent_name_match_bonus,
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
            "agent_brief_max_entities": self.agent_brief_max_entities,
            "agent_brief_max_tokens": self.agent_brief_max_tokens,
            "agent_full_max_entities": self.agent_full_max_entities,
            "agent_full_max_events": self.agent_full_max_events,
            "agent_full_max_tokens": self.agent_full_max_tokens,
        }
        for name, value in pos_int_fields.items():
            if value <= 0:
                errors.append(f"{name} must be a positive integer, got {value}")

        # Non-negative integers (0 is valid — e.g., orphan_max_age_days=0 means no grace period)
        nonneg_int_fields = {
            "orphan_max_age_days": self.orphan_max_age_days,
            "orphan_grace_period_days": self.orphan_grace_period_days,
        }
        for name, value in nonneg_int_fields.items():
            if value < 0:
                errors.append(f"{name} must be a non-negative integer, got {value}")

        # Non-negative floats
        if self.prune_edges_below < 0:
            errors.append(
                f"prune_edges_below must be non-negative, got {self.prune_edges_below}"
            )

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
                "Invalid GraphiteConfig:\n  " + "\n  ".join(errors)
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
    def from_toml(cls, toml_path: Path) -> "GraphiteConfig":
        """Load configuration from a .graphite.toml file.

        Reads known keys from the TOML file and uses defaults for anything
        not specified. Unknown keys are silently ignored.

        Args:
            toml_path: Path to the .graphite.toml file.

        Returns:
            A GraphiteConfig populated from the file.

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
        if "orphan_grace_period_days" in reflection:
            kwargs["orphan_grace_period_days"] = int(reflection["orphan_grace_period_days"])
        if "prune_edges_below" in reflection:
            kwargs["prune_edges_below"] = float(reflection["prune_edges_below"])
        if "merge_embedding_threshold" in reflection:
            kwargs["merge_embedding_threshold"] = float(reflection["merge_embedding_threshold"])
        if "merge_alias_overlap_threshold" in reflection:
            kwargs["merge_alias_overlap_threshold"] = float(reflection["merge_alias_overlap_threshold"])
        if "synthesis_max_chunks_per_entity" in reflection:
            kwargs["synthesis_max_chunks_per_entity"] = int(reflection["synthesis_max_chunks_per_entity"])
        if "lightweight_reflection_on_ingest" in reflection:
            kwargs["lightweight_reflection_on_ingest"] = bool(reflection["lightweight_reflection_on_ingest"])

        # [agent] section
        agent = data.get("agent", {})
        if "brief_max_entities" in agent:
            kwargs["agent_brief_max_entities"] = int(agent["brief_max_entities"])
        if "brief_max_tokens" in agent:
            kwargs["agent_brief_max_tokens"] = int(agent["brief_max_tokens"])
        if "full_max_entities" in agent:
            kwargs["agent_full_max_entities"] = int(agent["full_max_entities"])
        if "full_max_events" in agent:
            kwargs["agent_full_max_events"] = int(agent["full_max_events"])
        if "full_max_tokens" in agent:
            kwargs["agent_full_max_tokens"] = int(agent["full_max_tokens"])
        if "name_match_bonus" in agent:
            kwargs["agent_name_match_bonus"] = float(agent["name_match_bonus"])
        if "pending_chunk_types" in agent:
            kwargs["agent_pending_chunk_types"] = list(agent["pending_chunk_types"])

        # [conversation] section
        conversation = data.get("conversation", {})
        if "claude_data_dir" in conversation:
            kwargs["claude_data_dir"] = Path(conversation["claude_data_dir"])
        if "max_exchange_tokens" in conversation:
            kwargs["conversation_max_exchange_tokens"] = int(conversation["max_exchange_tokens"])
        if "include_tool_summaries" in conversation:
            kwargs["conversation_include_tool_summaries"] = bool(conversation["include_tool_summaries"])
        if "skip_tool_output" in conversation:
            kwargs["conversation_skip_tool_output"] = bool(conversation["skip_tool_output"])

        # [paths] section
        paths = data.get("paths", {})
        if "memory_root" in paths:
            kwargs["memory_root"] = Path(paths["memory_root"])
        if "graph_root" in paths:
            kwargs["graph_root"] = Path(paths["graph_root"]).expanduser()

        return cls(**kwargs)
