"""Tests for Phase 0-6 bug fixes.

Covers:
- Bug #14: UTC timestamp parsing in structural_parser
- Bug #8: GraphiteConfig validation
"""

import pytest

from graphite.config import GraphiteConfig
from graphite.extraction.structural_parser import StructuralParser


# ═══════════════════════════════════════════════════════════════════════════════
# Bug #14: UTC timestamp parsing
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimestampParsingUTC:
    """Verify _parse_date_string always returns UTC timestamps."""

    def setup_method(self):
        self.parser = StructuralParser()

    def test_parse_date_string_utc(self):
        """ISO date '2024-01-01' should parse to midnight UTC = 1704067200."""
        result = self.parser._parse_date_string("2024-01-01")
        assert result == 1704067200, (
            f"Expected 1704067200 (midnight UTC), got {result}"
        )

    def test_parse_date_written_utc(self):
        """Written date 'January 1, 2024' should parse to midnight UTC = 1704067200."""
        result = self.parser._parse_date_string("January 1, 2024")
        assert result == 1704067200, (
            f"Expected 1704067200 (midnight UTC), got {result}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Bug #8: Config validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    """Verify GraphiteConfig __post_init__ catches invalid values."""

    def test_config_default_values_pass_validation(self):
        """Default GraphiteConfig should pass validation without errors."""
        config = GraphiteConfig()
        assert config.tier1_budget_pct == 0.10

    def test_config_validates_budget_sum(self):
        """Budget percentages summing > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="tier1_budget_pct.*tier2_budget_pct.*<= 1.0"):
            GraphiteConfig(tier1_budget_pct=0.5, tier2_budget_pct=0.6)

    def test_config_validates_threshold_range(self):
        """Values outside [0.0, 1.0] should raise ValueError."""
        with pytest.raises(ValueError, match="must be in.*0.0.*1.0"):
            GraphiteConfig(similarity_weight=1.5)

    def test_config_validates_positive_integers(self):
        """Non-positive integers should raise ValueError."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            GraphiteConfig(max_chunk_tokens=0)
