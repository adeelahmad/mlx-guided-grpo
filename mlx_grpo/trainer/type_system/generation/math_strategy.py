"""
Math Generation Strategy - Auto-discovered for type='math'
===========================================================

Extend BaseGenerationStrategy and override methods you need.
All methods have sensible defaults, only override what's different!
"""

from ..auto_discovery import BaseGenerationStrategy

__all__ = ["MathGenerationStrategy"]


class MathGenerationStrategy(BaseGenerationStrategy):
    """Generation strategy for math problems.

    Auto-discovered for type='math'.
    Just override the methods you need!
    """

    def get_max_length(self) -> int:
        """Math problems need longer generations for reasoning."""
        return 1024

    def get_temperature(self) -> float:
        """Slightly higher temperature for diverse reasoning paths."""
        return 0.85

    def use_two_phase(self) -> bool:
        """Enable two-phase: thinking + answer."""
        return True

    def enforce_thinking(self) -> bool:
        """Require <think> tags for step-by-step reasoning."""
        return True

    # That's it! Other methods use base defaults
