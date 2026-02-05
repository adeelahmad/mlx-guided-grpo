"""
Exam Generation Strategy - Auto-discovered for type='exam'
===========================================================

For exam-style problems requiring deep reasoning.
"""

from ..auto_discovery import BaseGenerationStrategy

__all__ = ["ExamGenerationStrategy"]


class ExamGenerationStrategy(BaseGenerationStrategy):
    """Generation strategy for exam problems.

    Auto-discovered for type='exam'.
    Optimized for complex, multi-step reasoning.
    """

    def get_max_length(self) -> int:
        """Exam problems need long reasoning."""
        return 1536

    def get_temperature(self) -> float:
        """Slightly higher for diverse approaches."""
        return 0.85

    def get_top_p(self) -> float:
        """Standard nucleus sampling."""
        return 0.95

    def use_two_phase(self) -> bool:
        """Enable two-phase for thinking + answer."""
        return True

    def enforce_thinking(self) -> bool:
        """Require explicit reasoning."""
        return True
