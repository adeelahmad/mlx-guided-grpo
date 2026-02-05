"""
Tool Calling Generation Strategy
=================================

For tool/function calling: short, structured outputs.
"""

from ..auto_discovery import BaseGenerationStrategy

__all__ = ["ToolGenerationStrategy", "ToolCallGenerationStrategy"]


class ToolGenerationStrategy(BaseGenerationStrategy):
    """Generation for tool calling.

    Auto-discovered for type='tool'.
    """

    def get_max_length(self) -> int:
        """Function calls are short."""
        return 256

    def get_temperature(self) -> float:
        """Lower temperature for structured output."""
        return 0.7

    def use_two_phase(self) -> bool:
        """No thinking phase needed."""
        return False

    def enforce_thinking(self) -> bool:
        """Direct function call generation."""
        return False


# Alias for 'tool_call' type
class ToolCallGenerationStrategy(ToolGenerationStrategy):
    """Same strategy for type='tool_call'."""
    pass
