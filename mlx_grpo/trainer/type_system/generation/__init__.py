"""
Generation Strategies Package
==============================

Auto-discovery enabled for generation strategies.

Example:
    # Create code_strategy.py:
    class CodeGenerationStrategy(BaseGenerationStrategy):
        def get_max_length(self):
            return 2048

    # Automatically works for type='code'!
"""

from .math_strategy import MathGenerationStrategy
from .tool_strategy import ToolGenerationStrategy, ToolCallGenerationStrategy

__all__ = [
    "MathGenerationStrategy",
    "ToolGenerationStrategy",
    "ToolCallGenerationStrategy",
]
