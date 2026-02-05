"""
Rewards Package - Auto-discovery enabled
=========================================

Just drop in a new {Type}Reward class and it's automatically discovered!

Example:
    # Create summarization_reward.py:
    class SummarizationReward(BaseReward):
        def compute(self, prompts, completions, answers, types=None):
            return [0.8] * len(completions)

    # Automatically works for type='summarization'!
"""

from .math_reward import MathReward
from .tool_reward import ToolReward, ToolCallReward

__all__ = [
    "MathReward",
    "ToolReward",
    "ToolCallReward",
]
