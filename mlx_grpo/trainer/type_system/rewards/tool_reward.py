"""
Tool Calling Reward - Auto-discovered for type='tool' or 'tool_call'
====================================================================

Two ways to define rewards:
1. Class-based (extend BaseReward)
2. Decorator-based (use @reward decorator)
"""

from typing import Optional
from ..auto_discovery import BaseReward

__all__ = ["ToolReward", "ToolCallReward"]


# =============================================================================
# METHOD 1: Class-based (Auto-discovered as "tool")
# =============================================================================

class ToolReward(BaseReward):
    """Reward for tool/function calling.

    Auto-discovered for type='tool'.
    """

    def compute(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[str]] = None,
    ) -> list[float]:
        """Check if function calls match."""
        from ...tool_calling_reward import extract_function_calls, compare_function_calls

        scores = []

        for completion, answer in zip(completions, answers):
            pred_calls = extract_function_calls(completion)
            exp_calls = extract_function_calls(answer)

            comparison = compare_function_calls(pred_calls, exp_calls)
            scores.append(comparison['overall'])

        return scores

    def get_weight(self) -> float:
        return 0.8


# =============================================================================
# METHOD 2: Aliased version for 'tool_call' type
# =============================================================================

class ToolCallReward(ToolReward):
    """Same as ToolReward but for type='tool_call'.

    Auto-discovered for type='tool_call'.
    """
    pass


# =============================================================================
# METHOD 3: Decorator-based (Alternative pattern)
# =============================================================================

# This could be used with a @reward decorator instead:
#
# from ..decorators import reward
#
# @reward(type="tool")
# def tool_calling_reward(prompts, completions, answers, types=None):
#     """Simple function-based reward."""
#     from ...tool_calling_reward import extract_function_calls, compare_function_calls
#
#     scores = []
#     for completion, answer in zip(completions, answers):
#         pred = extract_function_calls(completion)
#         exp = extract_function_calls(answer)
#         scores.append(compare_function_calls(pred, exp)['overall'])
#     return scores
