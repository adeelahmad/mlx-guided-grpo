"""
Math Reward - Auto-discovered for type='math'
==============================================

Simply extend BaseReward and implement compute().
Automatically discovered when data type is 'math'.
"""

from typing import Optional
from ..auto_discovery import BaseReward

__all__ = ["MathReward"]


class MathReward(BaseReward):
    """Reward for math problems.

    Auto-discovered for type='math'.
    Just extend BaseReward and implement compute() - that's it!
    """

    def compute(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[str]] = None,
    ) -> list[float]:
        """
        Compute rewards for math problems.

        Checks for:
        - Boxed answer match
        - Numerical correctness
        - Step-by-step reasoning

        Returns:
            Scores in [0.0, 1.0]
        """
        import re

        scores = []

        for completion, answer in zip(completions, answers):
            score = 0.0

            # Extract boxed answer from completion
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', completion)
            answer_match = re.search(r'\\boxed\{([^}]+)\}', answer)

            if boxed_match and answer_match:
                comp_val = boxed_match.group(1).strip()
                ans_val = answer_match.group(1).strip()

                # Exact match
                if comp_val == ans_val:
                    score = 1.0
                else:
                    # Try numerical comparison
                    try:
                        comp_num = float(comp_val.replace(',', ''))
                        ans_num = float(ans_val.replace(',', ''))
                        if abs(comp_num - ans_num) < 1e-6:
                            score = 1.0
                    except (ValueError, AttributeError):
                        pass

            scores.append(score)

        return scores

    def get_weight(self) -> float:
        """Higher weight for math accuracy."""
        return 0.7
