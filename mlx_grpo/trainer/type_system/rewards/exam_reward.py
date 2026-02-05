"""
Exam Reward - Auto-discovered for type='exam'
==============================================

For exam-style problems (AIME, Math500, etc.)
Uses existing exam_* reward functions from grpo_reward_functions.py
"""

from typing import Optional
from ..auto_discovery import BaseReward

__all__ = ["ExamReward"]


class ExamReward(BaseReward):
    """Reward for exam-style problems.

    Auto-discovered for type='exam'.
    Integrates with existing exam reward functions.
    """

    def compute(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[str]] = None,
    ) -> list[float]:
        """
        Compute rewards for exam problems.

        Uses the existing exam_correctness reward from grpo_reward_functions.py
        """
        try:
            # Try to use existing exam_correctness reward
            from ...grpo_reward_functions import REWARD_REGISTRY

            if "exam_correctness" in REWARD_REGISTRY:
                exam_fn = REWARD_REGISTRY["exam_correctness"]
                return exam_fn(prompts, completions, answers, types)
        except Exception:
            pass

        # Fallback: simple boxed answer matching
        import re

        scores = []
        for completion, answer in zip(completions, answers):
            score = 0.0

            # Extract boxed answers
            comp_boxed = re.search(r'\\boxed\{([^}]+)\}', completion)
            ans_boxed = re.search(r'\\boxed\{([^}]+)\}', answer)

            if comp_boxed and ans_boxed:
                comp_val = comp_boxed.group(1).strip()
                ans_val = ans_boxed.group(1).strip()

                if comp_val == ans_val:
                    score = 1.0
                else:
                    # Try numerical comparison
                    try:
                        comp_num = float(comp_val.replace(',', ''))
                        ans_num = float(ans_val.replace(',', ''))
                        if abs(comp_num - ans_num) < 1e-6:
                            score = 1.0
                    except:
                        pass

            scores.append(score)

        return scores

    def get_weight(self) -> float:
        """High weight for exam correctness."""
        return 0.9
