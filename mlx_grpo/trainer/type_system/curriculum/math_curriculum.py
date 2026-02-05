"""
Math Curriculum - Auto-discovered for type='math'
=================================================

Handles scaffolding for math problems.
Gradually reduces hints from full solution to no hints.
"""

from ..auto_discovery_extended import BaseCurriculum

__all__ = ["MathCurriculum"]


class MathCurriculum(BaseCurriculum):
    """Curriculum learning for math problems.

    Auto-discovered for type='math'.
    Just implement scaffolding logic!
    """

    def apply_scaffolding(
        self,
        prompt: str,
        answer: str,
        ratio: float = 0.5
    ) -> str:
        """Apply math-specific scaffolding.

        Provides partial thinking steps as hints.

        Args:
            prompt: Original problem
            answer: Ground truth (contains <think> tags)
            ratio: How much scaffolding (1.0 = full, 0.0 = none)

        Returns:
            Prompt with scaffolding
        """
        if ratio <= 0.0:
            return prompt

        # Extract thinking steps if present
        import re
        think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)

        if not think_match:
            # No thinking tags, just give partial answer
            return super().apply_scaffolding(prompt, answer, ratio)

        thinking = think_match.group(1).strip()

        if ratio >= 1.0:
            # Full scaffolding - provide complete thinking
            return f"{prompt}\n\nHere are the steps to solve this:\n<think>{thinking}</think>\n\nNow complete the answer:"

        # Partial scaffolding - provide portion of thinking steps
        lines = thinking.split('\n')
        num_lines = max(1, int(len(lines) * ratio))
        partial_thinking = '\n'.join(lines[:num_lines])

        return (
            f"{prompt}\n\n"
            f"Here's a start on the solution:\n"
            f"<think>\n{partial_thinking}\n...\n"
            f"</think>\n\n"
            f"Complete the reasoning and provide your answer:"
        )

    def get_start_ratio(self) -> float:
        """Start with 80% scaffolding for math."""
        return 0.8

    def get_end_ratio(self) -> float:
        """End with no scaffolding."""
        return 0.0

    def get_warmup_steps(self) -> int:
        """100 steps before curriculum starts."""
        return 100

    def get_strategy(self) -> str:
        """Use cosine decay for smooth reduction."""
        return "cosine"
