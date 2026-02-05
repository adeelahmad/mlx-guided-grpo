"""
Math Phase Recovery - Auto-discovered for type='math'
=====================================================

Handles incomplete thinking for math problems.
Creates continuation prompts to complete reasoning.
"""

from ..auto_discovery_extended import BasePhaseRecovery

__all__ = ["MathPhaseRecovery"]


class MathPhaseRecovery(BasePhaseRecovery):
    """Phase recovery for math problems.

    Auto-discovered for type='math'.
    Handles incomplete <think> tags and continues reasoning.
    """

    def is_incomplete(self, output: str) -> bool:
        """Check if math thinking is incomplete.

        Math-specific: also checks if no answer was provided after thinking.
        """
        # Standard check: unclosed <think>
        if "<think>" in output and "</think>" not in output:
            return True

        # Math-specific: check if thinking closed but no boxed answer
        if "</think>" in output and r"\boxed{" not in output:
            return True

        return False

    def create_continuation_prompt(
        self,
        original_prompt: str,
        incomplete_output: str
    ) -> str:
        """Create continuation for incomplete math reasoning.

        Args:
            original_prompt: Original math problem
            incomplete_output: Incomplete generation

        Returns:
            Continuation prompt
        """
        # Case 1: Thinking tag not closed
        if "<think>" in incomplete_output and "</think>" not in incomplete_output:
            return (
                f"{incomplete_output}\n"
                "</think>\n\n"
                "Now provide your final answer in \\boxed{{}} format:"
            )

        # Case 2: Thinking closed but no boxed answer
        if "</think>" in incomplete_output and r"\boxed{" not in incomplete_output:
            return (
                f"{incomplete_output}\n\n"
                "Provide your final answer in \\boxed{{}} format:"
            )

        # Fallback
        return f"{incomplete_output}\n\nComplete your answer:"

    def get_continuation_tokens(self) -> int:
        """Math continuations can be longer (full answer)."""
        return 384

    def should_use_two_phase(self) -> bool:
        """Math benefits from two-phase generation."""
        return True
