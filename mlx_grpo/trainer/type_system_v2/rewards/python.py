"""
Python Reward - Scoring for code/programming tasks
=====================================================

Scoring Components:
1. Correctness (40%): Output match or code containment
2. Code Quality (30%): Syntax validity, code block presence
3. Thinking Quality (30%): Planning, approach explanation
"""

from __future__ import annotations

import ast
import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.reward import BaseReward, RewardResult, RewardHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["PythonReward"]

logger = logging.getLogger(__name__)

RE_THINK_EXTRACT = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)
RE_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
RE_INLINE_CODE = re.compile(r"`([^`]+)`")
RE_DEF_OR_CLASS = re.compile(r"(?:def |class )\w+")
RE_STRICT_FORMAT = re.compile(r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$")


class PythonReward(BaseReward):
    """Python Reward for python tasks."""

    type_name = "python"

    def __init__(
        self,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)

    def get_component_weights(self) -> dict[str, float]:
        return {
            "correctness": 0.40,
            "code_quality": 0.30,
            "thinking_quality": 0.30,
        }

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        if not completion or len(completion.strip()) < 2:
            return False, "Empty or too short completion"
        return True, None

    def compute_single(
        self,
        prompt: str,
        completion: str,
        answer: str,
        type_info: Optional[Any] = None,
    ) -> RewardResult:
        component_scores = {}
        metadata = {}

        thinking, answer_content = self._extract_components(completion)
        metadata["has_thinking"] = bool(thinking)

        # Component 1: Correctness
        component_scores["correctness"] = self._score_correctness(
            answer_content, answer
        )

        # Component 2: Code Quality
        component_scores["code_quality"] = self._score_code_quality(answer_content)

        # Component 3: Thinking Quality
        component_scores["thinking_quality"] = (
            self._score_thinking(thinking) if thinking else 0.0
        )

        total_score = self.combine_component_scores(component_scores)
        return RewardResult(
            total_score=total_score,
            component_scores=component_scores,
            metadata=metadata,
            valid=True,
        )

    # =========================================================================
    # SCORING HELPERS
    # =========================================================================

    def _extract_components(self, text: str) -> tuple[str | None, str]:
        """Extract thinking and answer content."""
        if not text:
            return None, ""
        match = RE_THINK_EXTRACT.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, text.strip()

    def _score_correctness(self, completion_answer: str, ground_truth: str) -> float:
        """Score code/output correctness - extracts code blocks for comparison."""
        if not completion_answer or not ground_truth:
            return 0.0

        # Extract code blocks from completion
        code_blocks = RE_CODE_BLOCK.findall(completion_answer)
        comp_code = "\n".join(code_blocks) if code_blocks else completion_answer

        # Exact match
        if comp_code.strip() == ground_truth.strip():
            return 1.0

        # Normalized match (whitespace-insensitive)
        comp_norm = " ".join(comp_code.split())
        gt_norm = " ".join(ground_truth.split())
        if comp_norm == gt_norm:
            return 1.0

        # Containment
        if gt_norm in comp_norm:
            return 0.8

        # Line-by-line overlap
        comp_lines = set(comp_code.strip().splitlines())
        gt_lines = set(ground_truth.strip().splitlines())
        if gt_lines:
            overlap = len(comp_lines & gt_lines) / len(gt_lines)
            if overlap >= 0.8:
                return 0.7
            if overlap >= 0.5:
                return 0.4

        return 0.0

    def _score_code_quality(self, answer_content: str) -> float:
        """Score code quality - syntax validity, structure, code block presence."""
        if not answer_content:
            return 0.0

        score = 0.0

        # Extract code blocks
        code_blocks = RE_CODE_BLOCK.findall(answer_content)
        has_code_block = len(code_blocks) > 0

        if has_code_block:
            score += 0.3
            code = "\n".join(code_blocks)
        else:
            code = answer_content

        # Check Python syntax validity
        try:
            ast.parse(code)
            score += 0.4  # Valid syntax
        except SyntaxError:
            # Try individual blocks
            for block in code_blocks:
                try:
                    ast.parse(block)
                    score += 0.2
                    break
                except SyntaxError:
                    pass

        # Has function/class definitions
        if RE_DEF_OR_CLASS.search(code):
            score += 0.2

        # Reasonable length
        lines = code.strip().splitlines()
        if 3 <= len(lines) <= 100:
            score += 0.1

        return min(1.0, score)

    def _score_thinking(self, thinking: str) -> float:
        """Score thinking quality for code tasks."""
        if not thinking:
            return 0.0
        score = 0.8
        words = len(thinking.split())
        # Reward approach/algorithm discussion
        approach_words = ["approach", "algorithm", "iterate", "loop", "recursive",
                          "edge case", "complexity", "return", "input", "output"]
        hits = sum(1 for w in approach_words if w in thinking.lower())
        if hits >= 3:
            score += 0.15
        elif hits >= 1:
            score += 0.05
        if words < 10:
            score *= max(0.3, words / 10)
        elif words > 200:
            score *= max(0.5, 1.0 - 0.2 * (words - 200) / 200)
        return max(0.0, min(1.0, score))
