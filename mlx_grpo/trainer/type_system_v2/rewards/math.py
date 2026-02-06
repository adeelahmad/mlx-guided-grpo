"""
Math Reward - Scoring for mathematical reasoning tasks
========================================================

Scoring Components:
1. Correctness (50%): Numeric/symbolic comparison, LaTeX normalization, boxed extraction
2. Format (25%): Think/answer/boxed structure
3. Reasoning Quality (25%): Step-by-step evaluation, equation presence
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.reward import BaseReward, RewardResult, RewardHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["MathReward"]

logger = logging.getLogger(__name__)

RE_THINK_EXTRACT = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)
RE_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
RE_BOXED_NESTED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
RE_STRICT_FORMAT = re.compile(r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$")
RE_FRACTION = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
RE_SQRT = re.compile(r"\\sqrt\{([^}]*)\}")
RE_LATEX_CLEAN = re.compile(r"\\(?:text|mathrm|mathbf|mathit|left|right|displaystyle)\b\s*")
RE_STEP_MARKERS = re.compile(
    r"(?:step\s*\d|therefore|thus|hence|so\s+we|this\s+gives|which\s+means)",
    re.IGNORECASE,
)
RE_EQUATION = re.compile(r"[=<>≤≥]")


class MathReward(BaseReward):
    """Math Reward for math tasks."""

    type_name = "math"

    def __init__(
        self,
        numeric_tolerance: float = 1e-6,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)
        self.numeric_tolerance = numeric_tolerance

    def get_component_weights(self) -> dict[str, float]:
        return {
            "correctness": 0.50,
            "format": 0.25,
            "reasoning_quality": 0.25,
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

        # Component 2: Format
        component_scores["format"] = self._score_format(completion)

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
        """Score math correctness with LaTeX normalization and numeric comparison."""
        if not completion_answer or not ground_truth:
            return 0.0

        # Extract boxed answers if present
        comp_boxed = RE_BOXED_NESTED.search(completion_answer)
        gt_boxed = RE_BOXED_NESTED.search(ground_truth)
        comp_ans = comp_boxed.group(1).strip() if comp_boxed else completion_answer.strip()
        gt_ans = gt_boxed.group(1).strip() if gt_boxed else ground_truth.strip()

        # Normalize LaTeX for comparison
        comp_norm = self._normalize_latex(comp_ans)
        gt_norm = self._normalize_latex(gt_ans)

        if comp_norm == gt_norm:
            return 1.0

        # Numeric comparison with tolerance
        comp_num = self._try_parse_numeric(comp_norm)
        gt_num = self._try_parse_numeric(gt_norm)
        if comp_num is not None and gt_num is not None:
            if abs(comp_num - gt_num) <= self.numeric_tolerance:
                return 1.0
            if gt_num != 0 and abs(comp_num - gt_num) / abs(gt_num) < 0.01:
                return 0.8

        # Containment fallback
        if gt_norm in comp_norm or comp_norm in gt_norm:
            return 0.7

        return 0.0

    def _normalize_latex(self, text: str) -> str:
        """Normalize LaTeX for comparison (\\frac, \\sqrt, whitespace, etc.)."""
        if not text:
            return ""
        result = text.strip()
        result = RE_FRACTION.sub(lambda m: f"({m.group(1)})/({m.group(2)})", result)
        result = RE_SQRT.sub(r"sqrt(\1)", result)
        result = RE_LATEX_CLEAN.sub("", result)
        for old, new in [("\\,", ""), ("\\;", ""), ("\\cdot", "*"),
                         ("\\times", "*"), ("\\div", "/"), ("\\pi", "pi"),
                         ("$", ""), ("\\{", ""), ("\\}", "")]:
            result = result.replace(old, new)
        return " ".join(result.split()).strip().lower()

    def _try_parse_numeric(self, text: str) -> float | None:
        """Try to parse text as a number (int, float, fraction, percentage)."""
        if not text:
            return None
        cleaned = text.strip().replace(",", "").replace(" ", "")
        if "/" in cleaned and cleaned.count("/") == 1:
            parts = cleaned.split("/")
            try:
                return float(parts[0].strip("()")) / float(parts[1].strip("()"))
            except (ValueError, ZeroDivisionError):
                pass
        try:
            return float(cleaned)
        except ValueError:
            pass
        if cleaned.endswith("%"):
            try:
                return float(cleaned[:-1]) / 100
            except ValueError:
                pass
        return None

    def _score_format(self, completion: str) -> float:
        """Score format quality (think/answer/boxed structure)."""
        if not completion:
            return 0.0
        if RE_STRICT_FORMAT.search(completion.strip()):
            return 1.0
        score = 0.0
        if "<think>" in completion:
            score += 0.2
        if "</think>" in completion:
            score += 0.2
        open_pos = completion.find("<think>")
        close_pos = completion.find("</think>")
        if open_pos >= 0 and close_pos > open_pos:
            between = completion[open_pos + 7:close_pos].strip()
            if len(between) >= 10:
                score += 0.2
        if close_pos >= 0:
            after = completion[close_pos + 8:].strip()
            if len(after) >= 2:
                score += 0.2
        if "\\boxed{" in completion:
            score += 0.2
        return min(1.0, score)

    def _score_thinking(self, thinking: str) -> float:
        """Score mathematical reasoning quality (equations, steps, structure)."""
        if not thinking:
            return 0.0
        score = 0.8
        step_count = len(RE_STEP_MARKERS.findall(thinking))
        if step_count >= 2:
            score += 0.15
        elif step_count >= 1:
            score += 0.05
        eq_count = len(RE_EQUATION.findall(thinking))
        if eq_count >= 3:
            score += 0.1
        elif eq_count >= 1:
            score += 0.05
        words = len(thinking.split())
        if words < 15:
            score *= max(0.3, words / 15)
        elif words > 200:
            score *= max(0.5, 1.0 - 0.2 * (words - 200) / 200)
        return max(0.0, min(1.0, score))
