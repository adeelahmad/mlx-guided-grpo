"""
MCQ Reward - Multiple Choice and Exam Question Scoring
========================================================

Reward function for MCQ and exam-style tasks with ground truth answers.

Scoring Components:
1. Answer Match (50%): Exact letter match or boxed answer match
2. Format Quality (25%): Proper think/answer structure
3. Reasoning Quality (25%): Thinking phase evaluation

Special Handling:
- Gaming detection (hedging, multiple answers)
- Supports both letter answers (A-D) and boxed math answers
- Reuses proven logic from exam_reward.py and grpo_reward_functions.py
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.reward import BaseReward, RewardResult, RewardHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["MCQReward"]

logger = logging.getLogger(__name__)

# Pre-compiled patterns
RE_THINK_EXTRACT = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)
RE_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
RE_MCQ_OPTION = re.compile(
    r"(?:^|\s|'|\"|\()([A-D])(?:$|\s|\.|'|\"|\)|:)", re.IGNORECASE
)
RE_MCQ_ANSWER = re.compile(r"answer:\s*([A-D])", re.IGNORECASE)
RE_MCQ_REF = re.compile(r"(?:^|\s)([A-D])(?=$|\s|\.|:|\))", re.IGNORECASE)
RE_STRICT_FORMAT = re.compile(r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$")
RE_STRUCTURED_LIST = re.compile(r"(\n\s*[-*]|\n\s*\d+\.\s+)")

# Gaming detection
_HEDGING_PATTERNS = [
    re.compile(r"\b[A-D]\s+(or|and)\s+[A-D]\b", re.IGNORECASE),
    re.compile(r"\b[A-D]\s*/\s*[A-D]\b", re.IGNORECASE),
    re.compile(r"\b[A-D]\s*,\s*[A-D]\b", re.IGNORECASE),
    re.compile(r"\b(either|both)\b", re.IGNORECASE),
]

_BAD_PHRASES = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bi think\b", r"\bi believe\b", r"\bmaybe\b",
        r"\bi'm not sure\b", r"\bconfused\b", r"\bstuck\b",
    ]
]

_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<think><think>", "<|im_end|>"]


class MCQReward(BaseReward):
    """Reward function for MCQ / exam-style tasks.

    Scores based on:
    1. Answer correctness (exact match or boxed match)
    2. Format compliance (think/answer structure)
    3. Reasoning quality (thinking phase content)

    Penalizes gaming (hedging, multiple answers).
    """

    type_name = "mcq"

    def __init__(
        self,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)

    def get_component_weights(self) -> dict[str, float]:
        return {
            "answer_match": 0.50,
            "format_quality": 0.25,
            "reasoning_quality": 0.25,
        }

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """MCQ completions should have some answer content."""
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

        # Extract thinking and answer content
        thinking, answer_content = self._extract_components(completion)
        metadata["has_thinking"] = bool(thinking)

        # Ground truth from type_info or answer param
        ground_truth = answer
        possible_answers = None
        if isinstance(type_info, dict):
            ground_truth = type_info.get("ground_truth", answer)
            possible_answers = type_info.get("possible_boxed_answers")

        # Component 1: Answer Match
        match_score, match_type = self._score_answer_match(
            answer_content, ground_truth, possible_answers
        )
        component_scores["answer_match"] = match_score
        metadata["match_type"] = match_type

        # Gaming detection - penalize hedging
        gaming_penalty = self._detect_gaming(answer_content)
        if gaming_penalty > 0:
            component_scores["answer_match"] = max(
                0.0, component_scores["answer_match"] - gaming_penalty
            )
            metadata["gaming_penalty"] = gaming_penalty

        # Component 2: Format Quality
        component_scores["format_quality"] = self._score_format(completion)

        # Component 3: Reasoning Quality
        component_scores["reasoning_quality"] = self._score_reasoning(thinking)

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
            thinking = match.group(1).strip()
            answer = match.group(2).strip()
            return thinking, answer

        return None, text.strip()

    def _score_answer_match(
        self,
        completion_answer: str,
        ground_truth: str,
        possible_answers: list[str] | None = None,
    ) -> tuple[float, str]:
        """Score answer correctness. Returns (score, match_type)."""
        if not completion_answer or not ground_truth:
            return 0.0, "no_answer"

        # Try boxed answer extraction
        boxed_match = RE_BOXED.search(completion_answer)
        model_answer = boxed_match.group(1).strip() if boxed_match else None

        # Try MCQ letter extraction
        if model_answer is None:
            mcq_matches = RE_MCQ_OPTION.findall(completion_answer)
            mcq_matches.extend(RE_MCQ_ANSWER.findall(completion_answer))
            if mcq_matches:
                model_answer = mcq_matches[-1].upper()

        # Fallback: use full answer content
        if model_answer is None:
            model_answer = completion_answer.strip()

        # Extract ground truth letter if applicable
        gt_clean = ground_truth.strip()
        gt_match = RE_MCQ_REF.search(gt_clean)
        gt_letter = gt_match.group(1).upper() if gt_match else None

        # Exact match
        if model_answer.strip().lower() == gt_clean.lower():
            return 1.0, "exact"

        # Letter match
        if gt_letter and model_answer.upper() == gt_letter:
            return 1.0, "letter_match"

        # Possible answers match
        if possible_answers:
            for pa in possible_answers:
                if model_answer.strip().lower() == str(pa).strip().lower():
                    return 1.0, "possible_match"

        return 0.0, "no_match"

    def _detect_gaming(self, answer_content: str) -> float:
        """Detect gaming patterns. Returns penalty [0.0, 1.0]."""
        if not answer_content:
            return 0.0

        penalty = 0.0
        for pattern in _HEDGING_PATTERNS:
            if pattern.search(answer_content):
                penalty += 0.3

        return min(1.0, penalty)

    def _score_format(self, completion: str) -> float:
        """Score format quality."""
        if not completion:
            return 0.0

        score = 0.0

        # Has opening think tag
        if "<think>" in completion:
            score += 0.25

        # Has closing think tag
        if "</think>" in completion:
            score += 0.25

        # Proper order
        open_pos = completion.find("<think>")
        close_pos = completion.find("</think>")
        if open_pos >= 0 and close_pos > open_pos:
            score += 0.25

        # Has content after think tags
        if close_pos >= 0:
            after = completion[close_pos + len("</think>"):].strip()
            if len(after) >= 2:
                score += 0.25

        return score

    def _score_reasoning(self, thinking: str | None) -> float:
        """Score thinking quality."""
        if not thinking:
            return 0.0

        score = 1.0

        # Penalize bad phrases
        bad_count = sum(1 for p in _BAD_PHRASES if p.search(thinking))
        if bad_count > 0:
            score -= min(0.4, 0.1 * bad_count)

        # Penalize special tokens
        for token in _SPECIAL_TOKENS:
            if token in thinking:
                score -= 0.3

        # Reward structured reasoning
        if RE_STRUCTURED_LIST.search(thinking):
            score += 0.1

        # Length checks
        words = len(thinking.split())
        if words < 10:
            score *= max(0.3, words / 10)
        elif words > 150:
            excess = (words - 150) / 150
            score *= max(0.5, 1.0 - 0.2 * excess)

        return max(0.0, min(1.0, score))
