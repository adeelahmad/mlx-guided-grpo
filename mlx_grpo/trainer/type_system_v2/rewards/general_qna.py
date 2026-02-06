"""
GeneralQNA Reward - General Q&A and Math Reasoning Scoring
============================================================

Default reward function for thinking-based tasks (math, reasoning, general).

Scoring Components:
1. Correctness (40%): Answer match (exact, boxed, fuzzy)
2. Format (30%): Think/answer tag structure
3. Thinking Quality (30%): Reasoning evaluation

This is the default/fallback reward for samples that aren't tool_call or MCQ.
Reuses proven logic from grpo_reward_functions.py.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.reward import BaseReward, RewardResult, RewardHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["GeneralQNAReward"]

logger = logging.getLogger(__name__)

# Pre-compiled patterns
RE_THINK_EXTRACT = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)
RE_ANSWER_EXTRACT = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RE_ANSWER_TAGS = re.compile(r"</?answer>")
RE_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
RE_STRICT_FORMAT = re.compile(r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$")
RE_STRUCTURED_LIST = re.compile(r"(\n\s*[-*]|\n\s*\d+\.\s+)")
RE_WORD_TOKENS = re.compile(r"\w+")

_BAD_PHRASES = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bi think\b", r"\bi believe\b", r"\bmaybe\b",
        r"\bi'm not sure\b", r"\bconfused\b", r"\bstuck\b",
        r"\bi will now\b", r"\bi'll start by\b", r"\blet's see\b",
    ]
]

_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<think><think>", "<|im_end|>"]


class GeneralQNAReward(BaseReward):
    """Reward function for general Q&A, math, and reasoning tasks.

    Default/fallback type. Scores based on:
    1. Answer correctness (exact, boxed, or fuzzy match)
    2. Format compliance (think/answer structure)
    3. Thinking quality (reasoning evaluation)
    """

    type_name = "general_qna"

    def __init__(
        self,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)

    def get_component_weights(self) -> dict[str, float]:
        return {
            "correctness": 0.40,
            "format": 0.30,
            "thinking_quality": 0.30,
        }

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """General completions should have some content."""
        if not completion or len(completion.strip()) < 5:
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
        metadata["answer_length"] = len(answer_content)

        # Component 1: Correctness
        correctness, match_type = self._score_correctness(
            answer_content, answer
        )
        component_scores["correctness"] = correctness
        metadata["match_type"] = match_type

        # Component 2: Format
        component_scores["format"] = self._score_format(completion)

        # Component 3: Thinking Quality
        component_scores["thinking_quality"] = self._score_thinking(thinking)

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

        # Try <think>...</think> extraction
        match = RE_THINK_EXTRACT.search(text)
        if match:
            thinking = match.group(1).strip()
            answer = match.group(2).strip()
            # Clean answer tags if present
            answer = RE_ANSWER_TAGS.sub("", answer).strip()
            return thinking, answer

        # Try <answer>...</answer> extraction
        answer_match = RE_ANSWER_EXTRACT.search(text)
        if answer_match:
            return None, answer_match.group(1).strip()

        # Content after </think> if partial
        if "</think>" in text:
            after = text.split("</think>", 1)[-1]
            after = RE_ANSWER_TAGS.sub("", after).strip()
            return None, after

        return None, text.strip()

    def _score_correctness(
        self, completion_answer: str, ground_truth: str
    ) -> tuple[float, str]:
        """Score answer correctness. Returns (score, match_type)."""
        if not completion_answer or not ground_truth:
            return 0.0, "no_answer"

        # Normalize
        comp_clean = completion_answer.strip().lower()
        gt_clean = ground_truth.strip().lower()

        # Remove answer tags from ground truth too
        gt_clean = RE_ANSWER_TAGS.sub("", gt_clean).strip()

        # Exact match
        if comp_clean == gt_clean:
            return 1.0, "exact"

        # Boxed answer match
        comp_boxed = RE_BOXED.search(completion_answer)
        gt_boxed = RE_BOXED.search(ground_truth)

        if comp_boxed and gt_boxed:
            if comp_boxed.group(1).strip().lower() == gt_boxed.group(1).strip().lower():
                return 1.0, "boxed_match"

        if comp_boxed and comp_boxed.group(1).strip().lower() == gt_clean:
            return 1.0, "boxed_vs_text"

        # Substring containment
        if gt_clean in comp_clean:
            return 0.8, "contains"

        if comp_clean in gt_clean:
            ratio = len(comp_clean) / len(gt_clean) if gt_clean else 0
            return 0.6 * ratio, "partial"

        # Word overlap (Jaccard)
        comp_words = set(RE_WORD_TOKENS.findall(comp_clean))
        gt_words = set(RE_WORD_TOKENS.findall(gt_clean))
        if gt_words and comp_words:
            intersection = len(comp_words & gt_words)
            union = len(comp_words | gt_words)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > 0.5:
                return jaccard * 0.7, "fuzzy"

        return 0.0, "no_match"

    def _score_format(self, completion: str) -> float:
        """Score format quality (think/answer tag structure)."""
        if not completion:
            return 0.0

        # Strict format check
        if RE_STRICT_FORMAT.search(completion.strip()):
            return 1.0

        score = 0.0

        # Has opening think tag
        if "<think>" in completion:
            score += 0.25

        # Has closing think tag
        if "</think>" in completion:
            score += 0.25

        # Proper order and non-empty content
        open_pos = completion.find("<think>")
        close_pos = completion.find("</think>")
        if open_pos >= 0 and close_pos > open_pos:
            # Content between tags
            between = completion[open_pos + 7:close_pos].strip()
            if len(between) >= 5:
                score += 0.25

        # Content after closing tag
        if close_pos >= 0:
            after = completion[close_pos + 8:].strip()
            if len(after) >= 2:
                score += 0.25

        return score

    def _score_thinking(self, thinking: str | None) -> float:
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

        # Length heuristic
        words = len(thinking.split())
        if words < 20:
            score *= max(0.3, words / 20)
        elif words > 80:
            excess = (words - 80) / 80
            score *= max(0.5, 1.0 - 0.2 * excess)
        elif 30 <= words <= 60:
            score *= 1.1

        return max(0.0, min(1.0, score))
