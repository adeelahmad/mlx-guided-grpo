"""
ToolCall Reward - STRICT enforcement for clean tool calling
============================================================

Reward function for tool/function calling tasks that STRICTLY enforces:
- NO <think> tags allowed
- NO \\boxed{} answers allowed
- ONLY clean function calls

Design:
- Returns 0.0 for ANY completion with thinking contamination
- Scores based on exact match, function name, and parameters
- Provides detailed component scoring for debugging
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.reward import BaseReward, RewardResult, RewardHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["ToolCallReward", "ToolCallRewardStrict"]

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL CALL REWARD
# =============================================================================

class ToolCallReward(BaseReward):
    """Reward function for tool/function calling tasks.

    Scoring Components:
    1. Exact Match (40%): Completion exactly matches expected answer
    2. Function Name (30%): Correct function name(s) extracted
    3. Parameters (30%): Correct parameter values extracted

    Contamination Penalties:
    - <think> tags: 0.0 score
    - \\boxed{} answers: 0.0 score
    - Non-function text: Penalty applied

    Component Weights:
        exact_match: 0.4
        function_name: 0.3
        parameters: 0.3
    """

    def __init__(
        self,
        strict: bool = True,
        allow_extra_text: bool = False,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize ToolCallReward.

        Args:
            strict: If True, ANY contamination returns 0.0
            allow_extra_text: Allow text before/after function calls
            hooks: Observer hooks
            event_bus: Optional event bus
        """
        super().__init__(hooks=hooks, event_bus=event_bus)
        self.strict = strict
        self.allow_extra_text = allow_extra_text

    def get_component_weights(self) -> dict[str, float]:
        """Component score weights."""
        return {
            "exact_match": 0.4,
            "function_name": 0.3,
            "parameters": 0.3,
        }

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate completion has NO contamination.

        Returns:
            (is_valid, reason_if_invalid)
        """
        # Check for thinking tags
        if "<think>" in completion or "</think>" in completion:
            return False, "Contains <think> tags (thinking contamination)"

        # Check for boxed answers
        if "\\boxed{" in completion or r"\boxed{" in completion:
            return False, "Contains \\boxed{} (math contamination)"

        # Check for Markdown code blocks (sometimes models wrap in ```python)
        if completion.strip().startswith("```"):
            # Allow if it's JUST code block wrapping, no other text
            if not self.allow_extra_text:
                return False, "Contains code block markers"

        # Must contain at least one function call pattern
        if not self._has_function_call(completion):
            return False, "No function call pattern detected"

        return True, None

    def compute_single(
        self,
        prompt: str,
        completion: str,
        answer: str,
        type_info: Optional[Any] = None,
    ) -> RewardResult:
        """Compute reward for a single tool call sample.

        Args:
            prompt: The prompt (with function definitions)
            completion: Model's completion
            answer: Expected function call(s)
            type_info: Type metadata

        Returns:
            RewardResult with scores and metadata
        """
        component_scores = {}
        metadata = {}

        # Normalize
        completion_clean = self._normalize_function_call(completion)
        answer_clean = self._normalize_function_call(answer)

        # Component 1: Exact Match
        if completion_clean == answer_clean:
            component_scores["exact_match"] = 1.0
        else:
            # Partial match based on overlap
            completion_set = set(completion_clean.split())
            answer_set = set(answer_clean.split())
            if answer_set:
                overlap = len(completion_set & answer_set) / len(answer_set)
                component_scores["exact_match"] = overlap
            else:
                component_scores["exact_match"] = 0.0

        # Component 2: Function Name Match
        completion_funcs = self._extract_function_names(completion)
        answer_funcs = self._extract_function_names(answer)

        if answer_funcs:
            func_matches = sum(
                1 for f in completion_funcs if f in answer_funcs
            )
            component_scores["function_name"] = func_matches / len(answer_funcs)
        else:
            component_scores["function_name"] = 0.0

        metadata["completion_functions"] = completion_funcs
        metadata["expected_functions"] = answer_funcs

        # Component 3: Parameters Match
        param_score = self._score_parameters(completion, answer)
        component_scores["parameters"] = param_score

        # Combine scores
        total_score = self.combine_component_scores(component_scores)

        return RewardResult(
            total_score=total_score,
            component_scores=component_scores,
            metadata=metadata,
            valid=True,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _has_function_call(self, text: str) -> bool:
        """Check if text contains function call pattern."""
        # Pattern: function_name(...) or function_name(key=value, ...)
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)'
        return bool(re.search(pattern, text))

    def _extract_function_names(self, text: str) -> list[str]:
        """Extract function names from text.

        Examples:
            "add(a=1, b=2)" -> ["add"]
            "sort(arr)\ncalc(x=5)" -> ["sort", "calc"]
        """
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, text)
        return list(set(matches))  # Unique function names

    def _normalize_function_call(self, text: str) -> str:
        """Normalize function call for comparison.

        - Strip whitespace
        - Remove code block markers
        - Normalize spacing around punctuation
        """
        text = text.strip()

        # Remove code blocks
        text = re.sub(r'```\w*\n?', '', text)
        text = re.sub(r'```', '', text)

        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([(),=])\s*', r'\1', text)

        return text.strip()

    def _score_parameters(self, completion: str, answer: str) -> float:
        """Score parameter matching between completion and answer.

        Extracts parameter values and compares.
        """
        # Extract all parameter assignments (key=value)
        param_pattern = r'(\w+)\s*=\s*([^,)]+)'

        completion_params = set(re.findall(param_pattern, completion))
        answer_params = set(re.findall(param_pattern, answer))

        if not answer_params:
            return 1.0  # No parameters to match

        # Exact parameter matches
        exact_matches = len(completion_params & answer_params)
        total_params = len(answer_params)

        return exact_matches / total_params if total_params > 0 else 0.0

    def preprocess_completion(self, completion: str) -> str:
        """Clean up completion before scoring."""
        # Strip whitespace
        completion = completion.strip()

        # Remove trailing newlines
        completion = completion.rstrip('\n')

        return completion


# =============================================================================
# STRICT VARIANT - ZERO TOLERANCE
# =============================================================================

class ToolCallRewardStrict(ToolCallReward):
    """Ultra-strict tool call reward.

    ANY deviation from clean function calls returns 0.0:
    - Thinking tags: 0.0
    - Boxed answers: 0.0
    - Extra text: 0.0
    - Wrong function: 0.0

    Use this during training to FORCE the model to generate ONLY function calls.
    """

    def __init__(self, hooks: Optional[RewardHooks] = None):
        super().__init__(
            strict=True,
            allow_extra_text=False,
            hooks=hooks,
        )

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """Extra strict validation."""
        # Base validation
        is_valid, reason = super().validate_completion(completion, type_info)

        if not is_valid:
            return False, reason

        # Additional strict checks
        # Must be ONLY function calls, no extra text
        completion_clean = self._normalize_function_call(completion)

        # Check if entire completion is function calls
        # Split by newlines, each line should be a function call
        lines = [line.strip() for line in completion_clean.split('\n') if line.strip()]

        for line in lines:
            if not self._is_pure_function_call(line):
                return False, f"Line contains non-function text: {line[:50]}"

        return True, None

    def _is_pure_function_call(self, line: str) -> bool:
        """Check if line is ONLY a function call, no extra text."""
        # Pattern: function_name(...)
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)$'
        return bool(re.match(pattern, line.strip()))


# =============================================================================
# HOOKS EXAMPLE - Logging
# =============================================================================

class ToolCallLoggingHook(RewardHooks):
    """Example hook that logs contamination events."""

    def on_invalid_completion(
        self,
        prompt: str,
        completion: str,
        answer: str,
        reason: str,
    ) -> None:
        """Log when tool call is contaminated."""
        logger.warning(
            f"Tool call contamination detected: {reason}\n"
            f"Completion preview: {completion[:200]}..."
        )
