"""
ToolCall Reward - Hermes format scoring for Qwen-native tool calling
=====================================================================

Reward function for tool/function calling using Qwen's native Hermes format:
    <tool_call>
    {"name": "function_name", "arguments": {"param1": "value1"}}
    </tool_call>

Scoring (strict):
- Exact tool name match required for any score
- Exact parameter match for full score
- Returns 0.0 for contamination (<think>, \\boxed{})
- Returns 0.0 if no <tool_call> block found
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.reward import BaseReward, RewardResult, RewardHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["ToolCallReward", "ToolCallRewardStrict"]

logger = logging.getLogger(__name__)

# Regex to extract <tool_call>...</tool_call> blocks
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# Regex to strip <think>...</think> blocks (thinking models produce these)
THINK_BLOCK_PATTERN = re.compile(
    r"<think>.*?</think>\s*", re.DOTALL
)
# Fallback: strip unclosed <think> block (truncated thinking)
THINK_UNCLOSED_PATTERN = re.compile(
    r"<think>.*$", re.DOTALL
)


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> content from completion.

    Thinking models (e.g. Qwen-Thinking) naturally produce thinking
    blocks before tool calls. These should be stripped before scoring
    the tool-call portion.

    Returns:
        Text with thinking blocks removed.
    """
    # First strip completed thinking blocks
    stripped = THINK_BLOCK_PATTERN.sub("", text)
    # Then strip any unclosed thinking block (truncated output)
    stripped = THINK_UNCLOSED_PATTERN.sub("", stripped)
    return stripped.strip()


# =============================================================================
# HELPER: Parse tool calls from Hermes format
# =============================================================================

def extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from Hermes-format text.

    Parses <tool_call>{"name": "...", "arguments": {...}}</tool_call> blocks.

    Returns:
        List of {"name": str, "arguments": dict} dicts.
    """
    matches = TOOL_CALL_PATTERN.findall(text)
    calls = []
    for m in matches:
        m = m.strip()
        try:
            parsed = json.loads(m)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            # Handle arguments as JSON string (double-encoded)
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    pass
            calls.append({"name": name, "arguments": arguments})
        except (json.JSONDecodeError, ValueError):
            continue
    return calls


# =============================================================================
# TOOL CALL REWARD
# =============================================================================

class ToolCallReward(BaseReward):
    """Reward function for Hermes-format tool calling.

    Scoring Strategy (strict, binary):
    - Tool name MUST match exactly -> 0.0 if wrong
    - If name matches: score based on exact argument accuracy
    - Exact args match: 1.0
    - Partial args match: proportional to matched params
    - No match at all: 0.0

    Component Weights:
        tool_name: 0.5  (must be exact for any score)
        arguments: 0.5  (exact param matching)
    """

    def __init__(
        self,
        strict: bool = True,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)
        self.strict = strict

    def get_component_weights(self) -> dict[str, float]:
        """Component score weights."""
        return {
            "tool_name": 0.5,
            "arguments": 0.5,
        }

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate completion for tool call scoring.

        Thinking models (Qwen-Thinking etc.) naturally produce
        <think>...</think> before tool calls - this is NOT contamination.
        We strip thinking content and validate the tool-call portion only.
        """
        # Strip thinking blocks - thinking models produce these naturally
        tool_portion = strip_thinking(completion)

        if "\\boxed{" in tool_portion or r"\boxed{" in tool_portion:
            return False, "Contains \\boxed{} (math contamination)"

        # Must contain at least one <tool_call> block in the non-thinking portion
        if "<tool_call>" not in tool_portion:
            return False, "No <tool_call> block found"

        return True, None

    def compute_single(
        self,
        prompt: str,
        completion: str,
        answer: str,
        type_info: Optional[Any] = None,
    ) -> RewardResult:
        """Compute reward for a single tool call sample.

        Strips thinking content before parsing tool calls, since thinking
        models naturally produce <think>...</think> before tool calls.

        Strict scoring:
        - Wrong tool name = 0.0 total
        - Right tool name + exact args = 1.0
        - Right tool name + partial args = proportional
        """
        component_scores = {}
        metadata = {}

        # Strip thinking content - only parse tool calls from tool portion
        tool_portion = strip_thinking(completion)

        # Parse tool calls
        completion_calls = extract_tool_calls(tool_portion)
        answer_calls = extract_tool_calls(answer)

        # Fallback: legacy python-style answer
        if not answer_calls:
            answer_calls = self._parse_legacy_answer(answer)

        metadata["completion_calls"] = len(completion_calls)
        metadata["expected_calls"] = len(answer_calls)
        metadata["completion_functions"] = [c["name"] for c in completion_calls]
        metadata["expected_functions"] = [c["name"] for c in answer_calls]

        if not answer_calls or not completion_calls:
            return RewardResult(
                total_score=0.0,
                component_scores={"tool_name": 0.0, "arguments": 0.0},
                metadata=metadata,
                valid=True,
            )

        # Score each expected call against completion calls
        name_score, args_score = self._score_call_list(
            completion_calls, answer_calls
        )

        component_scores["tool_name"] = name_score
        component_scores["arguments"] = args_score

        # If tool name is wrong, total score is 0
        if name_score == 0.0:
            return RewardResult(
                total_score=0.0,
                component_scores=component_scores,
                metadata=metadata,
                valid=True,
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

    def _score_call_list(
        self, comp_calls: list[dict], ans_calls: list[dict]
    ) -> tuple[float, float]:
        """Score completion calls against answer calls.

        Returns (name_score, args_score).
        """
        if len(comp_calls) != len(ans_calls):
            # Wrong number of calls - check if at least first call matches
            if not comp_calls:
                return 0.0, 0.0

        total_name = 0.0
        total_args = 0.0

        for ac in ans_calls:
            # Find matching completion call by name
            matched_cc = None
            for cc in comp_calls:
                if cc["name"] == ac["name"]:
                    matched_cc = cc
                    break

            if matched_cc is None:
                # Tool name not found
                total_name += 0.0
                total_args += 0.0
            else:
                total_name += 1.0
                total_args += self._score_exact_args(
                    matched_cc.get("arguments", {}),
                    ac.get("arguments", {}),
                )

        n = len(ans_calls)
        return total_name / n, total_args / n

    def _score_exact_args(self, comp_args: Any, ans_args: Any) -> float:
        """Score arguments with exact matching.

        Each parameter is scored independently:
        - Exact value match: 1.0
        - Key present but wrong value: 0.0
        - Key missing: 0.0
        """
        if not isinstance(ans_args, dict):
            return 1.0 if self._values_equal(comp_args, ans_args) else 0.0
        if not isinstance(comp_args, dict):
            return 0.0
        if not ans_args:
            return 1.0  # No params to match

        total = len(ans_args)
        matched = 0

        for key, expected in ans_args.items():
            if key in comp_args and self._values_equal(comp_args[key], expected):
                matched += 1

        return matched / total

    def _values_equal(self, a: Any, b: Any) -> bool:
        """Compare values with type coercion for numbers."""
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._values_equal(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(self._values_equal(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-6
        return str(a) == str(b)

    def _parse_legacy_answer(self, answer: str) -> list[dict]:
        """Parse legacy python-style answer as fallback."""
        calls = []
        for line in answer.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(\w+)\((.*)\)$", line, re.DOTALL)
            if match:
                calls.append({"name": match.group(1), "arguments": {}})
        return calls

    def preprocess_completion(self, completion: str) -> str:
        """Clean up completion before scoring."""
        return completion.strip().rstrip("\n")


# =============================================================================
# STRICT VARIANT
# =============================================================================

class ToolCallRewardStrict(ToolCallReward):
    """Ultra-strict tool call reward.

    ANY deviation from clean <tool_call> format returns 0.0.
    """

    def __init__(self, hooks: Optional[RewardHooks] = None):
        super().__init__(strict=True, hooks=hooks)

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """Extra strict validation (allows thinking before tool calls)."""
        is_valid, reason = super().validate_completion(completion, type_info)
        if not is_valid:
            return False, reason

        # Check tool portion only (thinking already stripped by parent)
        tool_portion = strip_thinking(completion).strip()
        remaining = TOOL_CALL_PATTERN.sub("", tool_portion).strip()
        if remaining:
            return False, f"Contains text outside <tool_call> blocks: {remaining[:50]}"

        return True, None


# =============================================================================
# HOOKS
# =============================================================================

class ToolCallLoggingHook(RewardHooks):
    """Hook that logs contamination events."""

    def on_invalid_completion(
        self,
        prompt: str,
        completion: str,
        answer: str,
        reason: str,
    ) -> None:
        logger.warning(
            f"Tool call contamination: {reason}\n"
            f"Completion preview: {completion[:200]}..."
        )
