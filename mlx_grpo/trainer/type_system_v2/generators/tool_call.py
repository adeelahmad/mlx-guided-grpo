"""
ToolCall Rollout Generator - Hermes format generation for Qwen-native tool calling
====================================================================================

Generator for tool/function calling using Qwen's native Hermes format:
    <tool_call>
    {"name": "function_name", "arguments": {"param1": "value1"}}
    </tool_call>

Key Features:
- Two-phase generation: if Phase 1 output missing <tool_call>, inject it for Phase 2
- Curriculum scaffolds partial Hermes JSON
- Stops at </tool_call> tag
- Contamination prevention (no thinking tags)
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.rollout_generator import (
    BaseRolloutGenerator,
    GenerationConfig,
    GeneratorHooks,
)

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["ToolCallRolloutGenerator", "ToolCallStrictHook"]

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL CALL ROLLOUT GENERATOR
# =============================================================================

class ToolCallRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator for Hermes-format tool calling.

    Generation Strategy:
    - max_length 384 tokens (Hermes JSON is more verbose than python-style)
    - Lower temperature (0.7) for focused outputs
    - Two-phase: if Phase 1 doesn't produce <tool_call>, inject it for Phase 2
    - Stops at </tool_call> tag

    Curriculum Strategy (Hermes format):
    - ratio=1.0: Full <tool_call>{"name":"f","arguments":{...}}</tool_call>
    - ratio=0.7: <tool_call>\\n{"name":"f","arguments":{partial...
    - ratio=0.5: <tool_call>\\n{"name":"f"
    - ratio=0.3: <tool_call>\\n{
    - ratio=0.0: No scaffolding
    """

    def __init__(
        self,
        max_length: int = 384,
        temperature: float = 0.7,
        continuation_tokens: int = 256,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)
        self.max_length = max_length
        self.temperature = temperature
        self.continuation_tokens = continuation_tokens

    def needs_phase_recovery(self) -> bool:
        """Tool calls use two-phase recovery.

        If Phase 1 output doesn't contain <tool_call>, we inject it
        as a prefix for Phase 2 to force the model into tool-call mode.
        """
        return True

    def get_generation_config(self) -> GenerationConfig:
        """Get generation config for Hermes tool calling."""
        stop_sequences = [
            "</tool_call>",  # Primary: end of tool call block
            "\\boxed{",      # Prevent math contamination
        ]

        return GenerationConfig(
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=0.95,
            stop_sequences=stop_sequences,
            two_phase=True,
            enforce_thinking=False,
            continuation_tokens=self.continuation_tokens,
            curriculum_ratio=0.0,
        )

    def check_incomplete(
        self,
        text: str,
        scaffold_ratio: float,
        target: str | None,
        type_info: Any | None,
        tokenizer: Any,
        **kwargs: Any,
    ) -> tuple[bool, str | None, int]:
        """Check if tool call completion is incomplete and build recovery prefix.

        Thinking-model aware: Qwen-Thinking naturally produces <think>...</think>
        before tool calls.  We strip completed thinking before checking the tool
        call portion and recover accordingly:

        Recovery strategy:
        - Completed thinking + complete tool_call  → done, no recovery
        - Completed thinking + incomplete tool_call → recover tool_call portion
        - Completed thinking + no tool_call at all  → inject <tool_call>\\n
        - Incomplete thinking (no </think>)         → inject </think>\\n<tool_call>\\n
        - No thinking + standard tool_call checks   → original logic
        - \\boxed{ contamination                     → abandon (not recoverable)

        Returns:
            (is_incomplete, fixed_prefix, injected_token_count)
        """
        # Math contamination = abandon (not recoverable for tool calls)
        if "\\boxed{" in text or r"\boxed{" in text:
            return False, None, 0

        # --- Handle thinking tags from thinking models ---
        has_think_open = "<think>" in text
        has_think_close = "</think>" in text

        if has_think_open and not has_think_close:
            # Incomplete thinking - close it and inject tool_call tag
            # Keep the model's thinking, just close + bridge to tool call
            injected = "\n</think>\n\n<tool_call>\n"
            fixed_prefix = text.rstrip() + injected
            count = len(tokenizer.encode(injected)) if tokenizer else 5
            logger.info(
                "Recovery: incomplete thinking → injecting </think> + <tool_call>"
            )
            return True, fixed_prefix, count

        # Strip completed thinking to inspect the tool-call portion
        tool_text = text
        thinking_prefix = ""
        if has_think_open and has_think_close:
            think_close_pos = text.rfind("</think>")
            thinking_prefix = text[: think_close_pos + len("</think>")]
            tool_text = text[think_close_pos + len("</think>"):].strip()

        has_tool_call_open = "<tool_call>" in tool_text
        has_tool_call_close = "</tool_call>" in tool_text

        # Complete: has both opening and closing tags
        if has_tool_call_close:
            return False, None, 0

        # Has opening tag - check JSON completeness
        if has_tool_call_open:
            inner = tool_text.split("<tool_call>", 1)[1].strip()
            try:
                json.loads(inner)
                # Valid JSON, just append closing tag
                fixed = text.rstrip() + "\n</tool_call>"
                injected = "\n</tool_call>"
                count = len(tokenizer.encode(injected)) if tokenizer else 2
                return True, fixed, count
            except (json.JSONDecodeError, ValueError):
                # Has <tool_call> but JSON is incomplete - continue generation
                # Use the full text (with thinking) as prefix so Phase 2
                # continues the JSON
                return True, text, 0

        # No <tool_call> tag at all - inject it for Phase 2
        injected = "<tool_call>\n"
        if thinking_prefix:
            # After completed thinking, inject tool_call
            fixed_prefix = thinking_prefix + "\n\n" + injected
        else:
            fixed_prefix = text.rstrip() + "\n" + injected
        count = len(tokenizer.encode(injected)) if tokenizer else 3
        return True, fixed_prefix, count

    def apply_curriculum(
        self,
        answer: str,
        ratio: float,
    ) -> str:
        """Apply curriculum scaffolding to Hermes tool call.

        Scaffolds partial <tool_call> JSON based on ratio.

        Examples (for answer="<tool_call>\\n{\\"name\\": \\"add\\", ...}\\n</tool_call>"):
        - ratio=1.0: full answer
        - ratio=0.7: '<tool_call>\\n{"name": "add", "arguments": {"a": 5, '
        - ratio=0.5: '<tool_call>\\n{"name": "add"'
        - ratio=0.3: '<tool_call>\\n{'
        - ratio=0.0: ''
        """
        if ratio <= 0:
            return ""

        if ratio >= 1.0:
            return answer

        # Extract inner JSON from <tool_call> blocks
        calls = self._extract_inner_json(answer)
        if not calls:
            cutoff = int(len(answer) * ratio)
            return answer[:cutoff]

        # Scaffold the first call
        first_call_json = calls[0]
        return self._scaffold_hermes_call(first_call_json, ratio)

    def _extract_inner_json(self, answer: str) -> list[str]:
        """Extract raw JSON strings from <tool_call> blocks."""
        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        return [m.strip() for m in pattern.findall(answer)]

    def _scaffold_hermes_call(self, json_str: str, ratio: float) -> str:
        """Scaffold a single Hermes tool call based on ratio."""
        try:
            parsed = json.loads(json_str)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
        except (json.JSONDecodeError, ValueError):
            cutoff = int(len(json_str) * ratio)
            return f"<tool_call>\n{json_str[:cutoff]}"

        if ratio >= 0.9:
            # Almost full - provide everything except closing
            args_str = json.dumps(arguments)
            full = f'{{"name": "{name}", "arguments": {args_str}}}'
            cutoff = int(len(full) * ratio)
            return f"<tool_call>\n{full[:cutoff]}"

        elif ratio >= 0.7:
            # Function name + partial arguments
            if isinstance(arguments, dict) and arguments:
                args_str = json.dumps(arguments)
                cutoff = max(1, int(len(args_str) * (ratio - 0.3)))
                return f'<tool_call>\n{{"name": "{name}", "arguments": {args_str[:cutoff]}'
            else:
                return f'<tool_call>\n{{"name": "{name}", "arguments": {{'

        elif ratio >= 0.5:
            return f'<tool_call>\n{{"name": "{name}"'

        elif ratio >= 0.3:
            return "<tool_call>\n{"

        else:
            return "<tool_call>"

    def is_generation_complete(
        self,
        text: str,
        phase: int,
    ) -> tuple[bool, Optional[str]]:
        """Check if generation is complete.

        For Hermes tool calls, complete when:
        1. </tool_call> tag is present
        2. OR math contamination detected
        3. OR valid JSON found (even without closing tag)

        Note: <think>/</ think> tags are NOT contamination - thinking models
        naturally produce them before tool calls.
        """
        if "\\boxed{" in text or r"\boxed{" in text:
            return True, "math_contamination"

        if "</tool_call>" in text:
            return True, "complete_tool_call"

        # Check for valid JSON (might be generated without closing tag)
        if "<tool_call>" in text:
            inner = text.split("<tool_call>", 1)[1].strip()
            try:
                json.loads(inner)
                return True, "json_complete_no_closing_tag"
            except (json.JSONDecodeError, ValueError):
                pass

        if len(text) >= self.max_length:
            return True, "max_length_reached"

        return False, None


# =============================================================================
# STRICT VALIDATION HOOK
# =============================================================================

class ToolCallStrictHook(GeneratorHooks):
    """Hook that aborts generation if math contamination is detected.

    Note: <think>/</ think> tags are allowed since thinking models naturally
    produce them before tool calls.
    """

    def __init__(self, abort_on_contamination: bool = True):
        self.abort_on_contamination = abort_on_contamination
        self.contamination_count = 0

    def on_phase_complete(
        self,
        phase_num: int,
        text: str,
        is_complete: bool,
    ) -> str:
        """Check for math contamination after each phase."""
        if "\\boxed{" in text or r"\boxed{" in text:
            self.contamination_count += 1
            logger.warning(
                f"Math contamination detected in phase {phase_num}: "
                f"{text[:100]}..."
            )
            if self.abort_on_contamination:
                return ""
        return text
