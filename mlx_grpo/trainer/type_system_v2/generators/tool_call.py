"""
ToolCall Rollout Generator - Clean function calling generation
===============================================================

Generator for tool/function calling that PREVENTS thinking contamination.

Key Features:
- Short, focused generation (no long reasoning)
- NO two-phase generation (no thinking phase)
- Curriculum provides partial function calls as scaffolding
- Strict validation to detect contamination early
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TYPE_CHECKING

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
    """Rollout generator for tool/function calling.

    Generation Strategy:
    - Short max_length (128-256 tokens)
    - Lower temperature (0.7) for more focused outputs
    - NO two_phase generation (no thinking phase)
    - Stop at newline (function calls are typically one line)

    Curriculum Strategy:
    - ratio=1.0: Provide full function call
    - ratio=0.7: Provide function name + opening paren
    - ratio=0.5: Provide function name only
    - ratio=0.0: No scaffolding

    This forces the model to:
    1. Start by just completing provided function calls
    2. Gradually learn to generate function names
    3. Finally generate everything from scratch
    """

    def __init__(
        self,
        max_length: int = 256,
        temperature: float = 0.7,
        allow_multi_line: bool = True,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize ToolCallRolloutGenerator.

        Args:
            max_length: Maximum generation length
            temperature: Sampling temperature
            allow_multi_line: Allow multiple function calls (multi-line)
            hooks: Observer hooks
            event_bus: Optional event bus
        """
        super().__init__(hooks=hooks, event_bus=event_bus)
        self.max_length = max_length
        self.temperature = temperature
        self.allow_multi_line = allow_multi_line

    def needs_phase_recovery(self) -> bool:
        """Tool calls never use two-phase recovery."""
        return False

    def get_generation_config(self) -> GenerationConfig:
        """Get generation config for tool calling.

        Short generations, no thinking phase.
        """
        stop_sequences = []

        # Stop at certain markers to prevent contamination
        if not self.allow_multi_line:
            stop_sequences.append("\n")  # Single function call

        # CRITICAL: Stop if model tries to start thinking
        stop_sequences.extend([
            "<think>",      # Prevent thinking contamination
            "\\boxed{",     # Prevent math contamination
            "```python",    # Prevent code block wrapping
            "```json",      # Prevent JSON code blocks
        ])

        return GenerationConfig(
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=0.95,
            stop_sequences=stop_sequences,
            two_phase=False,           # NO thinking phase
            enforce_thinking=False,    # NO thinking required
            continuation_tokens=0,     # No continuation
            curriculum_ratio=0.0,      # Set externally
        )

    def apply_curriculum(
        self,
        answer: str,
        ratio: float,
    ) -> str:
        """Apply curriculum scaffolding to function call.

        Strategy based on ratio:
        - 1.0: Full function call (model generates nothing)
        - 0.8: Function calls without last parameter
        - 0.6: Function name + opening paren
        - 0.4: Function name only
        - 0.0: No scaffolding (model generates everything)

        Examples (for answer="add(a=5, b=3)"):
        - ratio=1.0: "add(a=5, b=3)"  → model generates ""
        - ratio=0.8: "add(a=5, "      → model generates "b=3)"
        - ratio=0.6: "add("           → model generates "a=5, b=3)"
        - ratio=0.4: "add"            → model generates "(a=5, b=3)"
        - ratio=0.0: ""               → model generates "add(a=5, b=3)"
        """
        if ratio <= 0:
            return ""

        if ratio >= 1.0:
            return answer  # Full scaffolding

        # Handle multi-line answers (multiple function calls)
        lines = answer.split('\n')
        scaffolded_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            scaffolded = self._scaffold_single_call(line, ratio)
            scaffolded_lines.append(scaffolded)

        return '\n'.join(scaffolded_lines)

    def _scaffold_single_call(self, function_call: str, ratio: float) -> str:
        """Scaffold a single function call based on ratio."""
        # Parse function call: name(params)
        match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', function_call)

        if not match:
            # Can't parse, return proportional characters
            cutoff = int(len(function_call) * ratio)
            return function_call[:cutoff]

        func_name = match.group(1)
        params = match.group(2)

        if ratio >= 0.9:
            # Almost full - provide all but last few chars
            cutoff = int(len(function_call) * ratio)
            return function_call[:cutoff]

        elif ratio >= 0.7:
            # Provide function + most params
            if ',' in params:
                # Multi-param: provide all but last param
                param_list = params.split(',')
                partial_params = ','.join(param_list[:-1])
                if partial_params:
                    return f"{func_name}({partial_params}, "
                else:
                    return f"{func_name}("
            else:
                # Single param: provide function name + opening paren
                return f"{func_name}("

        elif ratio >= 0.5:
            # Provide function name + opening paren
            return f"{func_name}("

        elif ratio >= 0.3:
            # Provide function name only
            return func_name

        else:
            # Minimal scaffolding - just first few chars
            cutoff = int(len(func_name) * ratio * 3)  # Scale up for visibility
            return func_name[:cutoff]

    def is_generation_complete(
        self,
        text: str,
        phase: int,
    ) -> tuple[bool, Optional[str]]:
        """Check if generation is complete.

        For tool calls, generation is complete when:
        1. We have at least one closing paren
        2. Text doesn't end mid-parameter
        3. No thinking contamination detected
        """
        # Check for contamination - if present, mark as complete (invalid)
        if "<think>" in text or "</think>" in text:
            return True, "thinking_contamination"

        if "\\boxed{" in text or r"\boxed{" in text:
            return True, "math_contamination"

        # Check for function call completion
        # At minimum, must have opening and closing paren
        if '(' not in text or ')' not in text:
            return False, None  # Incomplete

        # Count parens
        open_count = text.count('(')
        close_count = text.count(')')

        if close_count < open_count:
            return False, None  # Still have unclosed parens

        # Check if ends with complete function call
        # Pattern: ends with ) or )\n or )\nfunc(...)
        if re.search(r'\)(\n|$)', text):
            return True, "complete_function_call"

        # If we hit max length, consider complete
        if len(text) >= self.max_length:
            return True, "max_length_reached"

        return False, None


# =============================================================================
# STRICT VALIDATION HOOK
# =============================================================================

class ToolCallStrictHook(GeneratorHooks):
    """Hook that aborts generation if contamination is detected.

    Use this to stop generation immediately when thinking tags appear.
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
        """Check for contamination after each phase."""
        if self._has_contamination(text):
            self.contamination_count += 1
            logger.warning(
                f"Contamination detected in phase {phase_num}: "
                f"{text[:100]}..."
            )

            if self.abort_on_contamination:
                # Return empty to signal invalid generation
                return ""

        return text

    def _has_contamination(self, text: str) -> bool:
        """Check if text has thinking contamination."""
        return (
            "<think>" in text or
            "</think>" in text or
            "\\boxed{" in text or
            r"\boxed{" in text
        )


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
# Basic usage
generator = ToolCallRolloutGenerator()

result = generator.generate(
    model=model,
    prompt_tokens=prompt_tokens,
    answer="add(a=5, b=3)",
    curriculum_ratio=0.7,  # Provide partial scaffolding
)

print(f"Generated: {result.completion}")
print(f"Phases: {result.num_phases}")

# With strict validation
strict_hook = ToolCallStrictHook(abort_on_contamination=True)
generator = ToolCallRolloutGenerator(hooks=strict_hook)

# This will abort if model tries to generate thinking tags
result = generator.generate(...)
"""
