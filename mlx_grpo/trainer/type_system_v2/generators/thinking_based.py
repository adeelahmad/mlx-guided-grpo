"""
ThinkingBasedGenerator - Shared base for thinking-structured generation
========================================================================

Intermediate base class for generators that use <think>...</think> structure.
Consolidates shared logic from MCQ and GeneralQNA generators.

Provides:
- Thinking-prefix curriculum scaffolding
- Completeness checking (</think> + answer)
- Two-phase recovery for incomplete thinking outputs
- Token masking for injected recovery tags

Subclasses only need to override:
- __init__() with type-specific defaults (max_length, temperature, etc.)
- check_incomplete() for type-specific recovery (e.g. exam-style)
"""

from __future__ import annotations

import random
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

__all__ = ["ThinkingBasedGenerator"]

logger = logging.getLogger(__name__)

# Constants for thinking structure
THINK_START = "<think>"
THINK_END = "</think>"
BOXED_OPEN = "\\boxed{"


class ThinkingBasedGenerator(BaseRolloutGenerator):
    """Intermediate base for thinking-based generation (MCQ, GeneralQNA).

    Handles all types that use <think>...</think> structure:
    - Curriculum: Provides partial thinking prefix as scaffolding
    - Completion check: Looks for </think> + answer content
    - Phase recovery: Injects closing tags for incomplete thinking
    - Token masking: Tracks injected tokens for loss masking

    Subclasses configure via constructor defaults:
        MCQ: max_length=1536, temperature=0.85, continuation_tokens=384
        GeneralQNA: max_length=1024, temperature=0.8, continuation_tokens=256
    """

    def __init__(
        self,
        max_length: int = 1024,
        temperature: float = 0.8,
        continuation_tokens: int = 256,
        think_start: str = THINK_START,
        think_end: str = THINK_END,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize ThinkingBasedGenerator.

        Args:
            max_length: Maximum generation length
            temperature: Sampling temperature
            continuation_tokens: Tokens for phase 2 continuation
            think_start: Opening thinking tag
            think_end: Closing thinking tag
            hooks: Observer hooks
            event_bus: Optional event bus
        """
        super().__init__(hooks=hooks, event_bus=event_bus)
        self.max_length = max_length
        self.temperature = temperature
        self.continuation_tokens = continuation_tokens
        self.think_start = think_start
        self.think_end = think_end

    # =========================================================================
    # GENERATION CONFIG
    # =========================================================================

    def get_generation_config(self) -> GenerationConfig:
        """Thinking-based generation with two-phase recovery."""
        return GenerationConfig(
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=0.95,
            stop_sequences=[],
            two_phase=True,
            enforce_thinking=True,
            continuation_tokens=self.continuation_tokens,
            curriculum_ratio=0.0,
        )

    # =========================================================================
    # PHASE RECOVERY
    # =========================================================================

    def needs_phase_recovery(self) -> bool:
        """Thinking-based types always use two-phase recovery."""
        return True

    def check_incomplete(
        self,
        text: str,
        scaffold_ratio: float,
        target: str | None,
        type_info: Any | None,
        tokenizer: Any,
        continuation_force_answer_ratio: float = 1.0,
        curriculum_preserve_intuition: bool = True,
        **kwargs: Any,
    ) -> tuple[bool, str | None, int]:
        """Check if thinking completion is incomplete and build recovery prefix.

        Handles the common case: <think> present but </think> missing.
        Injects closing tags and \\boxed{ opener for phase 2.

        Args:
            text: Full completion text
            scaffold_ratio: Curriculum scaffold ratio used
            target: Ground truth answer
            type_info: Type metadata for this sample
            tokenizer: Tokenizer for counting injected tokens
            continuation_force_answer_ratio: Probability of forcing answer
            curriculum_preserve_intuition: Preserve intuition bridges from target

        Returns:
            (is_incomplete, fixed_prefix, injected_token_count)
        """
        has_think_start = self.think_start in text
        has_think_end = self.think_end in text

        # Already complete - has thinking close tag and answer
        if has_think_end:
            # Check if answer is present after </think>
            after_think = text.split(self.think_end, 1)[-1].strip()
            if BOXED_OPEN in after_think or len(after_think) >= 10:
                return False, None, 0

            # Has </think> but no answer - inject boxed opener
            think_end_pos = text.find(self.think_end)
            injected_text = f"\n\n{BOXED_OPEN}"
            fixed_prefix = text[:think_end_pos + len(self.think_end)] + injected_text
            injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
            return True, fixed_prefix, injected_count

        # Has <think> but no </think> - needs recovery
        if has_think_start and scaffold_ratio < 1.0:
            force_answer = random.random() < continuation_force_answer_ratio

            if force_answer:
                return self._build_forced_answer_prefix(
                    text, target, tokenizer, curriculum_preserve_intuition
                )
            else:
                return self._build_natural_continuation_prefix(text, tokenizer)

        # No <think> at all - inject full structure
        if not has_think_start:
            injected_text = f"{self.think_start}\n{self.think_end}\n\n{BOXED_OPEN}"
            injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
            return True, injected_text, injected_count

        return False, None, 0

    def _build_forced_answer_prefix(
        self,
        text: str,
        target: str | None,
        tokenizer: Any,
        preserve_intuition: bool = True,
    ) -> tuple[bool, str, int]:
        """Build prefix that forces answer after truncated thinking.

        Truncates thinking to leave room for closing tags, then injects
        </think>\\n\\boxed{ as a bridge to phase 2 generation.
        """
        think_start_pos = text.find(self.think_start)
        model_thinking = text[think_start_pos + len(self.think_start):].rstrip()
        prefix_before = text[:think_start_pos] if think_start_pos > 0 else ""

        # Truncate thinking to leave room for closing tags
        max_thinking_chars = max(100, len(model_thinking) - 80)
        if len(model_thinking) > max_thinking_chars:
            model_thinking = model_thinking[:max_thinking_chars].rstrip() + "\n...[truncated]"

        # Extract intuition from target if available
        bridge_parts = []
        if target and preserve_intuition:
            target_think_start = target.find(self.think_start)
            target_think_end = target.find(self.think_end)
            if target_think_start != -1 and target_think_end != -1:
                target_thinking = target[
                    target_think_start + len(self.think_start):target_think_end
                ]
                intuition_match = re.search(
                    r"\[ANSWER INTUITION:[^\]]*\]", target_thinking
                )
                if intuition_match:
                    bridge_parts.append(intuition_match.group(0))

        # Build injected content
        if model_thinking and bridge_parts:
            bridge = "\n\n...\n\n" + "\n".join(bridge_parts)
            injected_text = f"{bridge}\n{self.think_end}\n{BOXED_OPEN}"
        elif model_thinking:
            injected_text = f"\n{self.think_end}\n{BOXED_OPEN}"
        else:
            injected_text = f"\n{self.think_end}\n{BOXED_OPEN}"

        fixed_prefix = (
            f"{prefix_before}{self.think_start}{model_thinking}{injected_text}"
        )
        injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
        return True, fixed_prefix, injected_count

    def _build_natural_continuation_prefix(
        self,
        text: str,
        tokenizer: Any,
    ) -> tuple[bool, str, int]:
        """Build prefix for natural continuation (close thinking, start answer)."""
        think_start_pos = text.find(self.think_start)
        model_thinking = text[think_start_pos + len(self.think_start):].rstrip()
        prefix_before = text[:think_start_pos] if think_start_pos > 0 else ""

        # Truncate if too long
        if len(model_thinking) > 500:
            truncated = model_thinking[:400] + "\n\n...[Truncated for brevity]...\n"
        else:
            truncated = model_thinking

        injected_text = f"\n{self.think_end}\n{BOXED_OPEN}"
        fixed_prefix = f"{prefix_before}{self.think_start}{truncated}{injected_text}"
        injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
        return True, fixed_prefix, injected_count

    # =========================================================================
    # CURRICULUM
    # =========================================================================

    def apply_curriculum(self, answer: str, ratio: float) -> str:
        """Apply thinking scaffolding.

        Provides a prefix of the thinking content based on ratio.
        Supports both <think>...</think> wrapped and plain text answers.

        Args:
            answer: Ground truth (may contain think tags)
            ratio: 0.0 = no scaffolding, 1.0 = full answer
        """
        if ratio <= 0:
            return ""

        if ratio >= 1.0:
            return answer

        # Try to extract thinking content
        think_match = re.search(
            r"<think>(.*?)</think>", answer, flags=re.DOTALL
        )

        if think_match:
            thinking_content = think_match.group(1)
            think_words = thinking_content.split()
            cutoff = max(1, int(len(think_words) * ratio))
            partial = " ".join(think_words[:cutoff])
            return f"{self.think_start}\n{partial}"

        # Plain text - provide proportional prefix as thinking
        words = answer.split()
        cutoff = max(1, int(len(words) * ratio))
        partial = " ".join(words[:cutoff])
        return f"{self.think_start}\n{partial}"

    # =========================================================================
    # COMPLETENESS CHECK
    # =========================================================================

    def is_generation_complete(
        self, text: str, phase: int
    ) -> tuple[bool, Optional[str]]:
        """Check if generation is complete.

        Phase 1: Need </think> and answer content
        Phase 2: Continuation always accepted
        """
        if phase == 1:
            has_close_think = self.think_end in text

            if has_close_think:
                after = text.split(self.think_end, 1)[-1].strip()
                if len(after) >= 2:
                    return True, "complete_with_answer"
                return False, None

            return False, None

        return True, "continuation_complete"
