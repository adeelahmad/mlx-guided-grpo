"""
MCQ Rollout Generator - Exam/MCQ Generation Strategy
======================================================

Generator for MCQ and exam tasks. Extends ThinkingBasedGenerator
with exam-specific phase recovery (boxed answer injection).

Differences from base ThinkingBasedGenerator:
- Longer max_length (1536 vs 1024)
- Higher temperature (0.85 vs 0.8)
- More continuation tokens (384 vs 256)
- Exam-specific check_incomplete: injects \\boxed{ after </think>
- Probabilistic recovery ratio for exam samples
"""

from __future__ import annotations

import random
import logging
from typing import Any, Optional, TYPE_CHECKING

from .thinking_based import ThinkingBasedGenerator, BOXED_OPEN
from ..base.rollout_generator import GeneratorHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["MCQRolloutGenerator"]

logger = logging.getLogger(__name__)


class MCQRolloutGenerator(ThinkingBasedGenerator):
    """Rollout generator for MCQ / exam tasks.

    Extends ThinkingBasedGenerator with exam-specific settings and
    recovery logic that injects \\boxed{ for answer extraction.

    Generation: max_length=1536, temperature=0.85, continuation_tokens=384
    Recovery: Exam-specific boxed answer injection
    """

    type_name = "mcq"

    def __init__(
        self,
        max_length: int = 1536,
        temperature: float = 0.85,
        continuation_tokens: int = 384,
        exam_phase_recovery_ratio: float = 0.5,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize MCQRolloutGenerator.

        Args:
            max_length: Maximum generation length (default: 1536)
            temperature: Sampling temperature (default: 0.85)
            continuation_tokens: Tokens for phase 2 (default: 384)
            exam_phase_recovery_ratio: Fraction of incomplete exam samples
                that get phase 2 recovery. Remaining are left as-is to provide
                learning signal for both truncated and recovered outputs.
            hooks: Observer hooks
            event_bus: Optional event bus
        """
        super().__init__(
            max_length=max_length,
            temperature=temperature,
            continuation_tokens=continuation_tokens,
            hooks=hooks,
            event_bus=event_bus,
        )
        self.exam_phase_recovery_ratio = exam_phase_recovery_ratio

    def check_incomplete(
        self,
        text: str,
        scaffold_ratio: float,
        target: str | None,
        type_info: Any | None,
        tokenizer: Any,
        **kwargs: Any,
    ) -> tuple[bool, str | None, int]:
        """Check if exam completion is incomplete with exam-specific recovery.

        Exam-specific recovery:
        1. Has </think> but no \\boxed{ → inject \\boxed{ opener
        2. Has <think> but no </think> → truncate + inject close tags
           (probabilistic: only exam_phase_recovery_ratio of samples)
        3. No <think> → inject full structure

        Args:
            text: Full completion text
            scaffold_ratio: Curriculum scaffold ratio
            target: Ground truth answer
            type_info: Type metadata
            tokenizer: Tokenizer for token counting
        """
        has_think_start = self.think_start in text
        has_think_end = self.think_end in text

        # Case 1: Has </think> but no boxed answer
        if has_think_end:
            think_end_pos = text.find(self.think_end)
            after_think = text[think_end_pos + len(self.think_end):]
            has_boxed = BOXED_OPEN in after_think

            if not has_boxed:
                injected_text = f"\n\n{BOXED_OPEN}"
                fixed_prefix = text[:think_end_pos + len(self.think_end)] + injected_text
                injected_count = (
                    len(tokenizer.encode(injected_text)) if tokenizer else 0
                )
                return True, fixed_prefix, injected_count

            # Has both </think> and boxed - complete
            return False, None, 0

        # Case 2: Has <think> but no </think>
        if has_think_start:
            # Probabilistic recovery: only recover this fraction
            if random.random() >= self.exam_phase_recovery_ratio:
                return False, None, 0

            from tqdm import tqdm

            think_start_pos = text.find(self.think_start)
            model_thinking = text[
                think_start_pos + len(self.think_start):
            ].rstrip()
            prefix_before = text[:think_start_pos] if think_start_pos > 0 else ""

            # Truncate thinking to leave room for closing tags
            max_thinking_chars = max(100, len(model_thinking) - 80)
            if len(model_thinking) > max_thinking_chars:
                model_thinking = model_thinking[:max_thinking_chars].rstrip()

            injected_text = f"\n... {self.think_end}\n{BOXED_OPEN}"
            fixed_prefix = (
                f"{prefix_before}{self.think_start}"
                f"{model_thinking}{injected_text}"
            )
            injected_count = (
                len(tokenizer.encode(injected_text)) if tokenizer else 0
            )
            tqdm.write(
                f"    [EXAM RECOVERY] Truncated thinking + injected "
                f"'... {self.think_end}\\n{BOXED_OPEN}' "
                f"({injected_count} masked tokens)"
            )
            return True, fixed_prefix, injected_count

        # Case 3: No <think> at all - inject full structure
        injected_text = f"{self.think_start}\n{self.think_end}\n\n{BOXED_OPEN}"
        injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
        return True, injected_text, injected_count
