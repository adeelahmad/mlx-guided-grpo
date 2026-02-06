"""
Math Rollout Generator - Generation for math tasks
==========================================

Extends ThinkingBasedGenerator with math-specific settings.
Uses two-phase generation with thinking structure.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .thinking_based import ThinkingBasedGenerator
from ..base.rollout_generator import GeneratorHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["MathRolloutGenerator"]

logger = logging.getLogger(__name__)


class MathRolloutGenerator(ThinkingBasedGenerator):
    """Math Rollout Generator for math tasks.

    Uses thinking-based two-phase generation.
    Generation: max_length=1536, temperature=0.7, continuation_tokens=384
    Lower temperature for precise math; longer context for complex proofs.
    """

    type_name = "math"

    def __init__(
        self,
        max_length: int = 1536,
        temperature: float = 0.7,
        continuation_tokens: int = 384,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(
            max_length=max_length,
            temperature=temperature,
            continuation_tokens=continuation_tokens,
            hooks=hooks,
            event_bus=event_bus,
        )
