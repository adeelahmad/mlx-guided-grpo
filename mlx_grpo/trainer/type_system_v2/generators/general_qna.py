"""
GeneralQNA Rollout Generator - Default Reasoning Generation Strategy
=====================================================================

Generator for general Q&A and math reasoning tasks.
Extends ThinkingBasedGenerator with default settings.

This is the simplest subclass - it inherits everything from
ThinkingBasedGenerator and only customizes constructor defaults.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .thinking_based import ThinkingBasedGenerator
from ..base.rollout_generator import GeneratorHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["GeneralQNARolloutGenerator"]

logger = logging.getLogger(__name__)


class GeneralQNARolloutGenerator(ThinkingBasedGenerator):
    """Rollout generator for general Q&A and reasoning tasks.

    Default/fallback generator. Inherits all behavior from
    ThinkingBasedGenerator with standard settings.

    Generation: max_length=1024, temperature=0.8, continuation_tokens=256
    Recovery: Standard thinking-based recovery (inherited)
    """

    type_name = "general_qna"

    def __init__(
        self,
        max_length: int = 1024,
        temperature: float = 0.8,
        continuation_tokens: int = 256,
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
