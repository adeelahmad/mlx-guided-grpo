"""
Python Rollout Generator - Generation for python tasks
============================================

Extends ThinkingBasedGenerator with python-specific settings.
Uses two-phase generation with thinking structure.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .thinking_based import ThinkingBasedGenerator
from ..base.rollout_generator import GeneratorHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["PythonRolloutGenerator"]

logger = logging.getLogger(__name__)


class PythonRolloutGenerator(ThinkingBasedGenerator):
    """Python Rollout Generator for python tasks.

    Uses thinking-based two-phase generation.
    Generation: max_length=1536, temperature=0.7, continuation_tokens=512
    Lower temperature for deterministic code; longer context for complete implementations.
    """

    type_name = "python"

    def __init__(
        self,
        max_length: int = 1536,
        temperature: float = 0.7,
        continuation_tokens: int = 512,
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
