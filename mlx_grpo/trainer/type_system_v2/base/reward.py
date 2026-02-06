"""
Base Reward Class - SOLID Architecture
=======================================

Abstract base class for all reward implementations following SOLID principles.

Design Patterns:
- Template Method: compute() orchestrates the scoring workflow
- Observer Pattern: Hooks at each lifecycle stage
- Strategy Pattern: Subclasses implement type-specific scoring

Key Responsibilities (SRP):
- Compute reward scores in [0.0, 1.0]
- Validate inputs
- Provide hooks for extensibility
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["BaseReward", "RewardResult", "RewardHooks"]

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class RewardResult:
    """Structured reward computation result.

    Attributes:
        total_score: Final reward score [0.0, 1.0]
        component_scores: Dict of individual component scores
        metadata: Additional info (reasoning, penalties, etc.)
        valid: Whether the completion is valid for this type
    """
    total_score: float
    component_scores: dict[str, float]
    metadata: dict[str, Any]
    valid: bool = True

    def __post_init__(self):
        """Validate score bounds."""
        if not 0.0 <= self.total_score <= 1.0:
            logger.warning(
                f"Invalid total_score {self.total_score}, clipping to [0,1]"
            )
            self.total_score = max(0.0, min(1.0, self.total_score))


# =============================================================================
# OBSERVER HOOKS
# =============================================================================

class RewardHooks:
    """Observer hooks for reward computation lifecycle.

    Subclass and override methods to customize behavior without
    modifying the base reward class.

    Example:
        class LoggingHook(RewardHooks):
            def after_compute(self, result: RewardResult) -> RewardResult:
                logger.info(f"Reward: {result.total_score}")
                return result
    """

    def before_compute(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[Any]] = None,
    ) -> tuple[list[str], list[str], list[str], Optional[list[Any]]]:
        """Called before reward computation.

        Can modify inputs or raise exceptions to skip computation.
        """
        return prompts, completions, answers, types

    def after_compute(self, results: list[RewardResult]) -> list[RewardResult]:
        """Called after reward computation.

        Can modify results (e.g., apply penalties, bonuses).
        """
        return results

    def on_component_score(
        self,
        component_name: str,
        score: float,
        prompt: str,
        completion: str,
        answer: str,
    ) -> float:
        """Called for each component score.

        Can modify individual component scores.
        """
        return score

    def on_invalid_completion(
        self,
        prompt: str,
        completion: str,
        answer: str,
        reason: str,
    ) -> None:
        """Called when a completion is invalid for this type."""
        pass


# =============================================================================
# BASE REWARD CLASS
# =============================================================================

class BaseReward(ABC):
    """Abstract base class for type-specific reward functions.

    Subclasses must implement:
    - compute_single(): Score a single prompt-completion pair
    - validate_completion(): Check if completion is valid for this type

    Subclasses may override:
    - get_component_weights(): Customize component score weights
    - preprocess_completion(): Normalize completion before scoring
    - postprocess_score(): Adjust final score

    Usage:
        class ToolCallReward(BaseReward):
            def compute_single(self, prompt, completion, answer, type_info):
                # Type-specific scoring logic
                ...
    """

    def __init__(
        self,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize reward function.

        Args:
            hooks: Observer hooks for lifecycle events
            event_bus: Optional event bus for publishing lifecycle events
        """
        self.hooks = hooks or RewardHooks()
        self.event_bus = event_bus
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # =========================================================================
    # ABSTRACT METHODS (Must implement in subclass)
    # =========================================================================

    @abstractmethod
    def compute_single(
        self,
        prompt: str,
        completion: str,
        answer: str,
        type_info: Optional[Any] = None,
    ) -> RewardResult:
        """Compute reward for a single sample.

        This is the core scoring logic that subclasses must implement.

        Args:
            prompt: Input prompt
            completion: Model's completion
            answer: Ground truth answer
            type_info: Type metadata (can be dict, str, etc.)

        Returns:
            RewardResult with score, components, and metadata
        """
        raise NotImplementedError

    @abstractmethod
    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate if completion is appropriate for this type.

        Args:
            completion: Model's completion
            type_info: Type metadata

        Returns:
            (is_valid, reason_if_invalid)
        """
        raise NotImplementedError

    # =========================================================================
    # TEMPLATE METHOD (Override if needed)
    # =========================================================================

    def compute(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[Any]] = None,
    ) -> list[float]:
        """Compute rewards for a batch (template method).

        This orchestrates the reward computation workflow:
        1. Call before_compute hook
        2. Validate and score each sample
        3. Call after_compute hook
        4. Return scores

        Args:
            prompts: List of prompts
            completions: List of completions
            answers: List of ground truth answers
            types: Optional list of type metadata

        Returns:
            List of reward scores [0.0, 1.0]
        """
        # Hook: before compute
        prompts, completions, answers, types = self.hooks.before_compute(
            prompts, completions, answers, types
        )

        # Prepare types
        if types is None:
            types = [None] * len(prompts)

        # Validate batch sizes
        if not (len(prompts) == len(completions) == len(answers) == len(types)):
            raise ValueError(
                f"Batch size mismatch: prompts={len(prompts)}, "
                f"completions={len(completions)}, answers={len(answers)}, "
                f"types={len(types)}"
            )

        # Compute rewards
        results = []
        for prompt, completion, answer, type_info in zip(
            prompts, completions, answers, types
        ):
            try:
                # Preprocess
                completion = self.preprocess_completion(completion)

                # Validate
                is_valid, reason = self.validate_completion(completion, type_info)

                if not is_valid:
                    # Invalid completion - return zero score
                    result = RewardResult(
                        total_score=0.0,
                        component_scores={},
                        metadata={"invalid_reason": reason},
                        valid=False,
                    )
                    self.hooks.on_invalid_completion(
                        prompt, completion, answer, reason or "Unknown"
                    )
                    self._publish_event("reward.invalid", {
                        "reason": reason,
                        "type": self.__class__.__name__,
                    })
                else:
                    # Compute score
                    result = self.compute_single(
                        prompt, completion, answer, type_info
                    )

                # Postprocess
                result.total_score = self.postprocess_score(
                    result.total_score, result
                )

                results.append(result)

            except Exception as e:
                self._logger.error(
                    f"Error computing reward: {e}",
                    exc_info=True
                )
                # Fallback: zero score on error
                results.append(
                    RewardResult(
                        total_score=0.0,
                        component_scores={},
                        metadata={"error": str(e)},
                        valid=False,
                    )
                )

        # Hook: after compute
        results = self.hooks.after_compute(results)

        # Publish aggregate event
        scores = [r.total_score for r in results]
        self._publish_event("reward.computed", {
            "type": self.__class__.__name__,
            "count": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
        })

        return scores

    # =========================================================================
    # HOOK POINTS (Override to customize behavior)
    # =========================================================================

    def preprocess_completion(self, completion: str) -> str:
        """Normalize completion before scoring.

        Override to apply type-specific preprocessing (strip whitespace,
        normalize case, remove artifacts, etc.).
        """
        return completion.strip()

    def postprocess_score(
        self,
        score: float,
        result: RewardResult,
    ) -> float:
        """Adjust final score after computation.

        Override to apply penalties, bonuses, or score transformations.
        """
        return score

    def get_component_weights(self) -> dict[str, float]:
        """Get weights for combining component scores.

        Override to customize how component scores are weighted.

        Returns:
            Dict mapping component names to weights (should sum to 1.0)
        """
        return {}

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def combine_component_scores(
        self,
        component_scores: dict[str, float],
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        """Combine component scores using weighted average.

        Args:
            component_scores: Dict of component names to scores
            weights: Optional custom weights (uses get_component_weights if None)

        Returns:
            Weighted average score
        """
        if not component_scores:
            return 0.0

        if weights is None:
            weights = self.get_component_weights()

        if not weights:
            # Equal weights if none specified
            return sum(component_scores.values()) / len(component_scores)

        # Weighted average
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            score * weights.get(name, 0.0)
            for name, score in component_scores.items()
        )

        return weighted_sum / total_weight

    def _publish_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Publish event if event bus is configured."""
        if self.event_bus is not None:
            from ..events import Event
            self.event_bus.publish(Event(name=event_name, data=data))

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[Any]] = None,
    ) -> list[float]:
        """Make reward function callable.

        Allows: scores = reward(prompts, completions, answers)
        """
        return self.compute(prompts, completions, answers, types)
