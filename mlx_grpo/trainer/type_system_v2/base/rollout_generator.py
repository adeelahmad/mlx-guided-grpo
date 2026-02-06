"""
Base Rollout Generator - SOLID Architecture
============================================

Abstract base class for type-specific generation strategies.

Responsibilities (SRP):
- Generate model completions
- Apply curriculum scaffolding
- Handle two-phase generation
- Stop sequence detection
- Phase recovery logic

Design Patterns:
- Template Method: generate() orchestrates the workflow
- Strategy Pattern: Type-specific generation configs
- Observer Pattern: Hooks for custom generation
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    from ..events import EventBus

__all__ = ["BaseRolloutGenerator", "GenerationConfig", "GenerationResult", "GeneratorHooks"]

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for generation.

    Attributes:
        max_length: Maximum completion length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        stop_sequences: List of stop sequences
        two_phase: Enable two-phase generation
        enforce_thinking: Require thinking tags
        continuation_tokens: Tokens for phase 2
        curriculum_ratio: Scaffolding ratio [0, 1]
    """
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    stop_sequences: list[str] = None
    two_phase: bool = False
    enforce_thinking: bool = False
    continuation_tokens: int = 256
    curriculum_ratio: float = 0.0

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class GenerationResult:
    """Result of generation.

    Attributes:
        completion: Generated text
        tokens: Generated token IDs
        num_phases: Number of generation phases (1 or 2)
        phase_info: Info about each phase
        metadata: Additional info
    """
    completion: str
    tokens: list[int]
    num_phases: int
    phase_info: dict[int, dict[str, Any]]
    metadata: dict[str, Any]


# =============================================================================
# OBSERVER HOOKS
# =============================================================================

class GeneratorHooks:
    """Observer hooks for generation lifecycle."""

    def before_generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationConfig:
        """Called before generation starts.

        Can modify generation config.
        """
        return config

    def after_generate(self, result: GenerationResult) -> GenerationResult:
        """Called after generation completes."""
        return result

    def on_phase_complete(
        self,
        phase_num: int,
        text: str,
        is_complete: bool,
    ) -> str:
        """Called when a generation phase completes.

        Args:
            phase_num: Phase number (1 or 2)
            text: Generated text for this phase
            is_complete: Whether this phase ended naturally

        Returns:
            Potentially modified text
        """
        return text

    def on_stop_sequence(self, sequence: str, position: int) -> None:
        """Called when stop sequence is detected."""
        pass

    def on_curriculum_applied(
        self,
        scaffolding: str,
        ratio: float,
    ) -> str:
        """Called when curriculum scaffolding is applied.

        Args:
            scaffolding: The scaffolded text
            ratio: Curriculum ratio used

        Returns:
            Potentially modified scaffolding
        """
        return scaffolding


# =============================================================================
# BASE ROLLOUT GENERATOR
# =============================================================================

class BaseRolloutGenerator(ABC):
    """Abstract base class for type-specific rollout generators.

    Subclasses must implement:
    - get_generation_config(): Type-specific generation settings
    - apply_curriculum(): Type-specific scaffolding logic
    - is_generation_complete(): Check if generation should stop

    Usage:
        class ToolCallGenerator(BaseRolloutGenerator):
            def get_generation_config(self):
                return GenerationConfig(
                    max_length=256,
                    temperature=0.7,
                    two_phase=False,  # No thinking phase for tools
                )

            def apply_curriculum(self, answer, ratio):
                # Provide partial function calls as scaffolding
                ...
    """

    def __init__(
        self,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize generator.

        Args:
            hooks: Observer hooks for lifecycle events
            event_bus: Optional event bus for publishing lifecycle events
        """
        self.hooks = hooks or GeneratorHooks()
        self.event_bus = event_bus
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # =========================================================================
    # ABSTRACT METHODS (Must implement)
    # =========================================================================

    @abstractmethod
    def get_generation_config(self) -> GenerationConfig:
        """Get type-specific generation configuration.

        Returns:
            GenerationConfig with defaults for this type
        """
        raise NotImplementedError

    @abstractmethod
    def apply_curriculum(
        self,
        answer: str,
        ratio: float,
    ) -> str:
        """Apply curriculum scaffolding to answer.

        Args:
            answer: Ground truth answer
            ratio: Scaffolding ratio [0.0, 1.0]
                  0.0 = no scaffolding (model generates everything)
                  1.0 = full scaffolding (full answer provided)

        Returns:
            Scaffolded text to prepend to generation
        """
        raise NotImplementedError

    @abstractmethod
    def is_generation_complete(
        self,
        text: str,
        phase: int,
    ) -> tuple[bool, Optional[str]]:
        """Check if generation is complete for this phase.

        Args:
            text: Generated text so far
            phase: Current phase number (1 or 2)

        Returns:
            (is_complete, reason_if_complete)
        """
        raise NotImplementedError

    # =========================================================================
    # PHASE RECOVERY INTERFACE (Override in subclasses)
    # =========================================================================

    def needs_phase_recovery(self) -> bool:
        """Whether this type uses two-phase recovery for incomplete outputs.

        Override in subclasses. Default: True if two_phase is enabled in config.
        Tool call generators should return False.
        """
        return self.get_generation_config().two_phase

    def check_incomplete(
        self,
        text: str,
        scaffold_ratio: float,
        target: str | None,
        type_info: Any | None,
        tokenizer: Any,
        **kwargs: Any,
    ) -> tuple[bool, str | None, int]:
        """Check if a completion is incomplete and build recovery prefix.

        Called after Phase 1 generation to determine if Phase 2 is needed.

        Args:
            text: The full completion text (prefix + generation)
            scaffold_ratio: Curriculum scaffold ratio used
            target: Ground truth answer (for bridge/intuition)
            type_info: Type metadata for this sample
            tokenizer: Tokenizer for token counting
            **kwargs: Additional type-specific args

        Returns:
            Tuple of (is_incomplete, fixed_prefix, injected_token_count)
            - is_incomplete: Whether phase 2 continuation is needed
            - fixed_prefix: The prefix to use for phase 2 (includes injected tags)
            - injected_token_count: Tokens injected (not from model) to mask from loss
        """
        return False, None, 0

    # =========================================================================
    # TEMPLATE METHOD
    # =========================================================================

    def generate(
        self,
        model: Any,  # MLX model
        prompt_tokens: list[int],
        answer: str,
        curriculum_ratio: float = 0.0,
        config_override: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate completion (template method).

        Orchestrates the generation workflow:
        1. Get generation config
        2. Apply curriculum scaffolding
        3. Generate phase 1
        4. Check if phase 2 needed
        5. Generate phase 2 if needed
        6. Return result

        Args:
            model: MLX model
            prompt_tokens: Tokenized prompt
            answer: Ground truth answer (for curriculum)
            curriculum_ratio: Scaffolding ratio [0, 1]
            config_override: Override default config

        Returns:
            GenerationResult with completion and metadata
        """
        # Get config
        config = config_override or self.get_generation_config()

        self._publish_event("generation.started", {
            "type": self.get_type_name(),
            "max_length": config.max_length,
            "two_phase": config.two_phase,
        })

        # Hook: before generate
        config = self.hooks.before_generate(prompt_tokens, config)

        # Apply curriculum
        scaffolding = ""
        if curriculum_ratio > 0:
            scaffolding = self.apply_curriculum(answer, curriculum_ratio)
            scaffolding = self.hooks.on_curriculum_applied(
                scaffolding, curriculum_ratio
            )

        # Phase 1: Initial generation
        phase_info = {}

        phase1_text, phase1_tokens = self._generate_phase(
            model=model,
            prompt_tokens=prompt_tokens,
            prefix=scaffolding,
            max_length=config.max_length,
            temperature=config.temperature,
            top_p=config.top_p,
            stop_sequences=config.stop_sequences,
            phase=1,
        )

        # Check if complete
        is_complete, reason = self.is_generation_complete(phase1_text, phase=1)

        phase_info[1] = {
            "text": phase1_text,
            "tokens": len(phase1_tokens),
            "complete": is_complete,
            "reason": reason,
        }

        # Hook: phase 1 complete
        phase1_text = self.hooks.on_phase_complete(1, phase1_text, is_complete)

        # Phase 2: Continuation (if needed)
        num_phases = 1
        full_text = phase1_text
        all_tokens = phase1_tokens

        if config.two_phase and not is_complete:
            self._logger.debug("Starting phase 2 generation")

            # Continue from phase 1
            phase2_text, phase2_tokens = self._generate_phase(
                model=model,
                prompt_tokens=prompt_tokens + phase1_tokens,
                prefix="",
                max_length=config.continuation_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_sequences=config.stop_sequences,
                phase=2,
            )

            phase_info[2] = {
                "text": phase2_text,
                "tokens": len(phase2_tokens),
                "complete": True,
                "reason": "continuation",
            }

            # Hook: phase 2 complete
            phase2_text = self.hooks.on_phase_complete(2, phase2_text, True)

            full_text = phase1_text + phase2_text
            all_tokens = phase1_tokens + phase2_tokens
            num_phases = 2

        # Build result
        result = GenerationResult(
            completion=full_text,
            tokens=all_tokens,
            num_phases=num_phases,
            phase_info=phase_info,
            metadata={
                "curriculum_ratio": curriculum_ratio,
                "scaffolding": scaffolding,
                "config": config,
            },
        )

        # Hook: after generate
        result = self.hooks.after_generate(result)

        self._publish_event("generation.completed", {
            "type": self.get_type_name(),
            "num_phases": result.num_phases,
            "completion_length": len(result.completion),
        })

        return result

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _generate_phase(
        self,
        model: Any,
        prompt_tokens: list[int],
        prefix: str,
        max_length: int,
        temperature: float,
        top_p: float,
        stop_sequences: list[str],
        phase: int,
    ) -> tuple[str, list[int]]:
        """Generate a single phase.

        This is a placeholder - actual implementation depends on MLX API.
        Subclasses can override for custom generation logic.

        Args:
            model: MLX model
            prompt_tokens: Input tokens
            prefix: Text to prepend (curriculum scaffolding)
            max_length: Max generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: Stop sequences
            phase: Phase number

        Returns:
            (generated_text, generated_tokens)
        """
        # This is a placeholder - actual implementation would use MLX
        # For now, just return the prefix
        self._logger.warning(
            "_generate_phase is a placeholder - override in subclass or provide model API"
        )
        return prefix, []

    def get_type_name(self) -> str:
        """Get type name for this generator.

        Override to customize. Default: class name without 'Generator'.
        """
        class_name = self.__class__.__name__
        if class_name.endswith('Generator'):
            return class_name[:-9].lower()
        if class_name.endswith('RolloutGenerator'):
            return class_name[:-16].lower()
        return class_name.lower()

    def _publish_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Publish event if event bus is configured."""
        if self.event_bus is not None:
            from ..events import Event
            self.event_bus.publish(Event(name=event_name, data=data))

    def __call__(
        self,
        model: Any,
        prompt_tokens: list[int],
        answer: str,
        curriculum_ratio: float = 0.0,
        config_override: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Make generator callable."""
        return self.generate(
            model, prompt_tokens, answer, curriculum_ratio, config_override
        )
