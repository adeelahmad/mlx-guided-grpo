"""
Type Coordinator - Central orchestration for type system
=========================================================

Coordinates Reward, DatasetLoader, and RolloutGenerator for each type.

Follows SOLID principles:
- Single Responsibility: Coordinates components, doesn't implement logic
- Dependency Inversion: Depends on abstractions (base classes)
- Open/Closed: Extensible via registration, not modification

Usage:
    coordinator = TypeCoordinator()

    # Register components for a type
    coordinator.register_type(
        type_name="tool_call",
        reward=ToolCallReward(),
        loader=ToolCallDatasetLoader(tokenizer),
        generator=ToolCallRolloutGenerator(),
    )

    # Use components
    reward = coordinator.get_reward("tool_call")
    loader = coordinator.get_loader("tool_call")
    generator = coordinator.get_generator("tool_call")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from .base.reward import BaseReward
from .base.dataset_loader import BaseDatasetLoader
from .base.rollout_generator import BaseRolloutGenerator
from .events import EventBus, Event, TYPE_REGISTERED

if TYPE_CHECKING:
    from pathlib import Path
    from transformers import PreTrainedTokenizer

__all__ = ["TypeCoordinator", "TypeComponents", "auto_register_builtin_types"]

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE NORMALIZATION
# =============================================================================

_TYPE_ALIASES: dict[str, str] = {
    # Tool calling
    "tool": "tool_call",
    "function": "tool_call",
    "function_calling": "tool_call",
    "functioncalling": "tool_call",
    "func": "tool_call",
    "tool_use": "tool_call",
    "api_call": "tool_call",
    # MCQ / Exam
    "exam": "mcq",
    "aime": "mcq",
    "exam_math": "mcq",
    "exam_aime": "mcq",
    "exam_olympiad": "mcq",
    "olympiad": "mcq",
    "multiple_choice": "mcq",
    # Math
    "arithmetic": "math",
    "calculus": "math",
    "algebra": "math",
    "geometry": "math",
    "number_theory": "math",
    "combinatorics": "math",
    "probability": "math",
    "statistics": "math",
    "gsm8k": "math",
    "math500": "math",
    "competition_math": "math",
    # Python / Code
    "code": "python",
    "coding": "python",
    "programming": "python",
    "py": "python",
    # General (fallback)
    "thinking": "general_qna",
    "reasoning": "general_qna",
    "default": "general_qna",
}


def normalize_type(raw_type: str | dict | None) -> str:
    """Normalize a type identifier to a canonical type name.

    Args:
        raw_type: String, dict with 'type' key, or None

    Returns:
        Canonical type name (tool_call, mcq, general_qna)
    """
    if raw_type is None:
        return "general_qna"

    if isinstance(raw_type, dict):
        raw_type = raw_type.get("type", "general_qna")

    type_str = str(raw_type).strip().lower()
    return _TYPE_ALIASES.get(type_str, type_str)


# =============================================================================
# TYPE COMPONENTS
# =============================================================================

@dataclass
class TypeComponents:
    """Bundle of components for a type.

    Attributes:
        reward: Reward function
        loader: Dataset loader
        generator: Rollout generator
        metadata: Additional type info
    """
    reward: BaseReward
    loader: BaseDatasetLoader
    generator: BaseRolloutGenerator
    metadata: dict[str, Any]


# =============================================================================
# TYPE COORDINATOR
# =============================================================================

class TypeCoordinator:
    """Central coordinator for type system components.

    Manages registration and retrieval of type-specific:
    - Reward functions
    - Dataset loaders
    - Rollout generators

    Provides auto-detection of types from samples.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize coordinator.

        Args:
            event_bus: Optional shared event bus for all components
        """
        self._registry: dict[str, TypeComponents] = {}
        self._default_type: Optional[str] = None
        self.event_bus = event_bus or EventBus()

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register_type(
        self,
        type_name: str,
        reward: BaseReward,
        loader: BaseDatasetLoader,
        generator: BaseRolloutGenerator,
        metadata: Optional[dict[str, Any]] = None,
        set_as_default: bool = False,
    ) -> None:
        """Register components for a type.

        Args:
            type_name: Type identifier (e.g., "tool_call", "mcq")
            reward: Reward function instance
            loader: Dataset loader instance
            generator: Rollout generator instance
            metadata: Optional metadata about this type
            set_as_default: Set this as the default/fallback type
        """
        if type_name in self._registry:
            logger.warning("Overwriting existing type: %s", type_name)

        components = TypeComponents(
            reward=reward,
            loader=loader,
            generator=generator,
            metadata=metadata or {},
        )

        self._registry[type_name] = components
        logger.info("Registered type: %s", type_name)

        if set_as_default:
            self._default_type = type_name
            logger.info("Set default type: %s", type_name)

        self.event_bus.publish(Event(
            name=TYPE_REGISTERED,
            data={"type_name": type_name, "is_default": set_as_default},
        ))

    def unregister_type(self, type_name: str) -> None:
        """Remove a registered type."""
        if type_name in self._registry:
            del self._registry[type_name]
            logger.info("Unregistered type: %s", type_name)

            if self._default_type == type_name:
                self._default_type = None

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_reward(self, type_name: str) -> BaseReward:
        """Get reward function for type."""
        return self._get_component(type_name).reward

    def get_loader(self, type_name: str) -> BaseDatasetLoader:
        """Get dataset loader for type."""
        return self._get_component(type_name).loader

    def get_generator(self, type_name: str) -> BaseRolloutGenerator:
        """Get rollout generator for type."""
        return self._get_component(type_name).generator

    def get_components(self, type_name: str) -> TypeComponents:
        """Get all components for type."""
        return self._get_component(type_name)

    def get_or_default(self, type_name: str | dict | None) -> TypeComponents:
        """Get components with type normalization and fallback.

        Args:
            type_name: Raw type (string, dict, or None)

        Returns:
            TypeComponents for the resolved type
        """
        canonical = normalize_type(type_name)
        return self._get_component(canonical)

    def _get_component(self, type_name: str) -> TypeComponents:
        """Internal: get components with fallback."""
        if type_name in self._registry:
            return self._registry[type_name]

        # Try default
        if self._default_type and self._default_type in self._registry:
            logger.debug(
                "Type '%s' not found, using default '%s'",
                type_name, self._default_type,
            )
            return self._registry[self._default_type]

        available = list(self._registry.keys())
        raise KeyError(
            f"Type '{type_name}' not registered. Available: {available}"
        )

    # =========================================================================
    # TYPE DETECTION
    # =========================================================================

    def detect_type(self, sample: dict) -> Optional[str]:
        """Auto-detect type from sample.

        Checks each loader's validate_sample() method.

        Args:
            sample: Data sample dict

        Returns:
            Type name or None if no match
        """
        # Check explicit type field first
        if "type" in sample:
            explicit = normalize_type(sample["type"])
            if explicit in self._registry:
                return explicit

        # Try validation against each registered loader
        for type_name, components in self._registry.items():
            is_valid, _ = components.loader.validate_sample(sample)
            if is_valid:
                return type_name

        return None

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def load_dataset(
        self,
        path: str | Path,
        type_name: Optional[str] = None,
        auto_detect: bool = True,
        **kwargs,
    ) -> Any:
        """Load dataset using appropriate loader.

        Args:
            path: Path to dataset file
            type_name: Explicit type (if None, auto-detect from first sample)
            auto_detect: Enable auto-detection
            **kwargs: Passed to loader.load()

        Returns:
            LoadedDataset
        """
        from pathlib import Path
        import json

        path = Path(path)

        # Auto-detect type from first sample if needed
        if type_name is None and auto_detect:
            with open(path, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        first_sample = json.loads(first_line)
                        type_name = self.detect_type(first_sample)
                    except json.JSONDecodeError:
                        pass

        # Fallback to default
        if type_name is None:
            type_name = self._default_type

        if type_name is None:
            raise ValueError(
                "Could not determine dataset type. "
                "Specify type_name or set a default type."
            )

        loader = self.get_loader(type_name)
        return loader.load(path, **kwargs)

    def list_types(self) -> list[str]:
        """List all registered types."""
        return sorted(self._registry.keys())

    def get_type_info(self, type_name: str) -> dict[str, Any]:
        """Get info about a registered type."""
        components = self._get_component(type_name)
        return {
            "type_name": type_name,
            "reward": components.reward.__class__.__name__,
            "loader": components.loader.__class__.__name__,
            "generator": components.generator.__class__.__name__,
            "metadata": components.metadata,
        }


# =============================================================================
# AUTO-REGISTRATION HELPER
# =============================================================================

def auto_register_builtin_types(
    coordinator: TypeCoordinator,
    tokenizer: PreTrainedTokenizer,
) -> None:
    """Auto-register all built-in types.

    Registers: tool_call, mcq, general_qna (default).

    Args:
        coordinator: TypeCoordinator instance
        tokenizer: Tokenizer to pass to loaders
    """
    event_bus = coordinator.event_bus

    # Tool Call
    try:
        from .rewards.tool_call import ToolCallReward
        from .loaders.tool_call import ToolCallDatasetLoader
        from .generators.tool_call import ToolCallRolloutGenerator

        coordinator.register_type(
            type_name="tool_call",
            reward=ToolCallReward(strict=True, event_bus=event_bus),
            loader=ToolCallDatasetLoader(
                tokenizer, strict=True, event_bus=event_bus
            ),
            generator=ToolCallRolloutGenerator(event_bus=event_bus),
            metadata={"description": "Tool/function calling tasks"},
        )
    except ImportError as e:
        logger.warning("Could not register tool_call: %s", e)

    # MCQ / Exam
    try:
        from .rewards.mcq import MCQReward
        from .loaders.mcq import MCQDatasetLoader
        from .generators.mcq import MCQRolloutGenerator

        coordinator.register_type(
            type_name="mcq",
            reward=MCQReward(event_bus=event_bus),
            loader=MCQDatasetLoader(tokenizer, event_bus=event_bus),
            generator=MCQRolloutGenerator(event_bus=event_bus),
            metadata={"description": "Multiple choice / exam tasks"},
        )
    except ImportError as e:
        logger.warning("Could not register mcq: %s", e)

    # Math
    try:
        from .rewards.math import MathReward
        from .loaders.math import MathDatasetLoader
        from .generators.math import MathRolloutGenerator

        coordinator.register_type(
            type_name="math",
            reward=MathReward(event_bus=event_bus),
            loader=MathDatasetLoader(tokenizer, event_bus=event_bus),
            generator=MathRolloutGenerator(event_bus=event_bus),
            metadata={"description": "Mathematical reasoning tasks"},
        )
    except ImportError as e:
        logger.warning("Could not register math: %s", e)

    # Python / Code
    try:
        from .rewards.python import PythonReward
        from .loaders.python import PythonDatasetLoader
        from .generators.python import PythonRolloutGenerator

        coordinator.register_type(
            type_name="python",
            reward=PythonReward(event_bus=event_bus),
            loader=PythonDatasetLoader(tokenizer, event_bus=event_bus),
            generator=PythonRolloutGenerator(event_bus=event_bus),
            metadata={"description": "Python/code generation tasks"},
        )
    except ImportError as e:
        logger.warning("Could not register python: %s", e)

    # General QNA (DEFAULT)
    try:
        from .rewards.general_qna import GeneralQNAReward
        from .loaders.general_qna import GeneralQNADatasetLoader
        from .generators.general_qna import GeneralQNARolloutGenerator

        coordinator.register_type(
            type_name="general_qna",
            reward=GeneralQNAReward(event_bus=event_bus),
            loader=GeneralQNADatasetLoader(tokenizer, event_bus=event_bus),
            generator=GeneralQNARolloutGenerator(event_bus=event_bus),
            metadata={"description": "General Q&A and reasoning tasks"},
            set_as_default=True,
        )
    except ImportError as e:
        logger.warning("Could not register general_qna: %s", e)

    logger.info(
        "Registered %d built-in types: %s",
        len(coordinator.list_types()),
        coordinator.list_types(),
    )
