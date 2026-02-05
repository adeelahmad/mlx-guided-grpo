"""
Type System Registry - Metaclass-based Plugin Architecture
===========================================================

An elegant, extensible type system using advanced Python metaprogramming.

Design Patterns:
- Registry Pattern: Central registration of type handlers
- Strategy Pattern: Interchangeable generation strategies
- Factory Pattern: Dynamic handler creation
- Protocol Pattern: Structural subtyping
- Decorator Pattern: Non-invasive registration

Key Features:
- Automatic validation via metaclasses
- Decorator-based registration
- Type-safe protocols
- Lazy loading support
- Inheritance-aware lookups
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeAlias, runtime_checkable
from functools import wraps
from collections.abc import Sequence

__all__ = [
    "DataTypeHandler",
    "GenerationStrategy",
    "RewardConfig",
    "CurriculumConfig",
    "register_type",
    "get_type_handler",
    "list_types",
    "detect_dataset_type",
    "TypeRegistry",
]

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class GenerationStrategy:
    """Immutable generation strategy configuration.

    Attributes:
        max_length: Maximum completion length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        two_phase: Enable two-phase thinking generation
        enforce_thinking: Require <think> tags
        continuation_tokens: Tokens for phase 2 continuation
        stop_sequences: Custom stop sequences
    """
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    two_phase: bool = False
    enforce_thinking: bool = False
    continuation_tokens: int = 256
    stop_sequences: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")

    def with_overrides(self, **kwargs) -> GenerationStrategy:
        """Create new strategy with overrides."""
        from dataclasses import replace
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RewardConfig:
    """Reward function configuration.

    Attributes:
        functions: List of reward function names
        weights: Corresponding weights (must sum to 1.0)
        require_all: Fail if any reward function is missing
    """
    functions: tuple[str, ...] = field(default_factory=tuple)
    weights: tuple[float, ...] = field(default_factory=tuple)
    require_all: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not self.functions:
            raise ValueError("Must specify at least one reward function")

        if self.weights:
            if len(self.functions) != len(self.weights):
                raise ValueError(
                    f"Mismatch: {len(self.functions)} functions, "
                    f"{len(self.weights)} weights"
                )

            total = sum(self.weights)
            if not abs(total - 1.0) < 1e-6:
                logger.warning(
                    f"Reward weights sum to {total:.3f}, not 1.0. "
                    "Will be normalized."
                )

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        """Get weights normalized to sum to 1.0."""
        if not self.weights:
            # Equal weights
            n = len(self.functions)
            return tuple(1.0 / n for _ in range(n))

        total = sum(self.weights)
        return tuple(w / total for w in self.weights)

    def with_overrides(self, **kwargs) -> RewardConfig:
        """Create new config with overrides."""
        from dataclasses import replace
        return replace(self, **kwargs)


@dataclass(frozen=True)
class CurriculumConfig:
    """Curriculum learning configuration.

    Attributes:
        enabled: Whether to use curriculum
        start_ratio: Starting scaffolding ratio (1.0 = full scaffolding)
        end_ratio: Ending scaffolding ratio (0.0 = no scaffolding)
        warmup_steps: Steps before curriculum starts
        total_steps: Total steps for curriculum (0 = use training iters)
        strategy: Curriculum strategy ('linear', 'exponential', 'cosine')
    """
    enabled: bool = False
    start_ratio: float = 1.0
    end_ratio: float = 0.0
    warmup_steps: int = 0
    total_steps: int = 0
    strategy: str = "linear"

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.start_ratio <= 1.0:
            raise ValueError(f"start_ratio must be in [0, 1], got {self.start_ratio}")
        if not 0.0 <= self.end_ratio <= 1.0:
            raise ValueError(f"end_ratio must be in [0, 1], got {self.end_ratio}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.strategy not in ("linear", "exponential", "cosine"):
            raise ValueError(f"Unknown strategy: {self.strategy}")


# =============================================================================
# PROTOCOL DEFINITIONS (Structural Subtyping)
# =============================================================================

@runtime_checkable
class DataTypeHandler(Protocol):
    """Protocol for data type handlers.

    Any class implementing these methods can be registered as a type handler,
    without needing to inherit from a base class (structural subtyping).
    """

    @property
    def type_name(self) -> str:
        """Unique identifier for this type."""
        ...

    def get_reward_config(self) -> RewardConfig:
        """Get reward configuration for this type."""
        ...

    def get_generation_strategy(self) -> GenerationStrategy:
        """Get generation strategy for this type."""
        ...

    def get_curriculum_config(self) -> CurriculumConfig:
        """Get curriculum configuration for this type."""
        ...

    def preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a data sample (optional hook)."""
        ...

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Validate if sample matches this type (optional)."""
        ...


# =============================================================================
# METACLASS FOR AUTO-VALIDATION
# =============================================================================

from abc import ABCMeta

class TypeHandlerMeta(ABCMeta):
    """Metaclass that validates handler implementations at class creation time.

    Ensures:
    - All required methods are implemented
    - type_name is a class attribute or property
    - Configurations are valid
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip validation for base classes
        if name in ('BaseDataTypeHandler', 'AbstractDataTypeHandler'):
            return cls

        # Check if type_name is defined
        if not hasattr(cls, 'type_name'):
            raise TypeError(
                f"{name} must define 'type_name' class attribute or property"
            )

        # Validate it implements the protocol
        if not isinstance(cls, type):
            return cls

        # Check required methods
        required_methods = [
            'get_reward_config',
            'get_generation_strategy',
            'get_curriculum_config',
        ]

        for method in required_methods:
            if not callable(getattr(cls, method, None)):
                logger.warning(
                    f"{name}.{method} is not callable. "
                    f"Handler may not work correctly."
                )

        logger.debug(f"Validated type handler: {name}")
        return cls


# =============================================================================
# BASE HANDLER CLASS
# =============================================================================

class BaseDataTypeHandler(ABC, metaclass=TypeHandlerMeta):
    """Abstract base class for type handlers with sensible defaults.

    Subclasses should override:
    - type_name (required)
    - get_reward_config() (optional, defaults to basic rewards)
    - get_generation_strategy() (optional, defaults to standard generation)
    - get_curriculum_config() (optional, defaults to disabled)
    """

    type_name: str = "base"

    def get_reward_config(self) -> RewardConfig:
        """Default: use basic accuracy rewards."""
        return RewardConfig(
            functions=("r1_correctness",),
            weights=(1.0,),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Default: standard generation."""
        return GenerationStrategy(
            max_length=512,
            temperature=0.8,
            two_phase=False,
            enforce_thinking=False,
        )

    def get_curriculum_config(self) -> CurriculumConfig:
        """Default: no curriculum."""
        return CurriculumConfig(enabled=False)

    def preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Default: no preprocessing."""
        return sample

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Default: accept all samples."""
        return True


# =============================================================================
# REGISTRY WITH DECORATOR
# =============================================================================

class TypeRegistry:
    """Central registry for type handlers using decorator pattern.

    Example:
        @TypeRegistry.register("my_type")
        class MyHandler(BaseDataTypeHandler):
            type_name = "my_type"

            def get_reward_config(self) -> RewardConfig:
                return RewardConfig(
                    functions=("custom_reward",),
                    weights=(1.0,)
                )
    """

    _handlers: dict[str, type[DataTypeHandler]] = {}
    _instances: dict[str, DataTypeHandler] = {}  # Singleton instances

    @classmethod
    def register(
        cls,
        type_name: str | None = None,
        *,
        override: bool = False,
    ) -> Callable[[type[DataTypeHandler]], type[DataTypeHandler]]:
        """Decorator to register a type handler.

        Args:
            type_name: Type identifier (uses handler.type_name if None)
            override: Allow overriding existing handlers

        Returns:
            Decorator function
        """
        def decorator(handler_cls: type[DataTypeHandler]) -> type[DataTypeHandler]:
            # Determine type name
            name = type_name or getattr(handler_cls, 'type_name', None)

            if name is None:
                raise ValueError(
                    f"Cannot determine type_name for {handler_cls.__name__}. "
                    "Specify in decorator or as class attribute."
                )

            # Check for conflicts
            if name in cls._handlers and not override:
                raise ValueError(
                    f"Type '{name}' already registered by {cls._handlers[name].__name__}. "
                    f"Use override=True to replace."
                )

            # Register
            cls._handlers[name] = handler_cls
            logger.info(f"Registered type handler: {name} -> {handler_cls.__name__}")

            return handler_cls

        return decorator

    @classmethod
    def get(cls, type_name: str, *, singleton: bool = True) -> DataTypeHandler:
        """Get handler instance for type.

        Args:
            type_name: Type identifier
            singleton: Return cached instance (True) or create new (False)

        Returns:
            Handler instance

        Raises:
            KeyError: If type not registered
        """
        if type_name not in cls._handlers:
            available = ", ".join(sorted(cls._handlers.keys()))
            raise KeyError(
                f"Unknown type: '{type_name}'. Available: [{available}]"
            )

        if singleton:
            if type_name not in cls._instances:
                cls._instances[type_name] = cls._handlers[type_name]()
            return cls._instances[type_name]
        else:
            return cls._handlers[type_name]()

    @classmethod
    def list_types(cls) -> list[str]:
        """List all registered type names."""
        return sorted(cls._handlers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear registry (mainly for testing)."""
        cls._handlers.clear()
        cls._instances.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def register_type(
    type_name: str | None = None,
    *,
    override: bool = False,
) -> Callable[[type[DataTypeHandler]], type[DataTypeHandler]]:
    """Decorator to register a type handler.

    Example:
        @register_type("math")
        class MathHandler(BaseDataTypeHandler):
            type_name = "math"
            ...
    """
    return TypeRegistry.register(type_name, override=override)


def get_type_handler(type_name: str) -> DataTypeHandler:
    """Get handler instance for type.

    Example:
        handler = get_type_handler("tool_call")
        rewards = handler.get_reward_config()
    """
    return TypeRegistry.get(type_name)


def list_types() -> list[str]:
    """List all registered type names."""
    return TypeRegistry.list_types()


def detect_dataset_type(sample: dict[str, Any]) -> str | None:
    """Auto-detect dataset type from sample.

    Checks each registered handler's validate_sample() method.
    Returns first matching type, or None if no match.

    Args:
        sample: Data sample to check

    Returns:
        Type name or None
    """
    # Try explicit type field first
    if "type" in sample:
        explicit_type = sample["type"]
        if explicit_type in TypeRegistry._handlers:
            return explicit_type

    # Try validation methods
    for type_name in TypeRegistry.list_types():
        handler = TypeRegistry.get(type_name)
        if handler.validate_sample(sample):
            return type_name

    return None


# =============================================================================
# CONTEXT MANAGER FOR TEMPORARY OVERRIDES
# =============================================================================

class temporary_handler:
    """Context manager for temporary type handler override.

    Example:
        with temporary_handler("math", custom_handler):
            # Use custom handler
            handler = get_type_handler("math")
        # Original handler restored
    """

    def __init__(self, type_name: str, handler: DataTypeHandler):
        self.type_name = type_name
        self.new_handler = handler
        self.old_handler = None

    def __enter__(self):
        self.old_handler = TypeRegistry._handlers.get(self.type_name)
        TypeRegistry._handlers[self.type_name] = type(self.new_handler)
        TypeRegistry._instances[self.type_name] = self.new_handler
        return self.new_handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_handler is not None:
            TypeRegistry._handlers[self.type_name] = self.old_handler
        else:
            TypeRegistry._handlers.pop(self.type_name, None)

        TypeRegistry._instances.pop(self.type_name, None)


# =============================================================================
# COMPOSITION HELPERS
# =============================================================================

def compose_rewards(*configs: RewardConfig) -> RewardConfig:
    """Compose multiple reward configs into one.

    Example:
        base = RewardConfig(functions=("r1_correctness",), weights=(0.5,))
        extra = RewardConfig(functions=("r1_format",), weights=(0.5,))
        combined = compose_rewards(base, extra)
    """
    all_functions = []
    all_weights = []

    for config in configs:
        all_functions.extend(config.functions)
        all_weights.extend(config.normalized_weights)

    return RewardConfig(
        functions=tuple(all_functions),
        weights=tuple(all_weights),
    )


def with_additional_rewards(
    base: RewardConfig,
    additional: dict[str, float],
) -> RewardConfig:
    """Add rewards to existing config.

    Args:
        base: Base reward config
        additional: Dict of {function_name: weight}

    Returns:
        New config with combined rewards
    """
    funcs = list(base.functions) + list(additional.keys())
    weights = list(base.normalized_weights) + list(additional.values())

    return RewardConfig(
        functions=tuple(funcs),
        weights=tuple(weights),
    )
