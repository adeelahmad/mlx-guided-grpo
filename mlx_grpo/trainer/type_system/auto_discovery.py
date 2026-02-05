"""
Auto-Discovery Type System
===========================

Elegant naming convention-based system for automatic class discovery.

Pattern:
    type="math" → looks for:
    - MathReward (or base Reward)
    - MathGenerationStrategy (or base GenerationStrategy)
    - MathDataLoader (or base DataLoader)

Features:
- Convention over configuration
- Automatic class discovery
- Graceful fallback to base classes
- Dynamic import support
- Metaclass-based registration

Usage:
    # Define a reward for 'math' type
    class MathReward(BaseReward):
        def compute(self, ...):
            return [1.0]

    # Automatically discovered when type="math"
    reward = get_reward_for_type("math")
    # Returns MathReward instance

Example Module Structure:
    rewards/
        base_reward.py      # BaseReward
        math_reward.py      # MathReward
        code_reward.py      # CodeReward
        tool_reward.py      # ToolReward

    generation/
        base_strategy.py    # BaseGenerationStrategy
        math_strategy.py    # MathGenerationStrategy
"""

from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Type, Optional
from functools import lru_cache

__all__ = [
    "BaseReward",
    "BaseGenerationStrategy",
    "BaseDataLoader",
    "get_reward_for_type",
    "get_generation_strategy_for_type",
    "get_data_loader_for_type",
    "discover_class",
    "register_discovery_path",
]

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# DISCOVERY REGISTRY
# =============================================================================

class DiscoveryRegistry:
    """Registry for class discovery paths.

    Maps component types to module paths for auto-discovery.
    """

    _paths: dict[str, list[str]] = {
        "reward": [
            "mlx_grpo.trainer.type_system.rewards",
            "mlx_grpo.trainer.rewards",
        ],
        "generation": [
            "mlx_grpo.trainer.type_system.generation",
            "mlx_grpo.trainer.generation",
        ],
        "loader": [
            "mlx_grpo.trainer.type_system.loaders",
            "mlx_grpo.trainer.loaders",
        ],
    }

    @classmethod
    def add_path(cls, component_type: str, module_path: str) -> None:
        """Add discovery path for component type."""
        if component_type not in cls._paths:
            cls._paths[component_type] = []
        if module_path not in cls._paths[component_type]:
            cls._paths[component_type].append(module_path)
            logger.debug(f"Added discovery path: {component_type} -> {module_path}")

    @classmethod
    def get_paths(cls, component_type: str) -> list[str]:
        """Get discovery paths for component type."""
        return cls._paths.get(component_type, [])


def register_discovery_path(component_type: str, module_path: str) -> None:
    """Register a module path for auto-discovery.

    Args:
        component_type: Type of component ('reward', 'generation', 'loader')
        module_path: Python module path to search

    Example:
        register_discovery_path("reward", "my_project.custom_rewards")
    """
    DiscoveryRegistry.add_path(component_type, module_path)


# =============================================================================
# AUTO-DISCOVERY FUNCTION
# =============================================================================

@lru_cache(maxsize=128)
def discover_class(
    data_type: str,
    component_type: str,
    base_class: Type[T],
    suffix: str = "",
) -> Type[T]:
    """Discover class for data type using naming conventions.

    Naming pattern:
        {DataType}{Suffix} (e.g., MathReward, ToolGenerationStrategy)

    Search order:
    1. Exact match: MathReward
    2. Lowercase: math_reward
    3. Base class fallback

    Args:
        data_type: Data type name (e.g., "math", "tool_call")
        component_type: Component category ("reward", "generation", "loader")
        base_class: Base class to fall back to
        suffix: Class name suffix (e.g., "Reward", "Strategy")

    Returns:
        Discovered class or base class

    Example:
        RewardClass = discover_class("math", "reward", BaseReward, "Reward")
        # Looks for: MathReward, math_reward, then returns BaseReward
    """
    # Normalize data type to class name format
    # "tool_call" -> "ToolCall", "math" -> "Math"
    class_name_base = "".join(word.capitalize() for word in data_type.split("_"))
    class_name = f"{class_name_base}{suffix}"

    logger.debug(
        f"Discovering {component_type} for type '{data_type}': "
        f"looking for {class_name}"
    )

    # Get search paths
    search_paths = DiscoveryRegistry.get_paths(component_type)

    # Try each path
    for module_path in search_paths:
        # Try different naming patterns
        patterns = [
            f"{module_path}.{data_type}_{component_type}",  # math_reward
            f"{module_path}.{data_type.lower()}_{suffix.lower()}",  # math_reward
            f"{module_path}.{class_name_base.lower()}",  # math
        ]

        for pattern in patterns:
            try:
                module = importlib.import_module(pattern)

                # Look for class in module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if it matches our class name
                    if name == class_name and issubclass(obj, base_class):
                        logger.info(
                            f"Discovered {class_name} in {pattern}"
                        )
                        return obj

            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not import {pattern}: {e}")
                continue

    # Fallback to base class
    logger.info(
        f"No specific class found for '{data_type}', "
        f"using {base_class.__name__}"
    )
    return base_class


# =============================================================================
# BASE CLASSES
# =============================================================================

class BaseReward:
    """Base class for reward functions.

    Subclasses should implement compute() method.

    Naming convention:
        - Math type: MathReward
        - Tool calling: ToolCallReward or ToolReward
        - Code generation: CodeReward
    """

    def __init__(self):
        self.type_name = self.__class__.__name__.replace("Reward", "").lower()

    def compute(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[str]] = None,
    ) -> list[float]:
        """Compute rewards for batch.

        Default: simple exact match.

        Returns:
            List of reward scores in [0.0, 1.0]
        """
        return [
            1.0 if comp.strip() == ans.strip() else 0.0
            for comp, ans in zip(completions, answers)
        ]

    def get_weight(self) -> float:
        """Default weight for this reward in combinations."""
        return 1.0


class BaseGenerationStrategy:
    """Base class for generation strategies.

    Naming convention:
        - Math type: MathGenerationStrategy
        - Tool calling: ToolGenerationStrategy
    """

    def __init__(self):
        self.type_name = (
            self.__class__.__name__
            .replace("GenerationStrategy", "")
            .replace("Strategy", "")
            .lower()
        )

    def get_max_length(self) -> int:
        """Maximum generation length."""
        return 512

    def get_temperature(self) -> float:
        """Sampling temperature."""
        return 0.8

    def get_top_p(self) -> float:
        """Nucleus sampling parameter."""
        return 0.95

    def use_two_phase(self) -> bool:
        """Whether to use two-phase generation."""
        return False

    def enforce_thinking(self) -> bool:
        """Whether to enforce thinking tags."""
        return False


class BaseDataLoader:
    """Base class for data loaders.

    Naming convention:
        - Math type: MathDataLoader
        - Tool calling: ToolDataLoader
    """

    def __init__(self):
        self.type_name = (
            self.__class__.__name__
            .replace("DataLoader", "")
            .replace("Loader", "")
            .lower()
        )

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load dataset from path.

        Default: load JSONL format.
        """
        import json
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def preprocess(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a sample (optional)."""
        return sample

    def validate(self, sample: dict[str, Any]) -> bool:
        """Validate sample (optional)."""
        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_reward_for_type(data_type: str) -> BaseReward:
    """Get reward instance for data type.

    Args:
        data_type: Data type (e.g., "math", "tool_call")

    Returns:
        Reward instance (type-specific or base)

    Example:
        reward = get_reward_for_type("math")
        scores = reward.compute(prompts, completions, answers)
    """
    RewardClass = discover_class(
        data_type=data_type,
        component_type="reward",
        base_class=BaseReward,
        suffix="Reward",
    )
    return RewardClass()


def get_generation_strategy_for_type(data_type: str) -> BaseGenerationStrategy:
    """Get generation strategy for data type.

    Args:
        data_type: Data type

    Returns:
        Generation strategy instance

    Example:
        strategy = get_generation_strategy_for_type("math")
        max_len = strategy.get_max_length()
    """
    StrategyClass = discover_class(
        data_type=data_type,
        component_type="generation",
        base_class=BaseGenerationStrategy,
        suffix="GenerationStrategy",
    )
    return StrategyClass()


def get_data_loader_for_type(data_type: str) -> BaseDataLoader:
    """Get data loader for data type.

    Args:
        data_type: Data type

    Returns:
        Data loader instance

    Example:
        loader = get_data_loader_for_type("tool_call")
        data = loader.load("data.jsonl")
    """
    LoaderClass = discover_class(
        data_type=data_type,
        component_type="loader",
        base_class=BaseDataLoader,
        suffix="DataLoader",
    )
    return LoaderClass()


# =============================================================================
# BATCH DISCOVERY
# =============================================================================

def discover_all_for_type(data_type: str) -> dict[str, Any]:
    """Discover all components for a data type.

    Args:
        data_type: Data type

    Returns:
        Dict with 'reward', 'generation', 'loader' keys

    Example:
        components = discover_all_for_type("math")
        reward = components['reward']
        strategy = components['generation']
        loader = components['loader']
    """
    return {
        'reward': get_reward_for_type(data_type),
        'generation': get_generation_strategy_for_type(data_type),
        'loader': get_data_loader_for_type(data_type),
    }


# =============================================================================
# METACLASS FOR AUTO-REGISTRATION
# =============================================================================

from abc import ABCMeta as _ABCMeta

class AutoRegisterMeta(_ABCMeta):
    """Metaclass that auto-registers classes based on naming convention.

    Classes using this metaclass are automatically discoverable.
    """

    _registry: dict[str, dict[str, Type]] = {
        'reward': {},
        'generation': {},
        'loader': {},
    }

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # Determine component type from base classes
        component_type = None
        if any(issubclass(b, BaseReward) for b in bases if b != BaseReward):
            component_type = 'reward'
        elif any(issubclass(b, BaseGenerationStrategy) for b in bases if b != BaseGenerationStrategy):
            component_type = 'generation'
        elif any(issubclass(b, BaseDataLoader) for b in bases if b != BaseDataLoader):
            component_type = 'loader'

        # Register if it's a concrete subclass
        if component_type and name not in ('BaseReward', 'BaseGenerationStrategy', 'BaseDataLoader'):
            # Extract type name from class name
            # MathReward -> math, ToolCallReward -> tool_call
            for suffix in ('Reward', 'GenerationStrategy', 'DataLoader', 'Strategy', 'Loader'):
                if name.endswith(suffix):
                    type_name = name[:-len(suffix)]
                    # Convert CamelCase to snake_case
                    import re
                    type_name = re.sub(r'(?<!^)(?=[A-Z])', '_', type_name).lower()

                    mcs._registry[component_type][type_name] = cls
                    logger.debug(f"Auto-registered {name} as {component_type} for '{type_name}'")
                    break

        return cls

    @classmethod
    def get_class(mcs, component_type: str, type_name: str) -> Optional[Type]:
        """Get registered class."""
        return mcs._registry.get(component_type, {}).get(type_name)
