"""
Extensible Type System for MLX-GRPO
====================================

Provides a plugin architecture for dataset types, rewards, and generation strategies.

Features:
- Auto-detection of dataset types
- Dynamic class loading based on type
- Type-specific reward configurations
- Type-specific generation strategies
- Easy extension for new types

Usage:
    # Register a new type
    @register_type("my_custom_type")
    class MyTypeHandler(DataTypeHandler):
        def get_rewards(self) -> list[str]:
            return ["custom_reward1", "custom_reward2"]

        def get_generation_strategy(self) -> GenerationStrategy:
            return GenerationStrategy(
                max_length=512,
                temperature=0.8,
                two_phase=False
            )

    # Auto-detect and use
    handler = get_type_handler(sample_type)
    rewards = handler.get_rewards()
    strategy = handler.get_generation_strategy()
"""

from .registry import (
    DataTypeHandler,
    GenerationStrategy,
    RewardConfig,
    register_type,
    get_type_handler,
    list_types,
    detect_dataset_type,
)

from .handlers import (
    ThinkingTypeHandler,
    ToolCallingTypeHandler,
    CodeGenerationTypeHandler,
    MathReasoningTypeHandler,
    DefaultTypeHandler,
)

__all__ = [
    # Core
    "DataTypeHandler",
    "GenerationStrategy",
    "RewardConfig",
    "register_type",
    "get_type_handler",
    "list_types",
    "detect_dataset_type",
    # Built-in handlers
    "ThinkingTypeHandler",
    "ToolCallingTypeHandler",
    "CodeGenerationTypeHandler",
    "MathReasoningTypeHandler",
    "DefaultTypeHandler",
]
