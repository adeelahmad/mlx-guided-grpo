"""Reward functions package for GRPO training.

This package provides:
- Decorator-based reward function registration
- Core reward functions (accuracy, format, quality)
- Exam-specific rewards
- Hierarchical reward system

Usage:
    from mlx_grpo.trainer.rewards import (
        reward,           # Decorator for custom rewards
        get_reward,       # Get reward by name
        list_rewards,     # List all available rewards
        get_defaults,     # Get default reward functions
    )

    # Register custom reward
    @reward("my_reward", default=True)
    def my_reward(prompts, completions, answers, types=None):
        return [1.0 for _ in completions]

    # Use registered reward
    accuracy_fn = get_reward("r1_accuracy_reward_func")
    scores = accuracy_fn(prompts, completions, answers)

Architecture:
    rewards/
    ├── __init__.py      # This file - public API
    ├── registry.py      # Decorator-based registration system
    └── hierarchical/    # Multi-level reward hierarchy (from hierarchical_rewards_v3)

The core reward functions are defined in grpo_reward_functions.py and
automatically registered when that module is imported.
"""

from __future__ import annotations

from .registry import (
    REWARD_REGISTRY,
    RewardFunc,
    clear_registry,
    get_defaults,
    get_reward,
    list_rewards,
    register_reward,
    reward,
)


def _get_backward_compat_functions():
    """Lazy import to avoid circular dependency."""
    from ..grpo_reward_functions import (
        RewardFunctions,
        get_default_reward_functions,
        get_reward_function,
        list_available_reward_functions,
        register_reward_function,
    )

    return {
        "get_reward_function": get_reward_function,
        "get_default_reward_functions": get_default_reward_functions,
        "list_available_reward_functions": list_available_reward_functions,
        "register_reward_function": register_reward_function,
        "RewardFunctions": RewardFunctions,
    }


# Lazy attribute access for backward compatibility
def __getattr__(name: str):
    compat_names = {
        "get_reward_function",
        "get_default_reward_functions",
        "list_available_reward_functions",
        "register_reward_function",
        "RewardFunctions",
        "r1_accuracy_reward_func",
        "r1_soft_format_reward_func",
        "r1_strict_format_reward_func",
        "r1_int_reward_func",
        "r1_count_xml",
    }
    exam_names = {
        "exam_accuracy_reward_func",
        "exam_format_reward_func",
        "exam_reasoning_reward_func",
        "exam_combined_reward_func",
        "exam_strict_format_reward_func",
        "exam_accuracy",
        "exam_format",
        "exam_reasoning",
        "exam_combined",
        "exam_strict",
    }
    if name in compat_names:
        from .. import grpo_reward_functions

        return getattr(grpo_reward_functions, name)
    if name in exam_names:
        from .. import exam_reward

        return getattr(exam_reward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # New registry API
    "reward",
    "get_reward",
    "list_rewards",
    "get_defaults",
    "register_reward",
    "clear_registry",
    "REWARD_REGISTRY",
    "RewardFunc",
    # Backward compatibility (via __getattr__)
    "get_reward_function",
    "get_default_reward_functions",
    "list_available_reward_functions",
    "register_reward_function",
    "RewardFunctions",
    # Core reward functions (via __getattr__)
    "r1_accuracy_reward_func",
    "r1_soft_format_reward_func",
    "r1_strict_format_reward_func",
    "r1_int_reward_func",
    "r1_count_xml",
    # Exam reward functions (via __getattr__)
    "exam_accuracy_reward_func",
    "exam_format_reward_func",
    "exam_reasoning_reward_func",
    "exam_combined_reward_func",
    "exam_strict_format_reward_func",
]
