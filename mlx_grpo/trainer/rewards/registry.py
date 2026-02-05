"""Reward function registry with decorator-based registration.

This module implements the Registry pattern for reward functions, enabling:
- Declarative registration via decorators
- Runtime discovery of available rewards
- Default reward configuration
- Type-safe function signatures

Design Patterns:
- Registry Pattern: Centralized function registration
- Decorator Pattern: Non-invasive function enhancement
- Strategy Pattern: Interchangeable reward algorithms

SOLID Principles:
- Open/Closed: New rewards added without modifying registry code
- Interface Segregation: Rewards only need to match the signature
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from functools import wraps
from typing import TypeAlias

__all__ = [
    "RewardFunc",
    "reward",
    "get_reward",
    "list_rewards",
    "get_defaults",
    "register_reward",
    "REWARD_REGISTRY",
]

logger = logging.getLogger(__name__)

# Type alias for reward function signature
RewardFunc: TypeAlias = Callable[[list[str], list[str], list[str], list[str] | None], list[float]]

# Global registry - maps name -> function
REWARD_REGISTRY: dict[str, RewardFunc] = {}

# Track default rewards
_DEFAULT_REWARDS: set[str] = set()


def reward(
    name: str | None = None,
    *,
    default: bool = False,
    validate: bool = True,
) -> Callable[[RewardFunc], RewardFunc]:
    """Decorator to register reward functions.

    This is the primary way to register new reward functions.
    The decorator handles:
    - Name resolution (explicit or from function name)
    - Input validation (optional)
    - Default marking

    Args:
        name: Registry name. If None, uses function.__name__
        default: If True, includes in get_defaults()
        validate: If True, wraps with input validation

    Returns:
        Decorator function

    Example:
        @reward("accuracy", default=True)
        def accuracy_reward(prompts, completions, answers, types=None):
            return [1.0 if c == a else 0.0 for c, a in zip(completions, answers)]

        # Or using function name:
        @reward(default=True)
        def format_reward(prompts, completions, answers, types=None):
            ...
    """

    def decorator(func: RewardFunc) -> RewardFunc:
        key = name if name is not None else func.__name__

        if key in REWARD_REGISTRY:
            logger.warning(f"Reward '{key}' already registered, overwriting")

        # Optionally wrap with validation
        wrapped = _validate_wrapper(func) if validate else func

        # Store metadata on function
        wrapped._reward_name = key  # type: ignore
        wrapped._is_default = default  # type: ignore

        # Register
        REWARD_REGISTRY[key] = wrapped

        if default:
            _DEFAULT_REWARDS.add(key)

        logger.debug(f"Registered reward: {key} (default={default})")
        return wrapped

    return decorator


def register_reward(
    name: str,
    func: RewardFunc,
    *,
    default: bool = False,
    validate: bool = True,
) -> RewardFunc:
    """Programmatic reward registration (non-decorator form).

    Use this for dynamic registration or when decorators aren't practical.

    Args:
        name: Registry name
        func: Reward function
        default: If True, includes in get_defaults()
        validate: If True, wraps with input validation

    Returns:
        The (possibly wrapped) registered function

    Example:
        def my_custom_reward(prompts, completions, answers, types=None):
            return [0.5] * len(completions)

        register_reward("custom", my_custom_reward, default=True)
    """
    return reward(name, default=default, validate=validate)(func)


def get_reward(name: str) -> RewardFunc:
    """Get reward function by name.

    Args:
        name: Registry name

    Returns:
        The registered reward function

    Raises:
        KeyError: If reward not found

    Example:
        accuracy_fn = get_reward("accuracy")
        scores = accuracy_fn(prompts, completions, answers)
    """
    if name not in REWARD_REGISTRY:
        available = ", ".join(sorted(REWARD_REGISTRY.keys()))
        raise KeyError(f"Unknown reward: '{name}'. Available: [{available}]")
    return REWARD_REGISTRY[name]


def list_rewards() -> list[str]:
    """List all registered reward names.

    Returns:
        Sorted list of reward names
    """
    return sorted(REWARD_REGISTRY.keys())


def get_defaults() -> list[RewardFunc]:
    """Get all default reward functions.

    Returns default rewards in a deterministic order.

    Returns:
        List of default reward functions
    """
    return [REWARD_REGISTRY[name] for name in sorted(_DEFAULT_REWARDS) if name in REWARD_REGISTRY]


def clear_registry() -> None:
    """Clear all registered rewards. Mainly for testing."""
    REWARD_REGISTRY.clear()
    _DEFAULT_REWARDS.clear()


# =============================================================================
# Input Validation
# =============================================================================


def _validate_wrapper(func: RewardFunc) -> RewardFunc:
    """Wrap reward function with input validation and error handling.

    Ensures:
    - All lists have matching lengths (with padding if needed)
    - Scores are bounded to [0, 1]
    - NaN/Inf values are replaced with 0.0
    - Exceptions are caught and logged
    """

    @wraps(func)
    def wrapper(
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: list[str] | None = None,
    ) -> list[float]:
        # Empty input
        if not completions:
            return []

        n = len(completions)

        # Align prompts
        aligned_prompts = _align_list(prompts, n) if prompts else [""] * n

        # Align answers
        aligned_answers = _align_list(answers, n) if answers else [""] * n

        # Align types
        aligned_types = _align_list(types, n) if types else None

        try:
            scores = func(aligned_prompts, completions, aligned_answers, aligned_types)

            # Validate and clamp scores
            return [_sanitize_score(s) for s in scores]

        except Exception as e:
            logger.error(f"Reward '{func.__name__}' failed: {e}", exc_info=True)
            return [0.0] * n

    return wrapper


def _align_list(lst: list[str] | None, target_len: int) -> list[str]:
    """Align list to target length by padding or truncating."""
    if not lst:
        return [""] * target_len

    if len(lst) == target_len:
        return lst
    elif len(lst) == 1:
        return lst * target_len
    elif len(lst) > target_len:
        return lst[:target_len]
    else:
        return lst + [lst[-1]] * (target_len - len(lst))


def _sanitize_score(score: float | int) -> float:
    """Sanitize a single score value."""
    if not isinstance(score, (int, float)):
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, min(1.0, float(score)))
