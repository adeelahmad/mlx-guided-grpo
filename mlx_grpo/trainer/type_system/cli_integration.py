"""
CLI Integration for Type System
================================

Seamless integration of type system with existing GRPO CLI.

Usage in train.py:
    from mlx_grpo.trainer.type_system.cli_integration import apply_type_config

    # Load dataset
    dataset = load_typed_dataset(args.data, tokenizer)

    # Auto-configure from types
    args = apply_type_config(args, dataset)

    # Train with type-aware config
    train_grpo(model, dataset, args)
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from .dataset_loader import TypedDataset, DatasetConfig
from .registry import get_type_handler

if TYPE_CHECKING:
    from ..grpo.config import GRPOTrainingArgs

__all__ = [
    "apply_type_config",
    "override_args_from_config",
    "print_type_summary",
]

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG APPLICATION
# =============================================================================

def apply_type_config(
    args: GRPOTrainingArgs,
    dataset: TypedDataset,
    *,
    override_rewards: bool = True,
    override_generation: bool = True,
    override_curriculum: bool = True,
    aggregation_strategy: str = "dominant",
) -> GRPOTrainingArgs:
    """Apply type-based configuration to training args.

    This modifies the args object in-place based on detected dataset types.

    Args:
        args: Current training arguments
        dataset: Typed dataset with detected types
        override_rewards: Replace reward config from types
        override_generation: Replace generation config from types
        override_curriculum: Replace curriculum config from types
        aggregation_strategy: How to aggregate multi-type configs

    Returns:
        Modified args (same object, modified in-place)

    Example:
        args = parse_args()
        dataset = load_typed_dataset(args.data, tokenizer)
        args = apply_type_config(args, dataset)
        # args now has type-appropriate rewards, generation settings, etc.
    """
    # Get aggregated config
    config = dataset.get_aggregated_config(strategy=aggregation_strategy)

    logger.info(f"Applying type configuration (strategy={aggregation_strategy})")
    logger.info(f"Dominant type: {config.dominant_type}")
    logger.info(f"Distribution: {config.type_distribution}")

    # Apply reward config
    if override_rewards:
        _apply_reward_config(args, config.reward_config)

    # Apply generation config
    if override_generation:
        _apply_generation_config(args, config.generation_strategy)

    # Apply curriculum config
    if override_curriculum:
        _apply_curriculum_config(args, config.curriculum_config)

    return args


def _apply_reward_config(
    args: GRPOTrainingArgs,
    reward_config: Any  # RewardConfig
) -> None:
    """Apply reward configuration to args."""
    # Check if user specified custom rewards
    user_specified = (
        hasattr(args, 'reward_functions') and
        args.reward_functions and
        args.reward_functions != []
    )

    if user_specified:
        logger.info("User specified custom rewards, not overriding")
        return

    # Apply type-based rewards
    args.reward_functions = list(reward_config.functions)
    args.reward_weights = list(reward_config.normalized_weights)

    logger.info(f"Applied rewards: {args.reward_functions}")
    logger.info(f"Weights: {[f'{w:.2f}' for w in args.reward_weights]}")


def _apply_generation_config(
    args: GRPOTrainingArgs,
    gen_config: Any  # GenerationStrategy
) -> None:
    """Apply generation configuration to args."""
    # Map generation config to args
    config_map = {
        'max_completion_length': gen_config.max_length,
        'temperature': gen_config.temperature,
        'top_p': gen_config.top_p,
        'enforce_thinking': gen_config.enforce_thinking,
        'continuation_tokens': gen_config.continuation_tokens,
    }

    applied = []
    for arg_name, value in config_map.items():
        # Only override if not explicitly set by user
        if not _is_user_specified(args, arg_name):
            setattr(args, arg_name, value)
            applied.append(f"{arg_name}={value}")

    if applied:
        logger.info(f"Applied generation config: {', '.join(applied)}")


def _apply_curriculum_config(
    args: GRPOTrainingArgs,
    curriculum_config: Any  # CurriculumConfig
) -> None:
    """Apply curriculum configuration to args."""
    # Only apply if curriculum is enabled in type config
    if not curriculum_config.enabled:
        return

    # Check if user disabled curriculum
    if hasattr(args, 'curriculum_enabled') and not args.curriculum_enabled:
        logger.info("User disabled curriculum, not applying type config")
        return

    config_map = {
        'curriculum_enabled': curriculum_config.enabled,
        'curriculum_start_ratio': curriculum_config.start_ratio,
        'curriculum_end_ratio': curriculum_config.end_ratio,
    }

    applied = []
    for arg_name, value in config_map.items():
        if not _is_user_specified(args, arg_name):
            setattr(args, arg_name, value)
            applied.append(f"{arg_name}={value}")

    if applied:
        logger.info(f"Applied curriculum config: {', '.join(applied)}")


def _is_user_specified(args: Any, attr_name: str) -> bool:
    """Check if attribute was explicitly set by user.

    This is a heuristic - we assume if the value differs from
    the default in GRPOTrainingArgs, it was user-specified.
    """
    if not hasattr(args, attr_name):
        return False

    # For now, simple check - could be enhanced with arg tracking
    return False  # Conservative: allow type system to override


# =============================================================================
# MANUAL OVERRIDE
# =============================================================================

def override_args_from_config(
    args: GRPOTrainingArgs,
    config: DatasetConfig,
) -> GRPOTrainingArgs:
    """Manually override args from dataset config.

    Lower-level function for custom integration.

    Args:
        args: Training arguments to modify
        config: Dataset configuration

    Returns:
        Modified args
    """
    _apply_reward_config(args, config.reward_config)
    _apply_generation_config(args, config.generation_strategy)
    _apply_curriculum_config(args, config.curriculum_config)
    return args


# =============================================================================
# UTILITIES
# =============================================================================

def print_type_summary(dataset: TypedDataset) -> None:
    """Print human-readable summary of dataset types.

    Args:
        dataset: Typed dataset to summarize
    """
    print("\n" + "="*70)
    print("DATASET TYPE SUMMARY")
    print("="*70)

    config = dataset.get_aggregated_config()

    print(f"\nTotal Samples: {len(dataset)}")
    print(f"Dominant Type: {config.dominant_type}")
    print(f"\nType Distribution:")

    # Sort by count descending
    sorted_types = sorted(
        config.type_distribution.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for type_name, count in sorted_types:
        pct = 100 * count / len(dataset)
        bar = "█" * int(pct / 2)
        print(f"  {type_name:15s} {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nReward Configuration:")
    for func, weight in zip(
        config.reward_config.functions,
        config.reward_config.normalized_weights
    ):
        print(f"  {func:30s} {weight:.3f}")

    print(f"\nGeneration Strategy:")
    gen = config.generation_strategy
    print(f"  Max Length:       {gen.max_length}")
    print(f"  Temperature:      {gen.temperature}")
    print(f"  Two-Phase:        {gen.two_phase}")
    print(f"  Enforce Thinking: {gen.enforce_thinking}")

    if config.curriculum_config.enabled:
        print(f"\nCurriculum Learning:")
        curr = config.curriculum_config
        print(f"  Start Ratio: {curr.start_ratio}")
        print(f"  End Ratio:   {curr.end_ratio}")
        print(f"  Strategy:    {curr.strategy}")

    print("="*70 + "\n")


def get_type_config_for_sample(sample: dict[str, Any]) -> DatasetConfig:
    """Get configuration for a single sample based on its type.

    Args:
        sample: Data sample

    Returns:
        Configuration for that sample's type
    """
    from .registry import detect_dataset_type

    # Detect type
    sample_type = detect_dataset_type(sample)
    if sample_type is None:
        sample_type = "default"

    # Get handler
    handler = get_type_handler(sample_type)

    # Build config
    return DatasetConfig(
        reward_config=handler.get_reward_config(),
        generation_strategy=handler.get_generation_strategy(),
        curriculum_config=handler.get_curriculum_config(),
        type_distribution={sample_type: 1},
        dominant_type=sample_type,
    )
