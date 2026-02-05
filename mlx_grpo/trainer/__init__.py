"""GRPO Trainer package for MLX-GRPO.

This package provides:
- GRPO (Group Relative Policy Optimization) training
- Reward function system with decorator-based registration
- Dataset utilities for GRPO training
- Training monitoring and logging

Example:
    from mlx_grpo.trainer import (
        train_grpo,
        GRPOTrainingArgs,
        get_reward,
        reward,
    )

    # Custom reward function
    @reward("my_reward", default=True)
    def my_reward(prompts, completions, answers, types=None):
        return [1.0 for _ in completions]

    # Train GRPO
    train_grpo(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=GRPOTrainingArgs(...),
    )
"""

from __future__ import annotations

# Base module
from .base import (
    BaseTrainingArgs,
    CheckpointCallback,
    RewardFunction,
    TrainingCallback,
    grad_checkpoint,
)

# Datasets
from .datasets import (
    CacheDataset,
    ConcatenatedDataset,
    GRPODataset,
    create_dataset,
    load_dataset,
)

# GRPO modular components
from .grpo import (  # Debug utilities; Checkpoint management; Layer utilities; Curriculum learning; Generation; Loss
    CheckpointManager,
    build_curriculum_prefix,
    calculate_rewards_and_advantages,
    compute_curriculum_ratio,
    dbg,
    dbg_mem,
    freeze_model_layers,
    generate_grpo,
    grpo_loss,
    mem_stats,
    parse_layer_spec,
    safe_clear,
    safe_eval,
)

# GRPO trainer (main entry point)
from .grpo_trainer import (
    GRPOTrainingArgs,
    evaluate_grpo,
    train_grpo,
)

# Rewards
from .rewards import (  # Backward compatibility
    REWARD_REGISTRY,
    RewardFunc,
    get_default_reward_functions,
    get_defaults,
    get_reward,
    get_reward_function,
    list_available_reward_functions,
    list_rewards,
    register_reward,
    reward,
)

__all__ = [
    # Base
    "BaseTrainingArgs",
    "RewardFunction",
    "TrainingCallback",
    "CheckpointCallback",
    "grad_checkpoint",
    # GRPO main
    "GRPOTrainingArgs",
    "train_grpo",
    "evaluate_grpo",
    # GRPO modular components
    "safe_eval",
    "safe_clear",
    "dbg",
    "dbg_mem",
    "mem_stats",
    "CheckpointManager",
    "parse_layer_spec",
    "freeze_model_layers",
    "compute_curriculum_ratio",
    "build_curriculum_prefix",
    "generate_grpo",
    "grpo_loss",
    "calculate_rewards_and_advantages",
    # Rewards (new API)
    "reward",
    "get_reward",
    "list_rewards",
    "get_defaults",
    "register_reward",
    "REWARD_REGISTRY",
    "RewardFunc",
    # Rewards (backward compatibility)
    "get_reward_function",
    "get_default_reward_functions",
    "list_available_reward_functions",
    # Datasets
    "GRPODataset",
    "CacheDataset",
    "ConcatenatedDataset",
    "create_dataset",
    "load_dataset",
]
