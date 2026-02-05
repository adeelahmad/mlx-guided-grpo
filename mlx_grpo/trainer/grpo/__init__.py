"""GRPO Training package - modular implementation.

This package provides a clean, SOLID-compliant implementation of
Group Relative Policy Optimization (GRPO) training for MLX.

Modules:
    debug: Debug and crash tracing utilities
    layers: Layer-level gradient control
    checkpoint: Checkpoint management
    config: Training configuration (GRPOTrainingArgs)
    curriculum: Curriculum learning and truncation
    generation: GRPO-specific text generation
    loss: Loss computation and reward calculation

Usage:
    from mlx_grpo.trainer.grpo import (
        GRPOTrainingArgs,
        generate_grpo,
        grpo_loss,
        calculate_rewards_and_advantages,
    )

SOLID Principles Applied:
- Single Responsibility: Each module handles one concern
- Open/Closed: Extensions via configuration, not modification
- Liskov Substitution: Protocol-based interfaces
- Interface Segregation: Focused module APIs
- Dependency Inversion: Modules depend on abstractions
"""
from __future__ import annotations

# Debug utilities
from .debug import (
    DEBUG,
    DEBUG_MEMORY,
    dbg,
    dbg_mem,
    get_current_iteration,
    get_eval_history,
    iter_end,
    iter_start,
    mem_stats,
    safe_clear,
    safe_eval,
)

# Layer utilities
from .layers import (
    combine_dual_gradients,
    create_layer_gradient_mask,
    detect_thinking_answer_positions,
    freeze_model_layers,
    parse_layer_spec,
)

# Checkpoint management
from .checkpoint import CheckpointManager

# Configuration
from .config import GRPOTrainingArgs

# Curriculum learning
from .curriculum import (
    build_curriculum_prefix,
    compute_curriculum_ratio,
    compute_gradient_alignment,
    extract_thinking_content,
    hierarchical_truncate_thinking,
    interpolate_gradients,
    smart_truncate_completion,
    truncate_thinking_by_ratio,
)

# Generation
from .generation import generate_grpo

# Loss computation
from .loss import (
    calculate_rewards_and_advantages,
    compute_sft_anchor_loss,
    get_per_token_logps,
    get_per_token_logps_with_prompt_mask,
    grpo_loss,
)

# Gradient manipulation
from .gradients import (
    project_gradient_toward_sft,
)

# Corruption detection
from .corruption import (
    EXIT_CODE_CORRUPTION,
    CorruptionError,
    AdapterCorruptionError,
    CompletionCorruptionError,
    validate_adapter_weights,
    validate_completions,
    validate_and_exit_on_corruption,
    log_completion_warnings,
)

__all__ = [
    # Debug
    "DEBUG",
    "DEBUG_MEMORY",
    "dbg",
    "dbg_mem",
    "mem_stats",
    "safe_eval",
    "safe_clear",
    "iter_start",
    "iter_end",
    "get_current_iteration",
    "get_eval_history",
    # Layers
    "parse_layer_spec",
    "create_layer_gradient_mask",
    "combine_dual_gradients",
    "detect_thinking_answer_positions",
    "freeze_model_layers",
    # Checkpoint
    "CheckpointManager",
    # Config
    "GRPOTrainingArgs",
    # Curriculum
    "compute_curriculum_ratio",
    "extract_thinking_content",
    "truncate_thinking_by_ratio",
    "build_curriculum_prefix",
    "hierarchical_truncate_thinking",
    "smart_truncate_completion",
    "compute_gradient_alignment",
    "interpolate_gradients",
    # Generation
    "generate_grpo",
    # Loss
    "grpo_loss",
    "get_per_token_logps",
    "get_per_token_logps_with_prompt_mask",
    "calculate_rewards_and_advantages",
    "compute_sft_anchor_loss",
    # Gradients
    "project_gradient_toward_sft",
    # Corruption detection
    "EXIT_CODE_CORRUPTION",
    "CorruptionError",
    "AdapterCorruptionError",
    "CompletionCorruptionError",
    "validate_adapter_weights",
    "validate_completions",
    "validate_and_exit_on_corruption",
    "log_completion_warnings",
]
