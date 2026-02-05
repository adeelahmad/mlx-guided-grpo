"""GRPO Training - Main training loop implementation.

This module provides the main training functions for GRPO:
- iterate_grpo_batches: Batch iteration for GRPO training
- evaluate_grpo: Evaluation loop
- train_grpo: Main training loop
- train_grpo_with_recovery: Training with crash recovery

All utility functions are imported from the grpo/ submodules:
- grpo.debug: Debug and crash tracing utilities
- grpo.layers: Layer control and gradient masking
- grpo.checkpoint: Checkpoint management
- grpo.config: GRPOTrainingArgs configuration
- grpo.curriculum: Curriculum learning
- grpo.generation: Text generation
- grpo.loss: Loss computation and reward calculation
- grpo.gradients: Gradient manipulation
"""
from __future__ import annotations

import gc as _gc
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.tuner.callbacks import TrainingCallback
from tqdm import tqdm

# Import from base module
from .base import BaseTrainingArgs, grad_checkpoint

# Import all utilities from grpo submodules
from .grpo import (
    # Debug utilities
    safe_eval as _safe_eval,
    safe_clear as _safe_clear,
    iter_start as _iter_start,
    iter_end as _iter_end,
    # Checkpoint management
    CheckpointManager,
    # Configuration
    GRPOTrainingArgs,
    # Layer utilities
    parse_layer_spec,
    create_layer_gradient_mask,
    combine_dual_gradients,
    detect_thinking_answer_positions,
    freeze_model_layers,
    # Curriculum learning
    compute_curriculum_ratio,
    build_curriculum_prefix,
    compute_gradient_alignment,
    interpolate_gradients,
    # Generation
    generate_grpo,
    # Loss and rewards
    grpo_loss,
    get_per_token_logps,
    get_per_token_logps_with_prompt_mask,
    calculate_rewards_and_advantages,
    compute_sft_anchor_loss,
    # Gradient manipulation
    project_gradient_toward_sft,
    # Corruption detection
    EXIT_CODE_CORRUPTION,
    validate_adapter_weights,
    validate_and_exit_on_corruption,
    log_completion_warnings,
)

# Import reward functions
from .grpo_reward_functions import (
    RewardFunctions,
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)

# Import exam reward function
from .exam_reward import (
    compute_reward as exam_compute_reward,
    compute_accuracy_reward as exam_accuracy_reward,
    extract_answer_from_completion,
    RewardWeights as ExamRewardWeights,
)

# Import logging and monitoring
from .rollout_logger import RolloutLogger, RolloutLoggerConfig
from .training_monitor import TrainingMonitor, MonitorConfig, ThresholdConfig

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Tuple


__all__ = [
    "GRPOTrainingArgs",
    "iterate_grpo_batches",
    "evaluate_grpo",
    "train_grpo",
    "train_grpo_with_recovery",
]


# =============================================================================
# BATCH ITERATION
# =============================================================================


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    """Iterate over batches for GRPO training.

    Args:
        dataset: List of (prompt_tokens, answer_tokens, prompt_str, answer_str[, type]) tuples
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        train: If True, shuffle batches

    Yields:
        Tuple of (prompts_tokens, answers_tokens, prompts_text, answers_text, types)
    """
    has_types = isinstance(dataset[0], tuple) and len(dataset[0]) == 5

    if (
        not dataset
        or not isinstance(dataset[0], tuple)
        or (not has_types and len(dataset[0]) != 4)
    ):
        raise ValueError(
            "Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str[, type]) tuples"
        )

    def length_key(i):
        return len(dataset[i][0]) + len(dataset[i][1])

    idx = sorted(range(len(dataset)), key=length_key)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i : i + batch_size : step]

    while True:
        indices = (
            np.random.permutation(list(batch_index_generator()))
            if train
            else batch_index_generator()
        )

        for batch_idx in indices:
            current_batch = [dataset[j] for j in batch_idx]

            prompts_tokens = [item[0] for item in current_batch]
            answers_tokens = [item[1] for item in current_batch]
            prompts_text = [item[2] for item in current_batch]
            answers_text = [item[3] for item in current_batch]
            types = [item[4] for item in current_batch] if has_types else None

            yield prompts_tokens, answers_tokens, prompts_text, answers_text, types

        if not train:
            break


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    epsilon: float,
    epsilon_high: float,
    group_size: int,
    max_seq_length: int,
    max_tokens: int,
    temperature: float,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    reward_weights: Optional[List[float]] = None,
    loss_fn: Callable = grpo_loss,
    iterate_batches: Callable = iterate_grpo_batches,
    grpo_loss_type: str = "grpo",
    importance_sampling_level: str = "token",
    end_answer_token: str = "</answer>"
):
    """Evaluate GRPO model on a dataset.

    Args:
        model: The policy model
        ref_model: Reference model (can be None)
        dataset: Evaluation dataset
        tokenizer: Tokenizer
        batch_size: Batch size
        num_batches: Number of batches to evaluate (-1 for all)
        beta: KL penalty coefficient
        epsilon: Lower-bound epsilon for clipping
        epsilon_high: Upper-bound epsilon for clipping
        group_size: Number of completions per prompt
        max_seq_length: Maximum sequence length
        max_tokens: Maximum completion tokens
        temperature: Sampling temperature
        reward_funcs: List of reward functions
        reward_weights: Weights for each reward function
        loss_fn: Loss function to use
        iterate_batches: Batch iterator function
        grpo_loss_type: Type of loss ('grpo', 'bnpo', 'dr_grpo')
        importance_sampling_level: 'token', 'sequence', or None
        end_answer_token: End token for generation

    Returns:
        Tuple of (avg_loss, ntokens, avg_metrics)
    """
    if reward_funcs is None:
        reward_funcs = [
            r1_accuracy_reward_func,
            r1_int_reward_func,
            r1_strict_format_reward_func,
            r1_soft_format_reward_func,
            r1_count_xml,
        ]

    all_losses = 0
    ntokens = 0
    all_metrics = None

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        prompt_tokens, answer_tokens, prompt_text, answer_text, type_info = batch

        all_completions, all_completion_texts, batch_indices, _, _, all_scaffold_token_counts = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            group_size=group_size,
            temperature=temperature,
            batch_size=batch_size,
            end_token=end_answer_token,
            type_info=type_info,
            generation_sub_batch_size=1,  # Generate one at a time to avoid GPU timeout
        )

        # Prepare expanded data for reward calculation
        expanded_answers = []
        expanded_prompts = []
        expanded_types = []
        unique_prompt_indices = sorted(set(batch_indices))
        grouped_completions = {idx: [] for idx in unique_prompt_indices}

        for i, completion_idx in enumerate(batch_indices):
            grouped_completions[completion_idx].append(i)

        ordered_completions = []
        ordered_completion_texts = []
        ordered_batch_indices = []
        ordered_scaffold_token_counts = []

        for prompt_idx in unique_prompt_indices:
            completion_indices = grouped_completions[prompt_idx]
            for idx in completion_indices:
                ordered_completions.append(all_completions[idx])
                ordered_completion_texts.append(all_completion_texts[idx])
                ordered_batch_indices.append(prompt_idx)
                ordered_scaffold_token_counts.append(all_scaffold_token_counts[idx] if idx < len(all_scaffold_token_counts) else 0)
                expanded_answers.append(answer_text[prompt_idx])
                expanded_prompts.append(prompt_text[prompt_idx])
                expanded_types.append(
                    type_info[prompt_idx] if type_info is not None else None
                )

        # Calculate rewards and advantages
        advantages, reward_metrics, _ = calculate_rewards_and_advantages(
            reward_funcs=reward_funcs,
            expanded_prompts=expanded_prompts,
            all_completion_texts=ordered_completion_texts,
            expanded_answers=expanded_answers,
            expanded_types=expanded_types,
            batch_indices=ordered_batch_indices,
            unique_prompt_indices=unique_prompt_indices,
            reward_weights=reward_weights,
            exam_compute_reward=exam_compute_reward,
        )

        # Compute loss
        losses, toks, metrics = loss_fn(
            model=model,
            ref_model=ref_model,
            batch=(prompt_tokens, answer_tokens, prompt_text, answer_text, type_info),
            completions=ordered_completions,
            completion_texts=ordered_completion_texts,
            batch_indices=ordered_batch_indices,
            advantages=advantages,
            reward_metrics=reward_metrics,
            beta=beta,
            epsilon=epsilon,
            epsilon_high=epsilon_high,
            importance_sampling_level=importance_sampling_level,
            grpo_loss_type=grpo_loss_type,
            max_tokens=max_tokens,
            scaffold_token_counts=ordered_scaffold_token_counts,
        )

        del all_completions, all_completion_texts, batch_indices, all_scaffold_token_counts
        del ordered_completions, ordered_completion_texts, ordered_batch_indices, ordered_scaffold_token_counts
        del advantages, reward_metrics
        mx.clear_cache()

        all_losses += losses * toks
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

    mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, ntokens, avg_metrics


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    args: GRPOTrainingArgs = None,
    loss_fn: Callable = grpo_loss,
    iterate_batches: Callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
    end_answer_token: str = "</answer>"
):
    """Train a model using GRPO (Group Relative Policy Optimization).

    This is the main training function that implements the GRPO algorithm
    with support for:
    - Curriculum learning with thinking scaffolding
    - Dual-gradient mode (CGS) for separate thinking/answer layer training
    - Two-phase generation for incomplete outputs
    - Smart truncation for natural thinking completion
    - SFT anchor with gradient alignment
    - Automatic crash recovery
    - Comprehensive logging and monitoring

    Args:
        model: The policy model to train
        ref_model: Reference model for KL penalty (can be None)
        tokenizer: Tokenizer for encoding/decoding
        optimizer: Optimizer for training
        train_dataset: Training dataset
        val_dataset: Validation dataset
        reward_funcs: List of reward functions (defaults to R1 functions)
        args: Training arguments (GRPOTrainingArgs)
        loss_fn: Loss function to use
        iterate_batches: Batch iterator function
        training_callback: Optional callback for training events
        end_answer_token: End token for generation
    """
    if args is None:
        args = GRPOTrainingArgs()
    if reward_funcs is None:
        reward_funcs = [
            r1_accuracy_reward_func,
            r1_int_reward_func,
            r1_strict_format_reward_func,
            r1_soft_format_reward_func,
            r1_count_xml,
        ]

    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        tqdm.write(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        for layer in model.layers:
            grad_checkpoint(layer)
            break

    grad_accum_steps = args.gradient_accumulation_steps
    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    # Warn about memory usage with gradient accumulation
    if grad_accum_steps > 1 and rank == 0:
        tqdm.write(f"\nWarning: Gradient accumulation ({grad_accum_steps} steps) increases memory usage.")
        tqdm.write(f"   If OOM occurs, try: --gradient-accumulation-steps 1 or reduce --max-completion-length")
        effective_batch = args.batch_size * grad_accum_steps * args.group_size
        tqdm.write(f"   Effective batch size: {args.batch_size} x {grad_accum_steps} x {args.group_size} = {effective_batch}")

    state = [model.state, optimizer.state, mx.random.state]

    # Initialize Rollout Logger (placeholder)
    rollout_logger = None

    # Initialize Checkpoint Manager
    checkpoint_manager = None
    if rank == 0 and (args.keep_last_n_checkpoints > 0 or args.keep_best_n_checkpoints > 0):
        adapter_dir = Path(args.adapter_file).parent if args.adapter_file else Path("./adapters")
        checkpoint_manager = CheckpointManager(
            adapter_dir=adapter_dir,
            keep_last_n=args.keep_last_n_checkpoints,
            keep_best_n=args.keep_best_n_checkpoints,
            metric_name=args.checkpoint_metric,
            higher_is_better=args.checkpoint_metric_higher_is_better,
        )
        checkpoint_manager.scan_existing()
        tqdm.write(f"Checkpoint management enabled: keep_last={args.keep_last_n_checkpoints}, keep_best={args.keep_best_n_checkpoints}")

    # Initialize Training Monitor
    training_monitor = None
    if args.enable_monitor and rank == 0:
        monitor_config = MonitorConfig(
            kl_mean=ThresholdConfig(
                name="KL",
                good_threshold=0.015,
                warning_threshold=args.monitor_kl_warning,
                critical_threshold=args.monitor_kl_critical,
                higher_is_better=False,
            ),
            correctness=ThresholdConfig(
                name="Correctness",
                good_threshold=0.50,
                warning_threshold=args.monitor_reward_warning,
                critical_threshold=args.monitor_reward_critical,
                higher_is_better=True,
            ),
            reward=ThresholdConfig(
                name="Reward",
                good_threshold=0.55,
                warning_threshold=args.monitor_reward_warning,
                critical_threshold=args.monitor_reward_critical,
                higher_is_better=True,
            ),
            stop_on_critical=args.monitor_stop_on_critical,
            critical_count_threshold=args.monitor_critical_count,
        )
        training_monitor = TrainingMonitor(
            config=monitor_config,
            correctness_key="hierarchical_rewards_mean",
            reward_key="total_rewards_mean",
            kl_key="kl",
        )
        tqdm.write(f"Training monitor enabled: KL warning={args.monitor_kl_warning}, critical={args.monitor_kl_critical}")

    # Layer Selection Setup (CGS)
    num_layers = len(model.layers) if hasattr(model, 'layers') else 0
    train_layer_set = None
    thinking_layer_set = None
    answer_layer_set = None
    dual_gradient_mode = False

    if args.thinking_layers is not None or args.answer_layers is not None:
        # Dual-gradient mode (CGS)
        if args.thinking_layers is None or args.answer_layers is None:
            raise ValueError("Both --thinking-layers and --answer-layers must be specified for dual-gradient mode")

        thinking_layer_set = parse_layer_spec(args.thinking_layers, num_layers)
        answer_layer_set = parse_layer_spec(args.answer_layers, num_layers)
        dual_gradient_mode = True

        # Train layers is the union of thinking and answer layers
        train_layer_set = thinking_layer_set | answer_layer_set

        if rank == 0:
            tqdm.write(f"\n{'='*60}")
            tqdm.write("Dual-Gradient Mode (CGS) Enabled")
            tqdm.write(f"{'='*60}")
            tqdm.write(f"  Thinking layers ({len(thinking_layer_set)}): {sorted(thinking_layer_set)[:10]}{'...' if len(thinking_layer_set) > 10 else ''}")
            tqdm.write(f"  Answer layers ({len(answer_layer_set)}): {sorted(answer_layer_set)[:10]}{'...' if len(answer_layer_set) > 10 else ''}")
            overlap = thinking_layer_set & answer_layer_set
            if overlap:
                tqdm.write(f"  Overlapping layers: {sorted(overlap)}")
            tqdm.write(f"  Thinking gradient weight: {args.thinking_gradient_weight}")
            tqdm.write(f"  Answer gradient weight: {args.answer_gradient_weight}")
            tqdm.write(f"{'='*60}\n")

    elif args.train_layers is not None:
        # Simple selective layer training
        train_layer_set = parse_layer_spec(args.train_layers, num_layers)
        if rank == 0:
            freeze_model_layers(model, train_layer_set, verbose=True)

    # SFT Anchor + Gradient Alignment Mode
    sft_anchor_layers = None
    if args.sft_anchor_enabled:
        if args.sft_anchor_layers is not None:
            sft_anchor_layers = parse_layer_spec(args.sft_anchor_layers, num_layers)

        if rank == 0:
            tqdm.write(f"\n{'='*60}")
            tqdm.write("SFT Anchor + Gradient Alignment Enabled")
            tqdm.write(f"{'='*60}")
            tqdm.write(f"  SFT anchor layers: {sorted(sft_anchor_layers) if sft_anchor_layers else 'all'}")
            tqdm.write(f"  SFT LR multiplier: {args.sft_anchor_lr_multiplier}")
            tqdm.write(f"  Gradient alignment mode: {args.gradient_alignment_mode}")
            if args.gradient_alignment_mode != 'none':
                tqdm.write(f"  Alignment weight: {args.gradient_alignment_weight}")
            tqdm.write(f"{'='*60}\n")

    # Enforce Thinking Mode
    if args.enforce_thinking and rank == 0:
        tqdm.write(f"\n{'='*60}")
        tqdm.write("Two-Phase Generation (Enforce Thinking) Enabled")
        tqdm.write(f"{'='*60}")
        tqdm.write(f"  Think markers: {args.think_start_token} ... {args.think_end_token}")
        tqdm.write(f"  Continuation tokens: {args.continuation_tokens}")
        force_ratio = args.continuation_force_answer_ratio if args.continuation_force_answer_ratio is not None else 0.8
        tqdm.write(f"  Force answer ratio: {force_ratio:.1%}")
        tqdm.write(f"{'='*60}\n")

    # Curriculum Thinking Scaffolding
    if args.curriculum_enabled and rank == 0:
        tqdm.write(f"\n{'='*60}")
        tqdm.write("Curriculum Thinking Scaffolding Enabled")
        tqdm.write(f"{'='*60}")
        tqdm.write(f"  Start ratio: {args.curriculum_start_ratio:.0%} thinking (warmup: {args.curriculum_warmup_iters} iters)")
        tqdm.write(f"  End ratio: {args.curriculum_end_ratio:.0%} thinking (after taper: {args.curriculum_taper_iters} iters)")
        tqdm.write(f"{'='*60}\n")

    # Multi-Curriculum Rollout
    if args.multi_curriculum_rollout and rank == 0:
        tqdm.write(f"\n{'='*60}")
        tqdm.write("Multi-Curriculum Rollout Enabled")
        tqdm.write(f"{'='*60}")
        if args.curriculum_scaffold_levels:
            levels = [float(x.strip()) for x in args.curriculum_scaffold_levels.split(',')]
        else:
            levels = [1.0 - k/(args.group_size-1) for k in range(args.group_size)] if args.group_size > 1 else [0.0]
        tqdm.write(f"  Scaffold levels per group: {[f'{l:.0%}' for l in levels]}")
        tqdm.write(f"  Group size: {args.group_size}")
        tqdm.write(f"{'='*60}\n")

    # Track current iteration for logging
    current_iteration = [0]
    last_val_metric = [None]
    last_val_loss = None

    def step(batch, prev_grad, do_update):
        """Single training step."""
        mx.clear_cache()
        if grad_accum_steps > 1 and prev_grad is not None:
            _gc.collect()

        prompt_tokens, answer_tokens, prompt_text, answer_text, type_info = batch

        # Build Curriculum Prefixes (if enabled, but NOT if multi_curriculum_rollout)
        curriculum_prefixes = None
        curriculum_ratio = None
        if args.curriculum_enabled and not args.multi_curriculum_rollout:
            curriculum_ratio = compute_curriculum_ratio(
                iteration=current_iteration[0],
                start_ratio=args.curriculum_start_ratio,
                end_ratio=args.curriculum_end_ratio,
                warmup_iters=args.curriculum_warmup_iters,
                taper_iters=args.curriculum_taper_iters,
            )

            curriculum_prefixes = []
            for target in answer_text:
                prefix = build_curriculum_prefix(
                    target_completion=target,
                    ratio=curriculum_ratio,
                    think_start=args.think_start_token,
                    think_end=args.think_end_token,
                    by_lines=args.curriculum_by_lines,
                    truncation_mode=args.curriculum_truncation_mode,
                    preserve_intuition=args.curriculum_preserve_intuition,
                )
                curriculum_prefixes.append(prefix)

        # Parse scaffold levels for multi-curriculum rollout
        scaffold_levels = None
        if args.multi_curriculum_rollout and args.curriculum_scaffold_levels:
            scaffold_levels = [float(x.strip()) for x in args.curriculum_scaffold_levels.split(',')]

        # Generate completions
        all_completions, all_completion_texts, batch_indices, two_phase_flags, all_scaffold_ratios, all_scaffold_token_counts = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_completion_length,
            group_size=args.group_size,
            temperature=args.temperature,
            batch_size=args.batch_size,
            end_token=end_answer_token,
            enforce_thinking=args.enforce_thinking,
            think_start=args.think_start_token,
            think_end=args.think_end_token,
            answer_end=end_answer_token,
            continuation_tokens=args.continuation_tokens,
            continuation_force_answer_ratio=args.continuation_force_answer_ratio if args.continuation_force_answer_ratio is not None else 0.5,
            curriculum_prefixes=curriculum_prefixes,
            target_completions=answer_text,
            multi_curriculum_rollout=args.multi_curriculum_rollout,
            curriculum_scaffold_levels=scaffold_levels,
            curriculum_truncation_mode=args.curriculum_truncation_mode,
            curriculum_preserve_intuition=args.curriculum_preserve_intuition,
            type_info=type_info,
            cross_sample_max_tokens=args.cross_sample_max_completion_length,
            smart_truncation_enabled=args.smart_truncation_enabled if args.smart_truncation_enabled is not None else False,
            max_extreme_tokens=args.max_extreme_tokens if args.max_extreme_tokens is not None else 1024,
            truncation_brevity_marker=args.truncation_brevity_marker if args.truncation_brevity_marker is not None else "[truncated due to brevity]",
            truncation_keep_start_ratio=args.truncation_keep_start_ratio if args.truncation_keep_start_ratio is not None else 0.3,
            truncation_keep_end_ratio=args.truncation_keep_end_ratio if args.truncation_keep_end_ratio is not None else 0.5,
            generation_sub_batch_size=args.generation_sub_batch_size if hasattr(args, 'generation_sub_batch_size') and args.generation_sub_batch_size is not None else 1,
        )

        _safe_eval(all_completions, checkpoint="generation_complete")
        _safe_clear("post_generation")

        # Log warnings for suspicious completions (but don't fail training)
        log_completion_warnings(all_completion_texts, current_iteration[0])

        # Prepare expanded data for reward calculation
        expanded_answers = []
        expanded_prompts = []
        expanded_types = []
        unique_prompt_indices = sorted(set(batch_indices))
        grouped_completions = {idx: [] for idx in unique_prompt_indices}

        for i, completion_idx in enumerate(batch_indices):
            grouped_completions[completion_idx].append(i)

        ordered_completions = []
        ordered_completion_texts = []
        ordered_batch_indices = []
        ordered_two_phase_flags = []
        ordered_scaffold_levels = []
        ordered_scaffold_token_counts = []

        for prompt_idx in unique_prompt_indices:
            completion_indices = grouped_completions[prompt_idx]
            for idx in completion_indices:
                ordered_completions.append(all_completions[idx])
                ordered_completion_texts.append(all_completion_texts[idx])
                ordered_batch_indices.append(prompt_idx)
                ordered_two_phase_flags.append(two_phase_flags[idx])
                ordered_scaffold_levels.append(all_scaffold_ratios[idx] if idx < len(all_scaffold_ratios) else 0.0)
                ordered_scaffold_token_counts.append(all_scaffold_token_counts[idx] if idx < len(all_scaffold_token_counts) else 0)
                expanded_answers.append(answer_text[prompt_idx])
                expanded_prompts.append(prompt_text[prompt_idx])
                expanded_types.append(
                    type_info[prompt_idx] if type_info is not None else None
                )

        del all_completions, all_completion_texts, batch_indices, grouped_completions, two_phase_flags, all_scaffold_ratios, all_scaffold_token_counts
        mx.clear_cache()

        # Calculate rewards and advantages
        advantages, reward_metrics, per_completion_rewards = calculate_rewards_and_advantages(
            reward_funcs=reward_funcs,
            expanded_prompts=expanded_prompts,
            all_completion_texts=ordered_completion_texts,
            expanded_answers=expanded_answers,
            expanded_types=expanded_types,
            batch_indices=ordered_batch_indices,
            unique_prompt_indices=unique_prompt_indices,
            reward_weights=args.reward_weights if hasattr(args, 'reward_weights') else None,
            scaffold_ratios=ordered_scaffold_levels if args.multi_curriculum_rollout else None,
            scaffold_penalty_weight=args.scaffold_penalty_weight,
            scaffold_penalty_mode=args.scaffold_penalty_mode,
            exam_compute_reward=exam_compute_reward,
        )

        mx.clear_cache()

        # SFT Anchor Step (if enabled)
        sft_grad = None
        sft_grad_flat = None
        sft_metrics = {}
        if args.sft_anchor_enabled:
            anchor_idx = current_iteration[0] % len(prompt_tokens)
            anchor_prompt = [prompt_tokens[anchor_idx]]
            anchor_target = [answer_tokens[anchor_idx]]

            def sft_loss_fn(model):
                loss, ntoks = compute_sft_anchor_loss(
                    model, anchor_prompt, anchor_target, tokenizer
                )
                return loss

            sft_value_and_grad = nn.value_and_grad(model, sft_loss_fn)
            sft_loss, sft_grad = sft_value_and_grad(model)
            mx.eval(sft_loss, sft_grad)

            sft_grad_flat = dict(tree_flatten(sft_grad))

            if sft_anchor_layers is not None:
                sft_grad_flat_masked = create_layer_gradient_mask(sft_grad_flat, sft_anchor_layers)
                sft_grad = tree_unflatten(list(sft_grad_flat_masked.items()))
                sft_grad_flat = sft_grad_flat_masked

            sft_metrics['sft_anchor_loss'] = float(sft_loss)

            lr_mult = args.sft_anchor_lr_multiplier
            sft_grad_flat = {k: v * lr_mult for k, v in sft_grad_flat.items()}
            sft_metrics['sft_lr_multiplier'] = lr_mult
            _safe_eval(*sft_grad_flat.values(), checkpoint="sft_grad")
            _safe_clear("sft_anchor")

        # Compute Loss and Gradients
        if dual_gradient_mode:
            # Dual-gradient mode (CGS)
            thinking_masks, answer_masks = detect_thinking_answer_positions(
                ordered_completion_texts,
                tokenizer,
                think_start=args.think_start_token,
                think_end=args.think_end_token,
            )
            _safe_eval(*thinking_masks, *answer_masks, checkpoint="thinking_masks")

            max_mask_len = max(m.shape[0] for m in thinking_masks) if thinking_masks else 1
            padded_thinking = []
            padded_answer = []
            for tm, am in zip(thinking_masks, answer_masks):
                pad_len = max_mask_len - tm.shape[0]
                if pad_len > 0:
                    padded_thinking.append(mx.concatenate([tm, mx.zeros((pad_len,))]))
                    padded_answer.append(mx.concatenate([am, mx.zeros((pad_len,))]))
                else:
                    padded_thinking.append(tm)
                    padded_answer.append(am)
            thinking_mask_batch = mx.stack(padded_thinking)
            answer_mask_batch = mx.stack(padded_answer)
            _safe_eval(thinking_mask_batch, answer_mask_batch, checkpoint="mask_batch")

            combined_weight_mask = (
                args.thinking_gradient_weight * thinking_mask_batch +
                args.answer_gradient_weight * answer_mask_batch
            )
            total_weight = args.thinking_gradient_weight + args.answer_gradient_weight
            combined_weight_mask = combined_weight_mask / (total_weight / 2.0)

            (lvalue, toks, metrics), grad = loss_value_and_grad(
                model,
                batch=(prompt_tokens, answer_tokens, prompt_text, answer_text, type_info),
                completions=ordered_completions,
                completion_texts=ordered_completion_texts,
                batch_indices=ordered_batch_indices,
                advantages=advantages,
                reward_metrics=reward_metrics,
                beta=args.beta,
                epsilon=args.epsilon,
                epsilon_high=args.epsilon_high,
                ref_model=ref_model,
                grpo_loss_type=args.grpo_loss_type,
                importance_sampling_level=args.importance_sampling_level,
                max_tokens=args.max_completion_length,
                token_type_mask=combined_weight_mask,
                scaffold_token_counts=ordered_scaffold_token_counts,
            )

            _safe_eval(lvalue, grad, checkpoint="dual_loss_grad")

            grad_flat = dict(tree_flatten(grad))
            del grad
            mx.clear_cache()

            layer_pattern = re.compile(r'model\.layers\.(\d+)\.')
            masked_grad = {}

            for key, g in grad_flat.items():
                match = layer_pattern.search(key)
                if match:
                    layer_idx = int(match.group(1))
                    in_thinking = layer_idx in thinking_layer_set
                    in_answer = layer_idx in answer_layer_set

                    if in_thinking or in_answer:
                        masked_grad[key] = g
                    else:
                        masked_grad[key] = mx.zeros_like(g)
                else:
                    masked_grad[key] = g

            del grad_flat

            grad = tree_unflatten(list(masked_grad.items()))
            del masked_grad
            mx.clear_cache()

            thinking_ratio = float(thinking_mask_batch.sum() / max(thinking_mask_batch.size, 1))
            answer_ratio = float(answer_mask_batch.sum() / max(answer_mask_batch.size, 1))
            metrics['thinking_token_ratio'] = thinking_ratio
            metrics['answer_token_ratio'] = answer_ratio

        else:
            # Standard mode
            (lvalue, toks, metrics), grad = loss_value_and_grad(
                model,
                batch=(prompt_tokens, answer_tokens, prompt_text, answer_text, type_info),
                completions=ordered_completions,
                completion_texts=ordered_completion_texts,
                batch_indices=ordered_batch_indices,
                advantages=advantages,
                reward_metrics=reward_metrics,
                beta=args.beta,
                epsilon=args.epsilon,
                epsilon_high=args.epsilon_high,
                ref_model=ref_model,
                grpo_loss_type=args.grpo_loss_type,
                importance_sampling_level=args.importance_sampling_level,
                max_tokens=args.max_completion_length,
                scaffold_token_counts=ordered_scaffold_token_counts,
            )
            _safe_eval(lvalue, grad, checkpoint="standard_loss_grad")

            if train_layer_set is not None and not dual_gradient_mode:
                grad_flat = dict(tree_flatten(grad))
                masked_grad_flat = create_layer_gradient_mask(grad_flat, train_layer_set)
                grad = tree_unflatten(list(masked_grad_flat.items()))

        # Gradient Alignment with SFT (if enabled)
        if args.sft_anchor_enabled and sft_grad_flat is not None:
            grpo_grad_flat = dict(tree_flatten(grad))

            alignment_metrics = compute_gradient_alignment(sft_grad_flat, grpo_grad_flat)
            metrics['grad_cosine_similarity'] = alignment_metrics['cosine_similarity']
            metrics['grad_kl_divergence'] = alignment_metrics['kl_divergence']
            metrics['sft_grad_norm'] = alignment_metrics['sft_norm']
            metrics['grpo_grad_norm'] = alignment_metrics['grpo_norm']

            if args.gradient_alignment_mode == 'interpolate':
                alpha = args.gradient_alignment_weight
                blended_grad_flat = interpolate_gradients(sft_grad_flat, grpo_grad_flat, alpha)
                grad = tree_unflatten(list(blended_grad_flat.items()))
                metrics['grad_blend_alpha'] = alpha

            elif args.gradient_alignment_mode == 'project':
                strength = args.gradient_alignment_weight
                projected_grad_flat = project_gradient_toward_sft(grpo_grad_flat, sft_grad_flat, strength)
                grad = tree_unflatten(list(projected_grad_flat.items()))
                metrics['grad_project_strength'] = strength

            elif args.gradient_alignment_mode == 'kl':
                kl = max(alignment_metrics['kl_divergence'], 0.0)
                scale_factor = 1.0 / (1.0 + args.gradient_alignment_weight * kl)
                grad = tree_map(lambda g: g * scale_factor, grad)
                metrics['grad_kl_scale'] = scale_factor

            elif args.gradient_alignment_mode == 'cosine':
                cos_sim = alignment_metrics['cosine_similarity']
                scale_factor = max(0.1, (1.0 + cos_sim) / 2.0)
                scale_factor = (1 - args.gradient_alignment_weight) + args.gradient_alignment_weight * scale_factor
                grad = tree_map(lambda g: g * scale_factor, grad)
                metrics['grad_cosine_scale'] = scale_factor

            del sft_grad_flat, grpo_grad_flat
            mx.clear_cache()

        metrics.update(sft_metrics)

        if curriculum_ratio is not None:
            metrics['curriculum_ratio'] = curriculum_ratio
            if curriculum_prefixes:
                avg_prefix_len = sum(len(p) for p in curriculum_prefixes) / len(curriculum_prefixes)
                metrics['curriculum_prefix_len'] = avg_prefix_len

        if args.multi_curriculum_rollout:
            metrics['multi_curriculum_enabled'] = 1.0
            if scaffold_levels:
                metrics['scaffold_levels_min'] = min(scaffold_levels)
                metrics['scaffold_levels_max'] = max(scaffold_levels)

        # Log rollouts
        if rollout_logger and current_iteration[0] % args.log_rollouts_every_n_steps == 0:
            rewards_per_func = {k: v for k, v in per_completion_rewards.items() if k not in ("total", "exam_details")}
            total_rewards = per_completion_rewards.get("total", [0.0] * len(ordered_completion_texts))
            exam_details_list = per_completion_rewards.get("exam_details", None)

            advantages_list = []
            for adv in advantages:
                if hasattr(adv, 'item'):
                    advantages_list.append(float(adv.item()))
                elif hasattr(adv, 'tolist'):
                    advantages_list.append(float(adv.tolist()))
                else:
                    advantages_list.append(float(adv) if adv is not None else 0.0)

            prompt_token_counts = [
                len(prompt_tokens[idx]) if idx < len(prompt_tokens) else 0
                for idx in ordered_batch_indices
            ]
            completion_token_counts = [
                len(c.split()) if c else 0 for c in ordered_completion_texts
            ]

            rollout_logger.log_rollout(
                iteration=current_iteration[0],
                update=current_iteration[0],
                prompts=expanded_prompts,
                prompt_texts=expanded_prompts,
                completions=ordered_completion_texts,
                answers=expanded_answers,
                rewards_per_func=rewards_per_func,
                total_rewards=total_rewards,
                advantages=advantages_list,
                prompt_tokens=prompt_token_counts,
                completion_tokens=completion_token_counts,
                batch_indices=ordered_batch_indices,
                type_info=expanded_types,
                group_size=args.group_size,
                two_phase_recovered=ordered_two_phase_flags,
                scaffold_levels=ordered_scaffold_levels,
                reward_details=exam_details_list,
            )

        # Add scaffold masking metrics
        total_scaffold_tokens = sum(ordered_scaffold_token_counts)
        if total_scaffold_tokens > 0:
            metrics['scaffold_tokens_masked'] = total_scaffold_tokens
            metrics['scaffold_tokens_per_completion'] = total_scaffold_tokens / len(ordered_scaffold_token_counts) if ordered_scaffold_token_counts else 0

        del ordered_completions, ordered_completion_texts, ordered_batch_indices, ordered_two_phase_flags, ordered_scaffold_levels, ordered_scaffold_token_counts
        del advantages, reward_metrics
        mx.clear_cache()

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
            _safe_eval(grad, checkpoint="grad_accum")

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            _safe_eval(model.parameters(), optimizer.state, checkpoint="optimizer_update")
            grad = None
            _safe_clear("post_optimizer")

        return lvalue, toks, metrics, grad

    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    # Resume from Checkpoint
    start_iteration = 1
    trained_tokens = 0
    wandb_run_id = None

    if args.resume_from_checkpoint or args.resume_iteration is not None:
        adapter_dir = Path(args.adapter_file).parent
        optimizer_state_file = adapter_dir / "optimizer_state.safetensors"
        training_state_file = adapter_dir / "training_state.json"

        if training_state_file.exists():
            try:
                with open(training_state_file, 'r') as f:
                    training_state = json.load(f)
                wandb_run_id = training_state.get('wandb_run_id', None)
                saved_tokens = training_state.get('trained_tokens', 0)
                saved_lr = training_state.get('learning_rate', None)
                saved_iteration = training_state.get('iteration', 0)

                if wandb_run_id:
                    tqdm.write(f"Found WandB run ID: {wandb_run_id}")
            except Exception as e:
                tqdm.write(f"Warning: Failed to load training state: {e}")
                training_state = {}
                saved_tokens = 0
                saved_iteration = 0
        else:
            training_state = {}
            saved_tokens = 0
            saved_iteration = 0

        if optimizer_state_file.exists():
            try:
                opt_state = mx.load(str(optimizer_state_file))
                current_state = optimizer.state
                if current_state:
                    try:
                        optimizer.state = tree_unflatten(list(opt_state.items()))
                        tqdm.write(f"Loaded optimizer state from {optimizer_state_file}")
                    except Exception as unflatten_err:
                        tqdm.write(f"  Warning: Could not unflatten optimizer state: {unflatten_err}")
            except Exception as e:
                tqdm.write(f"Warning: Failed to load optimizer state: {e}")

        if args.resume_iteration is not None:
            start_iteration = args.resume_iteration + 1
            trained_tokens = saved_tokens
            tqdm.write(f"Resuming from iteration {start_iteration} (manual override)")
        elif saved_iteration > 0:
            start_iteration = saved_iteration + 1
            trained_tokens = saved_tokens
            tqdm.write(f"Resuming from iteration {start_iteration}")

        if start_iteration > args.iters:
            tqdm.write(f"Warning: Start iteration {start_iteration} > total iters {args.iters}")
            args.iters = start_iteration + args.iters - 1

    # Initialize Rollout Logger
    if args.log_rollouts and rank == 0:
        output_dir = Path(args.adapter_file).parent if args.adapter_file else Path("./outputs")
        run_name = Path(args.adapter_file).stem if args.adapter_file else None
        rollout_logger = RolloutLogger(
            RolloutLoggerConfig(
                enabled=True,
                log_every_n_steps=args.log_rollouts_every_n_steps,
                log_to_wandb=args.log_rollouts_to_wandb,
                wandb_run_id=wandb_run_id if start_iteration > 1 else None,
                resume_from_iteration=start_iteration - 1 if start_iteration > 1 else 0,
            ),
            adapter_file=str(args.adapter_file) if args.adapter_file else None,
        )
        if start_iteration > 1:
            tqdm.write(f"Rollout logging enabled (resuming): {output_dir}")
        else:
            tqdm.write(f"Rollout logging enabled: {output_dir}")

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    accumulated_metrics = {
        "total_rewards_mean": 0,
        "total_rewards_std": 0,
        "grouped_rewards_mean": 0,
        "grouped_rewards_std": 0,
        "kl": 0,
        "average_generated_tokens": 0,
        "max_generated_tokens": 0,
        "min_generated_tokens": 0,
        "hit_max_tokens_ratio": 0,
        "clip_ratio_low": 0,
        "clip_ratio_high": 0,
        "clip_ratio_total": 0,
    }
    grad_accum = None
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"{func_name}_mean"] = 0
        accumulated_metrics[f"{func_name}_std"] = 0
        accumulated_metrics[f"{func_name}_coverage"] = 0

    start = time.perf_counter()
    pbar = tqdm(range(start_iteration, args.iters + 1), desc="Training", disable=rank != 0)
    if start_iteration > 1:
        tqdm.write(f"Starting training loop from iteration {start_iteration} to {args.iters}")

    last_checkpoint_iter = 0
    consecutive_failures = 0
    max_consecutive_failures = 3

    cached_batch = None
    batch_repeat_count = 0
    samples_per_scaffold = getattr(args, 'samples_per_scaffold', 1) or 1

    try:
        for it in pbar:
            current_iteration[0] = it
            mx.clear_cache()

            # Sample repetition
            if cached_batch is None or batch_repeat_count >= samples_per_scaffold:
                batch = next(
                    iterate_batches(
                        dataset=train_dataset,
                        batch_size=args.batch_size,
                        max_seq_length=args.max_seq_length,
                        train=True,
                    )
                )
                cached_batch = batch
                batch_repeat_count = 1
            else:
                batch = cached_batch
                batch_repeat_count += 1

            # Wrap step in try/except for Metal GPU crash recovery
            try:
                _iter_start(it)
                lvalue, toks, metrics, grad_accum = step(
                    batch,
                    grad_accum,
                    it % grad_accum_steps == 0,
                )
                consecutive_failures = 0
            except RuntimeError as e:
                error_msg = str(e)
                if "Command buffer execution failed" in error_msg or "METAL" in error_msg:
                    consecutive_failures += 1
                    tqdm.write(f"\nWarning: Metal GPU crash at iter {it}: {error_msg[:100]}...")
                    tqdm.write(f"   Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")

                    if rank == 0:
                        try:
                            emergency_path = Path(args.adapter_file).parent / f"emergency_{it:07d}_adapters.safetensors"
                            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                            # Validate emergency checkpoint (warn but save anyway)
                            is_valid, corrupt_keys = validate_adapter_weights(adapter_weights, raise_on_error=False)
                            if not is_valid:
                                tqdm.write(f"   WARNING: Emergency checkpoint may be corrupted: {corrupt_keys[:3]}")
                            mx.save_safetensors(str(emergency_path), adapter_weights)
                            tqdm.write(f"   Saved emergency checkpoint: {emergency_path}")
                        except Exception as save_error:
                            tqdm.write(f"   Failed to save emergency checkpoint: {save_error}")

                    mx.clear_cache()
                    _gc.collect()

                    if consecutive_failures >= max_consecutive_failures:
                        tqdm.write(f"\nError: Too many consecutive failures. Stopping training.")
                        break

                    tqdm.write(f"   Skipping iteration {it}, continuing...")
                    continue
                else:
                    raise

            losses += lvalue
            n_tokens += toks
            steps += 1

            for k, v in metrics.items():
                if k in accumulated_metrics:
                    accumulated_metrics[k] += v
                else:
                    accumulated_metrics[k] = v

            _safe_eval(state, losses, n_tokens, checkpoint="step_state")
            if grad_accum is not None:
                _safe_eval(grad_accum, checkpoint="grad_accum_state")
            _iter_end(it)

            if grad_accum_steps > 1:
                mx.clear_cache()
                _gc.collect()

            # Report metrics
            if it % args.steps_per_report == 0 or it == args.iters:
                stop = time.perf_counter()

                train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
                avg_metrics = {
                    k: v / (steps * world_size) for k, v in accumulated_metrics.items()
                }
                n_tokens = mx.distributed.all_sum(n_tokens).item()
                learning_rate = optimizer.learning_rate.item()
                it_sec = args.steps_per_report / (stop - start)
                tokens_sec = float(n_tokens) / (stop - start)
                trained_tokens += n_tokens
                peak_mem = mx.get_peak_memory() / 1e9

                if rank == 0:
                    avg_metrics = {}
                    for k, v in accumulated_metrics.items():
                        accumulated_v = v / (steps * world_size)
                        if isinstance(accumulated_v, mx.array):
                            avg_metrics[k] = float(accumulated_v.item())
                        else:
                            avg_metrics[k] = float(accumulated_v)

                    pbar.set_postfix({"loss": f"{train_loss:.3f}", "it/s": f"{it_sec:.3f}"})

                    reward_metrics_str = ""
                    for reward_func in reward_funcs:
                        func_name = reward_func.__name__
                        mean_key = f"{func_name}_mean"
                        std_key = f"{func_name}_std"
                        cov_key = f"{func_name}_coverage"

                        if mean_key in avg_metrics:
                            display_name = func_name.replace("_reward_func", "").replace("r1_", "")
                            reward_metrics_str += (
                                f"  - {display_name}: "
                                f"mean={avg_metrics[mean_key]:.3f}, "
                                f"std={avg_metrics[std_key]:.3f}, "
                                f"cov={avg_metrics[cov_key]:.2%}\n"
                            )

                    loss_str = f"Loss: {train_loss:.3f}"
                    if 'thinking_token_ratio' in avg_metrics:
                        loss_str += f" [CGS: think={avg_metrics['thinking_token_ratio']:.1%}, answer={avg_metrics['answer_token_ratio']:.1%}]"

                    tqdm.write(
                        f"\n{'='*80}\n"
                        f"Iter {it}:\n"
                        f"{'-'*80}\n"
                        f"{loss_str}\n"
                        f"Total Rewards:  mean={avg_metrics['total_rewards_mean']:.3f}, "
                        f"std={avg_metrics['total_rewards_std']:.3f}\n"
                        f"Group Rewards:  mean={avg_metrics['grouped_rewards_mean']:.3f}, "
                        f"std={avg_metrics['grouped_rewards_std']:.3f}\n"
                        f"KL Divergence: {avg_metrics['kl']:.12f}\n"
                        f"{'-'*80}\n"
                        f"Generation Stats:\n"
                        f"  - Avg tokens: {avg_metrics['average_generated_tokens']:.1f}\n"
                        f"  - Min tokens: {avg_metrics['min_generated_tokens']:.0f}\n"
                        f"  - Max tokens: {avg_metrics['max_generated_tokens']:.0f} "
                        f"(limit: {args.max_completion_length})\n"
                        f"  - Hit limit: {avg_metrics['hit_max_tokens_ratio']:.1%}\n"
                        f"{'-'*80}\n"
                        f"Individual Reward Functions:\n"
                        f"{reward_metrics_str}"
                        f"{'-'*80}\n"
                        f"Clipping:  low={avg_metrics['clip_ratio_low']:.3f}, "
                        f"high={avg_metrics['clip_ratio_high']:.3f}, "
                        f"total={avg_metrics['clip_ratio_total']:.3f}\n"
                        f"Learning Rate: {learning_rate:.4e}\n"
                        f"Speed: {it_sec:.3f} it/s, {tokens_sec:.1f} tok/s\n"
                        f"Memory: {peak_mem:.3f}GB\n"
                        f"{'='*80}\n"
                    )

                    if rollout_logger:
                        rollout_logger.log_iteration(
                            iteration=it,
                            update=it,
                            loss=train_loss,
                            learning_rate=learning_rate,
                            metrics=avg_metrics,
                            reward_funcs=reward_funcs,
                            tokens_per_sec=tokens_sec,
                            iterations_per_sec=it_sec,
                            memory_gb=peak_mem,
                        )

                    # Training Monitor Check
                    if training_monitor and it % grad_accum_steps == 0:
                        actual_update = it // grad_accum_steps
                        should_stop = training_monitor.log_step(
                            step=actual_update,
                            metrics=avg_metrics,
                            loss=train_loss,
                        )
                        if should_stop:
                            tqdm.write(training_monitor.get_stop_reason())
                            tqdm.write(training_monitor.get_summary())
                            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                            emergency_path = Path(args.adapter_file).parent / f"early_stop_{it:07d}_adapters.safetensors"
                            mx.save_safetensors(str(emergency_path), adapter_weights)
                            tqdm.write(f"Saved early stop checkpoint: {emergency_path}")

                            try:
                                adapter_dir = Path(args.adapter_file).parent
                                optimizer_state_file = adapter_dir / "optimizer_state.safetensors"
                                opt_state = dict(tree_flatten(optimizer.state))
                                saveable_state = {k: v for k, v in opt_state.items() if isinstance(v, mx.array)}
                                if saveable_state:
                                    mx.save_safetensors(str(optimizer_state_file), saveable_state)
                                meta_file = adapter_dir / "training_state.json"
                                state_dict = {
                                    'iteration': it,
                                    'trained_tokens': trained_tokens,
                                    'learning_rate': float(learning_rate),
                                    'early_stopped': True,
                                }
                                if rollout_logger:
                                    wb_id = rollout_logger.get_wandb_run_id()
                                    if wb_id:
                                        state_dict['wandb_run_id'] = wb_id
                                with open(meta_file, 'w') as f:
                                    json.dump(state_dict, f)
                            except Exception as save_err:
                                tqdm.write(f"  Warning: Could not save optimizer state: {save_err}")
                            break

                if training_callback is not None:
                    train_info = {
                        "iteration": it,
                        "train_loss": train_loss,
                        **{f"train_{k}": v for k, v in avg_metrics.items()},
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_sec,
                        "tokens_per_second": tokens_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory": peak_mem,
                    }
                    training_callback.on_train_loss_report(train_info)

                losses = 0
                n_tokens = 0
                steps = 0
                accumulated_metrics = {k: 0 for k in accumulated_metrics}
                start = time.perf_counter()

            # Save checkpoint
            if it % args.steps_per_save == 0:
                adapter_weights = dict(tree_flatten(model.trainable_parameters()))

                # Validate weights before saving - exits with code 42 if corrupted
                validate_and_exit_on_corruption(adapter_weights, f"iter_{it}")

                mx.save_safetensors(str(args.adapter_file), adapter_weights)
                checkpoint = (
                    Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
                )
                mx.save_safetensors(str(checkpoint), adapter_weights)

                optimizer_state_file = Path(args.adapter_file).parent / "optimizer_state.safetensors"
                try:
                    opt_state = dict(tree_flatten(optimizer.state))
                    saveable_state = {}
                    for k, v in opt_state.items():
                        if isinstance(v, mx.array):
                            saveable_state[k] = v
                    if saveable_state:
                        mx.save_safetensors(str(optimizer_state_file), saveable_state)
                        meta_file = Path(args.adapter_file).parent / "training_state.json"
                        state_dict = {
                            'iteration': it,
                            'trained_tokens': trained_tokens,
                            'learning_rate': float(learning_rate),
                        }
                        if rollout_logger:
                            wb_id = rollout_logger.get_wandb_run_id()
                            if wb_id:
                                state_dict['wandb_run_id'] = wb_id
                        with open(meta_file, 'w') as f:
                            json.dump(state_dict, f)
                except Exception as opt_save_err:
                    tqdm.write(f"  Warning: Failed to save optimizer state: {opt_save_err}")

                tqdm.write(
                    f"\n"
                    f"Iter {it}: Saved adapter weights to "
                    f"{args.adapter_file} and {checkpoint}."
                    f"\n         Optimizer state saved to {optimizer_state_file}"
                )

                if checkpoint_manager:
                    checkpoint_manager.register_checkpoint(
                        iteration=it,
                        path=checkpoint,
                        metric_value=last_val_loss,
                    )

        # Save final weights
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))

        # Validate final weights before saving - exits with code 42 if corrupted
        validate_and_exit_on_corruption(adapter_weights, "final_weights")

        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        tqdm.write(f"Saved final weights to {args.adapter_file}.")

    except KeyboardInterrupt:
        tqdm.write("\nWarning: Training interrupted by user.")
        if rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            # Validate interrupt checkpoint (warn but save anyway)
            is_valid, corrupt_keys = validate_adapter_weights(adapter_weights, raise_on_error=False)
            if not is_valid:
                tqdm.write(f"  WARNING: Interrupt checkpoint may be corrupted: {corrupt_keys[:3]}")
            interrupt_path = Path(args.adapter_file).parent / f"interrupted_{current_iteration[0]:07d}_adapters.safetensors"
            mx.save_safetensors(str(interrupt_path), adapter_weights)
            tqdm.write(f"Saved interrupt checkpoint: {interrupt_path}")

            try:
                adapter_dir = Path(args.adapter_file).parent
                optimizer_state_file = adapter_dir / "optimizer_state.safetensors"
                opt_state = dict(tree_flatten(optimizer.state))
                saveable_state = {k: v for k, v in opt_state.items() if isinstance(v, mx.array)}
                if saveable_state:
                    mx.save_safetensors(str(optimizer_state_file), saveable_state)
                meta_file = adapter_dir / "training_state.json"
                state_dict = {
                    'iteration': current_iteration[0],
                    'trained_tokens': trained_tokens,
                    'learning_rate': float(optimizer.learning_rate.item()) if hasattr(optimizer.learning_rate, 'item') else float(optimizer.learning_rate),
                    'interrupted': True,
                }
                if rollout_logger:
                    wb_id = rollout_logger.get_wandb_run_id()
                    if wb_id:
                        state_dict['wandb_run_id'] = wb_id
                with open(meta_file, 'w') as f:
                    json.dump(state_dict, f)
                tqdm.write(f"Saved optimizer state and progress for resume")
            except Exception as save_err:
                tqdm.write(f"  Warning: Could not save optimizer state: {save_err}")

    except Exception as e:
        tqdm.write(f"\nError: Training failed with error: {e}")
        if rank == 0:
            try:
                adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                # Validate crash checkpoint (warn but save anyway)
                is_valid, corrupt_keys = validate_adapter_weights(adapter_weights, raise_on_error=False)
                if not is_valid:
                    tqdm.write(f"  WARNING: Crash checkpoint may be corrupted: {corrupt_keys[:3]}")
                crash_path = Path(args.adapter_file).parent / f"crash_{current_iteration[0]:07d}_adapters.safetensors"
                mx.save_safetensors(str(crash_path), adapter_weights)
                tqdm.write(f"Saved crash checkpoint: {crash_path}")
            except:
                tqdm.write("Failed to save crash checkpoint")
        raise

    finally:
        if rollout_logger:
            rollout_logger.close()
            tqdm.write("Rollout logs saved.")
        mx.clear_cache()


# =============================================================================
# TRAINING WITH CRASH RECOVERY
# =============================================================================


def train_grpo_with_recovery(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    args: GRPOTrainingArgs = None,
    loss_fn: Callable = None,
    iterate_batches: Callable = None,
    training_callback: TrainingCallback = None,
    end_answer_token: str = "</answer>",
):
    """Wrapper for train_grpo with automatic crash recovery.

    On Metal GPU crashes (IOAF errors), this function will:
    1. Save emergency checkpoint
    2. Wait for cooldown
    3. Reload model from last checkpoint
    4. Resume training

    Args:
        Same as train_grpo
    """
    if args is None:
        args = GRPOTrainingArgs()
    if loss_fn is None:
        loss_fn = grpo_loss
    if iterate_batches is None:
        iterate_batches = iterate_grpo_batches
    if reward_funcs is None:
        reward_funcs = [
            r1_accuracy_reward_func,
            r1_int_reward_func,
            r1_strict_format_reward_func,
            r1_soft_format_reward_func,
            r1_count_xml,
        ]

    crash_count = 0
    max_retries = args.max_crash_retries if args.auto_resume_on_crash else 0
    cooldown = args.crash_cooldown_seconds

    while crash_count <= max_retries:
        try:
            train_grpo(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                reward_funcs=reward_funcs,
                args=args,
                loss_fn=loss_fn,
                iterate_batches=iterate_batches,
                training_callback=training_callback,
                end_answer_token=end_answer_token,
            )
            return

        except RuntimeError as e:
            error_msg = str(e).lower()
            is_metal_crash = any(x in error_msg for x in [
                "command buffer execution failed",
                "ioaf code",
                "metal",
                "gpu",
                "device reset",
                "memory allocation failed",
            ])

            if not is_metal_crash or not args.auto_resume_on_crash:
                raise

            crash_count += 1
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"CRASH RECOVERY ({crash_count}/{max_retries})")
            tqdm.write(f"   Error: {str(e)[:200]}...")
            tqdm.write(f"{'='*60}")

            if crash_count > max_retries:
                tqdm.write(f"\nError: Max crash retries ({max_retries}) exceeded. Stopping.")
                raise

            mx.clear_cache()
            _gc.collect()

            tqdm.write(f"   Waiting {cooldown}s before retry...")
            time.sleep(cooldown)

            if args.adapter_file:
                adapter_dir = Path(args.adapter_file).parent
                checkpoints = list(adapter_dir.glob("*_adapters.safetensors"))
                checkpoints = [c for c in checkpoints if "crash" not in c.name and "emergency" not in c.name]

                if checkpoints:
                    def get_iter(p):
                        try:
                            return int(p.stem.split('_')[0])
                        except:
                            return 0
                    checkpoints.sort(key=get_iter, reverse=True)
                    latest = checkpoints[0]

                    tqdm.write(f"   Loading checkpoint: {latest}")
                    try:
                        adapter_weights = mx.load(str(latest))
                        model.load_weights(list(adapter_weights.items()), strict=False)
                        mx.eval(model.parameters())

                        state_file = adapter_dir / "training_state.json"
                        if state_file.exists():
                            with open(state_file) as f:
                                state = json.load(f)
                                start_iter = state.get('iteration', 0)
                                tqdm.write(f"   Resuming from iteration {start_iter}")
                                args.resume_from_checkpoint = True
                    except Exception as load_err:
                        tqdm.write(f"   Warning: Could not load checkpoint: {load_err}")
                else:
                    tqdm.write("   No checkpoints found, restarting from beginning")

            tqdm.write(f"   Retrying training...")
            tqdm.write(f"{'='*60}\n")

    tqdm.write("Training completed after recovery.")
