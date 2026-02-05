"""Loss functions for GRPO training.

This module provides:
- GRPO loss computation with importance sampling
- Per-token log probability computation
- Reward calculation and advantage estimation
- SFT anchor loss for format grounding

SOLID Principles:
- Single Responsibility: Only handles loss computation
- Open/Closed: Different loss types via grpo_loss_type parameter
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .debug import safe_eval as _safe_eval, safe_clear as _safe_clear

if TYPE_CHECKING:
    from typing import Any, Callable

__all__ = [
    "grpo_loss",
    "get_per_token_logps",
    "get_per_token_logps_with_prompt_mask",
    "calculate_rewards_and_advantages",
    "compute_sft_anchor_loss",
]


def get_per_token_logps(
    model: nn.Module,
    inputs: mx.array,
    lengths: mx.array,
) -> list[mx.array]:
    """Compute per-token log probabilities for sequences.

    Args:
        model: The language model
        inputs: Padded input sequences [batch, max_seq_len]
        lengths: Length of each sequence

    Returns:
        List of log probability arrays, one per sample
    """
    logits = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]
    per_token_logps = []
    for i in range(logits.shape[0]):
        seq_len = int(lengths[i]) - 1
        seq_logits = logits[i, :seq_len]
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
    mx.eval(logits)
    return per_token_logps


def get_per_token_logps_with_prompt_mask(
    model: nn.Module,
    inputs: mx.array,
    total_lengths: mx.array,
    prompt_lengths: mx.array,
) -> list[mx.array]:
    """Compute log probs for completion tokens only, with full prompt context.

    Args:
        model: The language model
        inputs: Padded sequences [batch, max_seq_len] containing [prompt + completion]
        total_lengths: Length of each full sequence (prompt + completion)
        prompt_lengths: Length of prompt for each sequence

    Returns:
        List of log prob arrays, one per sample, containing ONLY completion token log probs
    """
    logits = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab]
    targets = inputs[:, 1:]      # [batch, seq_len-1]

    per_token_logps = []

    for i in range(logits.shape[0]):
        total_len = int(total_lengths[i])
        prompt_len = int(prompt_lengths[i])

        completion_start_logit_idx = prompt_len - 1 if prompt_len > 0 else 0
        completion_end_logit_idx = total_len - 1

        if completion_end_logit_idx <= completion_start_logit_idx:
            per_token_logps.append(mx.zeros((1,)))
            continue

        completion_logits = logits[i, completion_start_logit_idx:completion_end_logit_idx]
        completion_targets = targets[i, completion_start_logit_idx:completion_end_logit_idx]

        log_probs = nn.log_softmax(completion_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs, completion_targets.reshape(-1, 1), axis=-1
        ).squeeze(-1)

        per_token_logps.append(token_log_probs)

    mx.eval(logits)
    return per_token_logps


def compute_sft_anchor_loss(
    model: nn.Module,
    prompt_tokens: list[Any],
    target_tokens: list[Any],
    tokenizer: Any,
    layer_mask: set[int] | None = None,
) -> tuple[mx.array, int]:
    """Compute SFT loss on target completion for format anchoring.

    Args:
        model: The model
        prompt_tokens: List of prompt token sequences
        target_tokens: List of target completion token sequences
        tokenizer: Tokenizer for padding
        layer_mask: Optional set of layer indices for gradients

    Returns:
        (loss, ntoks) tuple
    """
    sequences = []
    prompt_lengths = []

    for prompt, target in zip(prompt_tokens, target_tokens):
        if isinstance(prompt, mx.array):
            prompt = prompt.tolist()
        if isinstance(target, mx.array):
            target = target.tolist()

        full_seq = prompt + target
        sequences.append(full_seq)
        prompt_lengths.append(len(prompt))

    max_len = max(len(s) for s in sequences)
    padded = []
    lengths = []
    for seq in sequences:
        lengths.append(len(seq))
        if len(seq) < max_len:
            padded.append(seq + [tokenizer.pad_token_id or 0] * (max_len - len(seq)))
        else:
            padded.append(seq)

    inputs = mx.array(padded)
    lengths_arr = mx.array(lengths)
    prompt_lengths_arr = mx.array(prompt_lengths)

    logits = model(inputs)

    total_loss = mx.array(0.0)
    total_tokens = 0

    for i in range(len(sequences)):
        seq_len = int(lengths_arr[i])
        prompt_len = int(prompt_lengths_arr[i])

        if seq_len <= prompt_len:
            continue

        target_logits = logits[i, prompt_len:seq_len - 1]
        target_ids = inputs[i, prompt_len + 1:seq_len]

        log_probs = nn.log_softmax(target_logits, axis=-1)
        token_losses = -mx.take_along_axis(
            log_probs,
            target_ids.reshape(-1, 1),
            axis=-1
        ).squeeze(-1)

        total_loss = total_loss + mx.sum(token_losses)
        total_tokens += len(token_losses)

    if total_tokens > 0:
        loss = total_loss / total_tokens
    else:
        loss = mx.array(0.0)

    return loss, total_tokens


def calculate_rewards_and_advantages(
    reward_funcs: list[Callable],
    expanded_prompts: list[str],
    all_completion_texts: list[str],
    expanded_answers: list[str],
    expanded_types: list[Any],
    batch_indices: list[int],
    unique_prompt_indices: list[int],
    reward_weights: list[float] | None = None,
    scaffold_ratios: list[float] | None = None,
    scaffold_penalty_weight: float = 0.0,
    scaffold_penalty_mode: str = "multiplicative",
    exam_compute_reward: Callable | None = None,
) -> tuple[mx.array, dict[str, Any], dict[str, list[float]]]:
    """Calculate rewards and advantages for completions.

    Args:
        reward_funcs: List of reward functions
        expanded_prompts: Prompts for each completion
        all_completion_texts: Completion texts
        expanded_answers: Answers for each completion
        expanded_types: Type info for each completion
        batch_indices: Prompt index for each completion
        unique_prompt_indices: Unique prompt indices
        reward_weights: Weights for each reward function
        scaffold_ratios: Scaffold ratio for each completion
        scaffold_penalty_weight: Penalty for scaffold assistance
        scaffold_penalty_mode: "multiplicative" or "additive"
        exam_compute_reward: Optional exam reward function

    Returns:
        Tuple of (advantages, reward_metrics, per_completion_rewards)
    """
    # Detect exam samples
    exam_mask = []
    ground_truths = []
    possible_boxed_answers_list = []
    for i, type_info in enumerate(expanded_types):
        is_exam = False
        ground_truth = None
        possible_boxed_answers = None
        if isinstance(type_info, dict):
            is_exam = type_info.get("is_exam", False)
            ground_truth = type_info.get("ground_truth", None)
            possible_boxed_answers = type_info.get("possible_boxed_answers", None)
        elif isinstance(type_info, str):
            is_exam = "exam" in type_info.lower()
        exam_mask.append(is_exam)
        ground_truths.append(ground_truth)
        possible_boxed_answers_list.append(possible_boxed_answers)

    has_exam_samples = any(exam_mask)

    # Calculate rewards from all functions
    all_func_rewards = []
    for reward_func in reward_funcs:
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
            types=expanded_types,
        )
        if raw_rewards is None:
            processed_rewards = [float("nan")] * len(all_completion_texts)
        else:
            processed_rewards = [
                float(r) if r is not None else float("nan") for r in raw_rewards
            ]
        func_rewards = mx.array(processed_rewards)
        all_func_rewards.append(func_rewards)

    rewards = mx.stack(all_func_rewards, axis=1)
    _safe_eval(rewards, checkpoint="rewards_stack")

    # Override rewards for exam samples
    exam_details_list: list[Any] = [None] * len(all_completion_texts)

    if has_exam_samples and exam_compute_reward is not None:
        rewards_np = np.array(rewards)

        for i, (is_exam, ground_truth, possible_boxed, completion) in enumerate(
            zip(exam_mask, ground_truths, possible_boxed_answers_list, all_completion_texts)
        ):
            if is_exam and (ground_truth is not None or possible_boxed is not None):
                exam_score, exam_details = exam_compute_reward(
                    completion,
                    ground_truth,
                    possible_boxed_answers=possible_boxed
                )
                rewards_np[i, :] = exam_score
                exam_details_list[i] = exam_details

        rewards = mx.array(rewards_np)
        _safe_eval(rewards, checkpoint="rewards_exam")

    # Check for all NaN rows
    all_nan_rows = mx.all(mx.isnan(rewards), axis=1)
    if mx.any(all_nan_rows):
        nan_row_idx = mx.argmax(all_nan_rows).item()
        raise RuntimeError(
            f"All reward functions returned None for prompt: {expanded_prompts[nan_row_idx]}"
        )

    # Apply reward weights
    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError("Number of reward weights must match number of reward functions")
        reward_weights_arr = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights_arr = mx.ones(len(reward_funcs), dtype=mx.float32)

    # Compute weighted sum
    valid_reward_mask = ~mx.isnan(rewards)
    rewards_no_nan = mx.where(valid_reward_mask, rewards, mx.zeros_like(rewards))
    rewards = (rewards_no_nan * mx.expand_dims(reward_weights_arr, 0)).sum(axis=1)
    _safe_eval(rewards, checkpoint="rewards_weighted")

    raw_rewards_for_metrics = rewards

    # Apply scaffold penalty
    if scaffold_ratios is not None and scaffold_penalty_weight > 0.0:
        scaffold_ratios_arr = mx.array(scaffold_ratios, dtype=mx.float32)

        if scaffold_penalty_mode == "multiplicative":
            adjustment = 1.0 - (scaffold_ratios_arr * scaffold_penalty_weight)
            rewards = rewards * adjustment
        elif scaffold_penalty_mode == "additive":
            penalty = scaffold_ratios_arr * scaffold_penalty_weight
            rewards = rewards - penalty

    # Group rewards by prompt
    num_unique_prompts = len(unique_prompt_indices)
    rewards_by_prompt: list[list[mx.array]] = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    # Calculate advantages
    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards in enumerate(rewards_by_prompt):
        if len(prompt_rewards) > 1:
            prompt_rewards_arr = mx.array(prompt_rewards)
            mean_reward = mx.mean(prompt_rewards_arr)
            std_reward = mx.std(prompt_rewards_arr)
            indices = [
                j for j, idx in enumerate(batch_indices)
                if idx == unique_prompt_indices[i]
            ]
            for j, idx in enumerate(indices):
                advantages[idx] = (prompt_rewards_arr[j] - mean_reward) / (std_reward + 1e-4)
        else:
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0

    _safe_eval(advantages, checkpoint="advantages")

    # Calculate reward metrics
    reward_metrics: dict[str, Any] = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
        )
        valid_mask = ~mx.isnan(
            mx.array([r if r is not None else float("nan") for r in raw_rewards])
        )
        valid_rewards = mx.array([r for r in raw_rewards if r is not None and not np.isnan(r)])
        if len(valid_rewards) > 0:
            reward_metrics[f"{func_name}_mean"] = mx.mean(valid_rewards)
            reward_metrics[f"{func_name}_std"] = mx.std(valid_rewards) if len(valid_rewards) > 1 else mx.zeros(1)
            reward_metrics[f"{func_name}_coverage"] = valid_mask.sum() / len(raw_rewards)
        else:
            reward_metrics[f"{func_name}_mean"] = float("nan")
            reward_metrics[f"{func_name}_std"] = float("nan")
            reward_metrics[f"{func_name}_coverage"] = 0.0

    # Grouped rewards statistics
    grouped_rewards_mean = mx.array(
        [mx.mean(mx.array(rewards_list)) for rewards_list in rewards_by_prompt]
    )
    grouped_rewards_std = mx.array([
        mx.std(mx.array(rewards_list)) if len(rewards_list) > 1 else mx.zeros(1)
        for rewards_list in rewards_by_prompt
    ])
    _safe_eval(grouped_rewards_mean, grouped_rewards_std, checkpoint="grouped_rewards")

    reward_specific_metrics = {
        "total_rewards_mean": mx.mean(rewards),
        "total_rewards_std": mx.std(rewards),
        "grouped_rewards_mean": mx.mean(grouped_rewards_mean),
        "grouped_rewards_std": mx.mean(grouped_rewards_std),
        **reward_metrics,
    }

    if scaffold_ratios is not None and scaffold_penalty_weight > 0.0:
        reward_specific_metrics["scaffold_penalty_applied"] = 1.0
        reward_specific_metrics["scaffold_penalty_weight"] = scaffold_penalty_weight
        reward_specific_metrics["raw_rewards_mean"] = mx.mean(raw_rewards_for_metrics)

    # Per-completion rewards dict
    per_completion_rewards: dict[str, list[float]] = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        func_rewards = all_func_rewards[i]
        per_completion_rewards[func_name] = [float(r) for r in func_rewards.tolist()]
    per_completion_rewards["total"] = [float(r) for r in rewards.tolist()]
    per_completion_rewards["exam_details"] = exam_details_list  # type: ignore

    return advantages, reward_specific_metrics, per_completion_rewards


def grpo_loss(
    model: nn.Module,
    ref_model: nn.Module | None,
    batch: tuple[Any, ...],
    completions: list[mx.array] | None = None,
    completion_texts: list[str] | None = None,
    batch_indices: list[int] | None = None,
    advantages: mx.array | None = None,
    reward_metrics: dict[str, Any] | None = None,
    beta: float = 0.1,
    epsilon: float = 1e-4,
    epsilon_high: float | None = None,
    max_tokens: int = 64,
    importance_sampling_level: str | None = "token",
    grpo_loss_type: str = "grpo",
    token_type_mask: mx.array | None = None,
    scaffold_token_counts: list[int] | None = None,
) -> tuple[mx.array, mx.array, dict[str, Any]]:
    """Compute GRPO loss for a batch.

    Args:
        model: The policy model
        ref_model: Reference model (None to use model)
        batch: Tuple of (prompt_tokens, answer_tokens, prompt_text, answer_text, type_info)
        completions: Generated completions
        completion_texts: Completion strings
        batch_indices: Prompt index for each completion
        advantages: Pre-computed advantages
        reward_metrics: Pre-computed reward metrics
        beta: KL penalty coefficient
        epsilon: Lower-bound epsilon for clipping
        epsilon_high: Upper-bound epsilon for clipping
        max_tokens: Maximum tokens
        importance_sampling_level: "token", "sequence", or None
        grpo_loss_type: "grpo", "bnpo", or "dr_grpo"
        token_type_mask: Optional mask for dual-gradient mode
        scaffold_token_counts: Number of scaffold/injected tokens to mask for each completion.
            These are tokens from curriculum learning or two-phase recovery that the model
            didn't generate and should be excluded from loss computation.

    Returns:
        Tuple of (loss, ntokens, metrics)
    """
    prompt_tokens_batch, _, prompt_text, answer_text, type_info = batch

    all_completions = completions
    all_completion_texts = completion_texts

    if not all_completions:
        raise ValueError("No completions were generated.")

    # Build full sequences (prompt + completion)
    full_sequences = []
    prompt_lengths = []
    completion_lengths = []

    for i, completion_ids in enumerate(all_completions):
        prompt_idx = batch_indices[i]
        prompt = prompt_tokens_batch[prompt_idx]

        if isinstance(prompt, mx.array):
            prompt = prompt.tolist()
        if isinstance(completion_ids, mx.array):
            completion = completion_ids.tolist()
        else:
            completion = list(completion_ids)

        prompt_len = len(prompt)
        completion_len = len(completion)

        full_seq = prompt + completion

        full_sequences.append(full_seq)
        prompt_lengths.append(prompt_len)
        completion_lengths.append(completion_len)

    # Adjust prompt lengths to include scaffold tokens (which shouldn't have loss computed)
    # Scaffold tokens are at the START of the completion and were not generated by the model
    if scaffold_token_counts is not None:
        adjusted_prompt_lengths = []
        for i in range(len(prompt_lengths)):
            scaffold_count = scaffold_token_counts[i] if i < len(scaffold_token_counts) else 0
            adjusted_prompt_lengths.append(prompt_lengths[i] + scaffold_count)
        prompt_lengths = adjusted_prompt_lengths

    # Pad sequences
    max_length = max(len(s) for s in full_sequences)
    padded_sequences = []
    total_lengths = []

    for seq in full_sequences:
        total_len = len(seq)
        total_lengths.append(total_len)

        if len(seq) < max_length:
            padded = seq + [0] * (max_length - len(seq))
        else:
            padded = seq
        padded_sequences.append(padded)

    inputs = mx.array(padded_sequences)
    total_lengths_arr = mx.array(total_lengths)
    prompt_lengths_arr = mx.array(prompt_lengths)

    # Calculate log probs for completion tokens
    token_log_probs = get_per_token_logps_with_prompt_mask(
        model, inputs, total_lengths_arr, prompt_lengths_arr
    )
    mx.eval(token_log_probs)

    if ref_model is None:
        ref_token_log_probs = token_log_probs
    else:
        ref_token_log_probs = get_per_token_logps_with_prompt_mask(
            ref_model, inputs, total_lengths_arr, prompt_lengths_arr
        )
        mx.eval(ref_token_log_probs)
        mx.clear_cache()

    del inputs
    mx.clear_cache()

    completion_lens = mx.array([x.shape[0] for x in token_log_probs])

    # Pad log probs
    max_len = max(x.shape[0] for x in token_log_probs)
    padded_log_probs = []
    padded_ref_log_probs = []

    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        padding = mx.zeros((max_len - seq_len,))

        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)
    _safe_eval(token_log_probs, ref_token_log_probs, completion_lens, checkpoint="log_probs_stack")

    # Create mask
    length_mask = mx.arange(token_log_probs.shape[1])[None, :] < completion_lens[:, None]

    # Combine with token_type_mask if provided
    if token_type_mask is not None:
        if token_type_mask.shape[1] < length_mask.shape[1]:
            padding = mx.zeros((token_type_mask.shape[0], length_mask.shape[1] - token_type_mask.shape[1]))
            token_type_mask = mx.concatenate([token_type_mask, padding], axis=1)
        elif token_type_mask.shape[1] > length_mask.shape[1]:
            token_type_mask = token_type_mask[:, :length_mask.shape[1]]
        effective_mask = length_mask * token_type_mask
    else:
        effective_mask = length_mask

    # Compute log ratio
    log_ratio = token_log_probs - mx.stop_gradient(ref_token_log_probs)

    # Apply importance sampling
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        sequence_log_ratio = (log_ratio * length_mask).sum(axis=1) / mx.maximum(
            length_mask.sum(axis=1), 1.0
        )
        log_importance_weights = mx.expand_dims(sequence_log_ratio, axis=1)
    elif importance_sampling_level is None or importance_sampling_level == "none":
        log_importance_weights = mx.zeros_like(log_ratio)
    else:
        raise ValueError(f"Unknown importance sampling level: {importance_sampling_level}")

    coef_1 = mx.exp(log_importance_weights)

    # PPO-like clipping
    epsilon_high_val = epsilon_high if epsilon_high else epsilon
    coef_2 = mx.clip(coef_1, 1 - epsilon, 1 + epsilon_high_val)

    # Track clipping
    is_low_clipped = (coef_1 < 1 - epsilon) & (advantages.reshape(-1, 1) < 0)
    is_high_clipped = (coef_1 > 1 + epsilon_high_val) & (advantages.reshape(-1, 1) > 0)
    is_region_clipped = is_low_clipped | is_high_clipped

    # Compute objectives
    unclipped_obj = coef_1 * advantages.reshape(-1, 1)
    clipped_obj = coef_2 * advantages.reshape(-1, 1)
    per_token_loss = -mx.minimum(unclipped_obj, clipped_obj)

    # KL penalty
    if beta != 0.0:
        log_ratio_ref_theta = ref_token_log_probs - token_log_probs
        ratio_ref_theta = mx.exp(log_ratio_ref_theta)
        kl_div = coef_1 * ratio_ref_theta - log_ratio_ref_theta - 1
        per_token_loss = per_token_loss + beta * kl_div
    else:
        kl_div = (
            mx.exp(ref_token_log_probs - token_log_probs)
            - (ref_token_log_probs - token_log_probs)
            - 1
        )

    # Compute loss
    if grpo_loss_type == "grpo":
        loss = (per_token_loss * effective_mask).sum() / mx.maximum(effective_mask.sum(), 1.0)
    elif grpo_loss_type == "bnpo":
        loss = (per_token_loss * effective_mask).sum() / mx.maximum(effective_mask.sum(), 1.0)
    elif grpo_loss_type == "dr_grpo":
        loss = (per_token_loss * effective_mask).sum() / (per_token_loss.shape[0] * max_tokens)
    else:
        raise ValueError(f"Unknown loss type: {grpo_loss_type}")

    # Metrics
    mean_kl = ((kl_div * length_mask).sum(axis=1) / mx.maximum(length_mask.sum(axis=1), 1.0)).mean()

    comp_lengths = [comp.shape[0] for comp in all_completions]
    max_generated = max(comp_lengths) if comp_lengths else 0
    min_generated = min(comp_lengths) if comp_lengths else 0
    avg_generated = sum(comp_lengths) / len(comp_lengths) if comp_lengths else 0
    hit_max_tokens = sum(1 for length in comp_lengths if length >= max_tokens)
    hit_max_ratio = hit_max_tokens / len(comp_lengths) if comp_lengths else 0

    metrics = {
        "kl": mean_kl,
        "average_generated_tokens": avg_generated,
        "max_generated_tokens": max_generated,
        "min_generated_tokens": min_generated,
        "hit_max_tokens_ratio": hit_max_ratio,
        "clip_ratio_low": (
            (is_low_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        "clip_ratio_high": (
            (is_high_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        "clip_ratio_total": (
            (is_region_clipped * length_mask).sum() / length_mask.sum()
            if length_mask.sum() > 0
            else mx.zeros(1)
        ),
        **(reward_metrics or {}),
    }

    _safe_eval(loss, checkpoint="grpo_loss_final")
    _safe_clear("grpo_loss")

    return loss, length_mask.sum(axis=1).sum(), metrics
