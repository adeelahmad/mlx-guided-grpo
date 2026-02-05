"""Configuration dataclass for GRPO training.

This module provides:
- GRPOTrainingArgs dataclass with all training hyperparameters
- Organized into logical sections with metadata for CLI generation

SOLID Principles:
- Single Responsibility: Only handles configuration definition
- Open/Closed: Can be extended with additional fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..base import BaseTrainingArgs

if TYPE_CHECKING:
    from typing import Optional

__all__ = ["GRPOTrainingArgs"]


@dataclass
class GRPOTrainingArgs(BaseTrainingArgs):
    """GRPO (Group Relative Policy Optimization) training configuration.

    This dataclass extends BaseTrainingArgs with GRPO-specific parameters.
    Fields are organized into logical sections for clarity.

    Example:
        >>> args = GRPOTrainingArgs(
        ...     batch_size=4,
        ...     group_size=4,
        ...     beta=0.1,
        ...     temperature=0.8,
        ... )
    """

    # ==== Core GRPO Parameters ====
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient."},
    )
    epsilon: float = field(
        default=1e-4,
        metadata={"help": "The Epsilon for numerical stability."},
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={
            "help": "For DAPO: Upper-bound epsilon value for clipping. "
            "If not specified, defaults to the lower-bound epsilon."
        },
    )
    max_completion_length: int = field(
        default=512,
        metadata={"help": "Maximum completion tokens for generation."},
    )
    generation_sub_batch_size: int = field(
        default=1,
        metadata={
            "help": "Number of completions to generate at once. Lower values prevent GPU "
            "timeouts but may be slower. Set to 1 to generate one completion at a time."
        },
    )
    cross_sample_max_completion_length: int = field(
        default=512,
        metadata={
            "help": "Maximum completion tokens for cross-sampled examples. "
            "Cross-sampled examples have conversation history in the prompt."
        },
    )
    reference_model_path: str | None = field(
        default=None,
        metadata={"help": "Path to reference model weights. If None, uses the same model."},
    )
    temperature: float = field(
        default=0.8,
        metadata={"help": "Temperature for sampling. Higher = more random completions."},
    )
    grpo_loss_type: str = field(
        default="grpo",
        metadata={"help": "Type of loss: 'grpo', 'bnpo', 'dr_grpo'."},
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match number of reward functions. "
            "If None, all rewards weighted equally with weight 1.0."
        },
    )
    importance_sampling_level: str | None = field(
        default=None,
        metadata={
            "help": "Controls importance sampling ratios: 'token' or 'sequence' level. "
            "Sequence-level often yields more stable training."
        },
    )

    # ==== Dataset Caching ====
    cache_dataset: bool = field(
        default=True,
        metadata={"help": "Enable dataset caching for faster subsequent loads."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Cache directory. Default: {data_dir}/.cache/"},
    )
    force_reload: bool = field(
        default=False,
        metadata={"help": "Force reload dataset ignoring cache."},
    )

    # ==== Cross-Sampling ====
    cross_sampling_enabled: bool = field(
        default=False,
        metadata={"help": "Enable cross-sampling of conversation histories."},
    )
    cross_sampling_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of samples to cross-sample (0.0-1.0)."},
    )
    cross_sampling_max_history_tokens: int = field(
        default=512,
        metadata={"help": "Maximum tokens for cross-sampled history."},
    )
    cross_sampling_seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducible cross-sampling."},
    )
    cross_sampling_truncation_marker: str = field(
        default="\n[...truncated for brevity...]\n",
        metadata={"help": "Marker to insert when truncating thinking sections."},
    )

    # ==== Dataset Options ====
    shuffle_data: bool = field(
        default=True,
        metadata={"help": "Shuffle training data at start."},
    )
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "Random seed for data shuffling."},
    )
    require_think_tags: bool = field(
        default=True,
        metadata={"help": "Skip samples without <think> tags in answer."},
    )

    # ==== Crash Recovery ====
    auto_resume_on_crash: bool = field(
        default=True,
        metadata={"help": "Automatically resume training from last checkpoint on Metal GPU crash."},
    )
    max_crash_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of crash recovery attempts."},
    )
    crash_cooldown_seconds: int = field(
        default=10,
        metadata={"help": "Seconds to wait before resuming after a crash."},
    )

    # ==== Rollout Logging ====
    log_rollouts: bool = field(
        default=True,
        metadata={"help": "Enable comprehensive rollout logging."},
    )
    log_rollouts_every_n_steps: int = field(
        default=1,
        metadata={"help": "Log rollouts every N training steps."},
    )
    log_rollouts_to_wandb: bool = field(
        default=True,
        metadata={"help": "Upload rollout logs to WandB."},
    )
    rollout_log_file: str | None = field(
        default=None,
        metadata={"help": "Custom path for rollout log files."},
    )

    # ==== Checkpoint Management ====
    keep_last_n_checkpoints: int = field(
        default=5,
        metadata={"help": "Keep only the last N checkpoints. Set to 0 to keep all."},
    )
    keep_best_n_checkpoints: int = field(
        default=3,
        metadata={"help": "Keep the best N checkpoints by validation loss. Set to 0 to disable."},
    )
    checkpoint_metric: str = field(
        default="val_loss",
        metadata={"help": "Metric for determining best checkpoints (val_loss, reward_mean)."},
    )
    checkpoint_metric_higher_is_better: bool = field(
        default=False,
        metadata={"help": "If True, higher metric values are better (e.g., reward)."},
    )

    # ==== Training Monitor ====
    enable_monitor: bool = field(
        default=True,
        metadata={"help": "Enable real-time training monitor with threshold alerts."},
    )
    monitor_kl_warning: float = field(
        default=0.025,
        metadata={"help": "KL divergence warning threshold."},
    )
    monitor_kl_critical: float = field(
        default=0.04,
        metadata={"help": "KL divergence critical threshold (triggers stop)."},
    )
    monitor_reward_warning: float = field(
        default=0.40,
        metadata={"help": "Reward warning threshold (below this is warning)."},
    )
    monitor_reward_critical: float = field(
        default=0.30,
        metadata={"help": "Reward critical threshold (below this triggers stop)."},
    )
    monitor_stop_on_critical: bool = field(
        default=True,
        metadata={"help": "Stop training after consecutive critical metrics."},
    )
    monitor_critical_count: int = field(
        default=3,
        metadata={"help": "Number of consecutive criticals before stopping."},
    )

    # ==== Resume Training ====
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Resume training from saved optimizer state if available."},
    )
    resume_iteration: int | None = field(
        default=None,
        metadata={"help": "Iteration to resume from. If None, reads from training_state.json."},
    )

    # ==== Selective Layer Training (CGS) ====
    train_layers: str | None = field(
        default=None,
        metadata={"help": "Layers to train. Format: '0-8,20-28' or 'all'. Default: all layers."},
    )
    thinking_layers: str | None = field(
        default=None,
        metadata={
            "help": "Layers for thinking token gradients. Format: '0-8'. "
            "If set, enables dual-gradient mode."
        },
    )
    answer_layers: str | None = field(
        default=None,
        metadata={
            "help": "Layers for answer token gradients. Format: '20-28'. "
            "Required if thinking_layers is set."
        },
    )
    thinking_gradient_weight: float = field(
        default=1.0,
        metadata={"help": "Weight multiplier for thinking token gradients."},
    )
    answer_gradient_weight: float = field(
        default=1.0,
        metadata={"help": "Weight multiplier for answer token gradients."},
    )

    # ==== Two-Phase Generation (Enforce Thinking) ====
    enforce_thinking: bool = field(
        default=False,
        metadata={"help": "Enable two-phase generation to recover incomplete thinking outputs."},
    )
    think_start_token: str = field(
        default="<think>",
        metadata={"help": "Start marker for thinking section."},
    )
    think_end_token: str = field(
        default="</think>",
        metadata={"help": "End marker for thinking section."},
    )
    continuation_tokens: int = field(
        default=256,
        metadata={"help": "Max tokens for continuation phase when recovering incomplete thinking."},
    )
    continuation_force_answer_ratio: float = field(
        default=0.8,
        metadata={
            "help": "Ratio of incomplete completions that force answer (add </think>). "
            "Rest continue naturally. Range [0.0, 1.0]. "
            "Mixed values let model learn that verbose thinking leads to truncation."
        },
    )
    two_phase_samples_per_group: int = field(
        default=-1,
        metadata={
            "help": "Max samples per group to apply two-phase recovery. "
            "-1 means all incomplete samples get recovery. "
            "E.g., group_size=4, two_phase_samples_per_group=2 means "
            "only 2 incomplete samples per group get 2nd phase recovery."
        },
    )

    # ==== Smart Truncation ====
    smart_truncation_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable smart truncation: let model generate until natural </think>, "
            "then truncate middle if over max_completion_length."
        },
    )
    max_extreme_tokens: int = field(
        default=1024,
        metadata={
            "help": "Maximum tokens to generate before forcing closure. "
            "Should be 2-4x max_completion_length."
        },
    )
    truncation_brevity_marker: str = field(
        default="[truncated due to brevity]",
        metadata={"help": "Marker to insert when truncating thinking middle."},
    )
    truncation_keep_start_ratio: float = field(
        default=0.3,
        metadata={"help": "Ratio of thinking to keep from start when truncating (0.0-1.0)."},
    )
    truncation_keep_end_ratio: float = field(
        default=0.5,
        metadata={"help": "Ratio of thinking to keep from end when truncating (0.0-1.0)."},
    )

    # ==== SFT Anchor + Gradient Alignment ====
    sft_anchor_enabled: bool = field(
        default=False,
        metadata={"help": "Enable SFT anchor step before GRPO to ground format learning."},
    )
    sft_anchor_layers: str | None = field(
        default=None,
        metadata={"help": "Layers to apply SFT anchor (e.g., '20-28'). None = all layers."},
    )
    sft_anchor_lr_multiplier: float = field(
        default=0.1,
        metadata={"help": "Learning rate multiplier for SFT anchor (relative to main LR)."},
    )
    gradient_alignment_mode: str = field(
        default="none",
        metadata={"help": "Gradient alignment mode: 'none', 'cosine', 'kl', 'interpolate'."},
    )
    gradient_alignment_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for gradient alignment/interpolation (0-1)."},
    )

    # ==== Curriculum Thinking Scaffolding ====
    curriculum_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable curriculum thinking scaffolding (gradually remove target thinking)."
        },
    )
    curriculum_start_ratio: float = field(
        default=1.0,
        metadata={"help": "Initial ratio of target thinking to include (1.0 = full thinking)."},
    )
    curriculum_end_ratio: float = field(
        default=0.0,
        metadata={"help": "Final ratio of target thinking (0.0 = no thinking)."},
    )
    curriculum_warmup_iters: int = field(
        default=100,
        metadata={"help": "Iterations to stay at start_ratio before tapering."},
    )
    curriculum_taper_iters: int = field(
        default=500,
        metadata={"help": "Iterations to linearly taper from start_ratio to end_ratio."},
    )
    curriculum_by_lines: bool = field(
        default=True,
        metadata={"help": "Remove thinking by lines (True) or by characters (False)."},
    )
    curriculum_truncation_mode: str = field(
        default="prefix",
        metadata={
            "help": "How to truncate thinking: 'prefix' = keep start only, 'middle' = keep start + end."
        },
    )
    curriculum_preserve_intuition: bool = field(
        default=True,
        metadata={"help": "Always preserve [ANSWER INTUITION: ...] block."},
    )

    # ==== Multi-Curriculum Rollout ====
    multi_curriculum_rollout: bool = field(
        default=False,
        metadata={"help": "Use different scaffold levels for each completion in group."},
    )
    curriculum_scaffold_levels: str | None = field(
        default=None,
        metadata={"help": "Comma-separated scaffold ratios (e.g., '1.0,0.66,0.33,0.0')."},
    )

    # ==== Scaffold-Aware Rewards ====
    scaffold_penalty_weight: float = field(
        default=0.0,
        metadata={"help": "Penalty weight for scaffold-assisted completions."},
    )
    scaffold_penalty_mode: str = field(
        default="multiplicative",
        metadata={"help": "How to apply scaffold penalty: 'multiplicative' or 'additive'."},
    )

    # ==== Structured Batching ====
    samples_per_scaffold: int = field(
        default=1,
        metadata={"help": "Number of samples to generate per scaffold level."},
    )
