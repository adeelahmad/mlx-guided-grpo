"""GRPO Training CLI for MLX-GRPO.

This module provides the main entry point for GRPO (Group Relative Policy Optimization)
training on Apple Silicon using MLX.

Usage:
    python -m mlx_grpo train --model <model> --data <data_path> [options]

    # Or use the CLI:
    mlx-grpo --model <model> --data <data_path> [options]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx_lm.tuner.callbacks import WandBCallback as _BaseWandBCallback
from mlx_lm.tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)
from mlx_lm.utils import load, load_tokenizer, save_config

from .trainer.datasets import CacheDataset, load_dataset
from .trainer.grpo_reward_functions import (
    REWARD_REGISTRY,
    get_default_reward_functions,
    get_reward_function,
    list_available_reward_functions,
)
from .trainer.grpo_trainer import GRPOTrainingArgs, evaluate_grpo, train_grpo
from .trainer.type_system_v2.bridge import create_v2_coordinator, v2_reward_adapter
from .utils import from_pretrained, fuse_and_save_model
from .visuals import (
    Colors,
    print_banner,
    print_error,
    print_info,
    print_section,
    print_success,
    print_warning,
)

if TYPE_CHECKING:
    from mlx_lm.tuner.callbacks import TrainingCallback

__all__ = ["main", "run", "build_parser", "CONFIG_DEFAULTS"]

# =============================================================================
# YAML Loader Configuration
# =============================================================================

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


# =============================================================================
# WandB Callback with Resume Support
# =============================================================================


class WandBCallback(_BaseWandBCallback):
    """Custom WandBCallback with proper iteration step for resume support."""

    def __init__(
        self,
        project_name: str | None = None,
        log_dir: str | None = None,
        config: dict | None = None,
        wrapped_callback: Any = None,
    ):
        self._project_name = project_name
        self._log_dir = log_dir
        self._config = config
        self._wrapped_callback = wrapped_callback
        self._initialized = False

    def _ensure_wandb_init(self) -> None:
        if self._initialized:
            return
        try:
            import wandb

            if wandb.run is None:
                run_id = None
                if self._log_dir:
                    wandb_id_file = Path(self._log_dir) / "wandb_run_id.txt"
                    if wandb_id_file.exists():
                        run_id = wandb_id_file.read_text().strip()
                        logging.info(f"[WandB] Resuming run: {run_id}")

                wandb.init(
                    project=self._project_name or "grpo-training",
                    config=self._config,
                    dir=self._log_dir,
                    resume="allow",
                    id=run_id,
                )

                if self._log_dir and wandb.run:
                    wandb_id_file = Path(self._log_dir) / "wandb_run_id.txt"
                    wandb_id_file.parent.mkdir(parents=True, exist_ok=True)
                    wandb_id_file.write_text(wandb.run.id)

            self._initialized = True
        except Exception as e:
            logging.warning(f"[WandB] Initialization warning: {e}")

    def on_train_loss_report(self, train_info: dict) -> None:
        self._ensure_wandb_init()
        try:
            import wandb

            if wandb.run is not None:
                step = train_info.get("iteration")
                metrics = {
                    "train/loss": train_info.get("train_loss"),
                    "train/learning_rate": train_info.get("learning_rate"),
                    "train/tokens_per_second": train_info.get("tokens_per_second"),
                    "train/iterations_per_second": train_info.get("iterations_per_second"),
                }
                for k, v in train_info.items():
                    if k.startswith("train_") and k != "train_loss":
                        metrics[f"train/{k[6:]}"] = v
                if "peak_memory" in train_info:
                    metrics["train/peak_memory_gb"] = train_info["peak_memory"]
                metrics = {k: v for k, v in metrics.items() if v is not None}
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
        except Exception:
            pass

        if self._wrapped_callback:
            self._wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict) -> None:
        self._ensure_wandb_init()
        try:
            import wandb

            if wandb.run is not None:
                step = val_info.get("iteration")
                metrics = {
                    "val/loss": val_info.get("val_loss"),
                    "val/time": val_info.get("val_time"),
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
        except Exception:
            pass

        if self._wrapped_callback:
            self._wrapped_callback.on_val_loss_report(val_info)


# =============================================================================
# Configuration Defaults
# =============================================================================

CONFIG_DEFAULTS: dict[str, Any] = {
    # Model & Training
    "model": "mlx_model",
    "train": False,
    "test": False,
    "load_in_4bits": False,
    "load_in_6bits": False,
    "load_in_8bits": False,
    "train_type": "lora",
    "force_dora": False,
    "optimizer": "adam",
    "optimizer_config": {"adam": {}, "adamw": {}, "muon": {}},
    "data": "data/",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": None,
    "epochs": None,
    "gradient_accumulation_steps": 1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test_batches": 500,
    "max_seq_length": 2048,
    "config": None,
    "grad_checkpoint": False,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.05, "scale": 16.0},
    "fuse": True,
    # GRPO-Specific
    "beta": 0.1,
    "group_size": 4,
    "epsilon": 1e-4,
    "epsilon_high": None,
    "max_completion_length": 512,
    "cross_sample_max_completion_length": 512,
    "temperature": 0.8,
    "reward_weights": None,
    "reward_functions": None,
    "reward_functions_file": None,
    "grpo_loss_type": "grpo",
    "importance_sampling_level": None,
    "reference_model_path": None,
    # Dataset Caching
    "cache_dataset": True,
    "cache_dir": None,
    "force_reload": False,
    # Cross-Sampling
    "cross_sampling_enabled": False,
    "cross_sampling_ratio": 0.0,
    "cross_sampling_max_history_tokens": 128,
    "cross_sampling_seed": 42,
    "cross_sampling_truncation_marker": "\n[...truncated for brevity...]\n",
    # Dataset Options
    "shuffle_data": True,
    "balanced_shuffle": True,
    "shuffle_seed": 42,
    "require_think_tags": True,
    # Crash Recovery
    "auto_resume_on_crash": True,
    "max_crash_retries": 3,
    "crash_cooldown_seconds": 10,
    # Rollout Logging
    "log_rollouts": True,
    "log_rollouts_every_n_steps": 1,
    "log_rollouts_to_wandb": True,
    "rollout_log_file": None,
    # Checkpoint Management
    "keep_last_n_checkpoints": 5,
    "keep_best_n_checkpoints": 3,
    "checkpoint_metric": "val_loss",
    "checkpoint_metric_higher_is_better": False,
    # Training Monitor
    "enable_monitor": False,
    "monitor_kl_warning": 0.025,
    "monitor_kl_critical": 0.04,
    "monitor_reward_warning": 0.40,
    "monitor_reward_critical": 0.30,
    "monitor_stop_on_critical": True,
    "monitor_critical_count": 3,
    # Resume Training
    "resume_from_checkpoint": False,
    "resume_iteration": None,
    # Selective Layer Training
    "train_layers": None,
    "thinking_layers": None,
    "answer_layers": None,
    "thinking_gradient_weight": 1.0,
    "answer_gradient_weight": 1.0,
    # Two-Phase Generation
    "enforce_thinking": False,
    "think_start_token": "<think>",
    "think_end_token": "</think>",
    "continuation_tokens": 256,
    "continuation_force_answer_ratio": 0.8,
    "two_phase_samples_per_group": -1,
    "exam_phase_recovery_ratio": 0.5,
    # Smart Truncation
    "smart_truncation_enabled": False,
    "max_extreme_tokens": 1024,
    "truncation_brevity_marker": "[truncated due to brevity]",
    "truncation_keep_start_ratio": 0.3,
    "truncation_keep_end_ratio": 0.5,
    # SFT Anchor
    "sft_anchor_enabled": False,
    "sft_anchor_layers": None,
    "sft_anchor_lr_multiplier": 0.1,
    "gradient_alignment_mode": "none",
    "gradient_alignment_weight": 0.3,
    # Curriculum Scaffolding
    "curriculum_enabled": True,
    "curriculum_start_ratio": 0.85,
    "curriculum_end_ratio": 0.25,
    "curriculum_warmup_iters": 100,
    "curriculum_taper_iters": 500,
    "curriculum_by_lines": True,
    "curriculum_truncation_mode": "prefix",
    "curriculum_preserve_intuition": True,
    # Multi-Curriculum Rollout
    "multi_curriculum_rollout": True,
    "curriculum_scaffold_levels": None,
    # Scaffold-Aware Rewards
    "scaffold_penalty_weight": 0.0,
    "scaffold_penalty_mode": "multiplicative",
    "samples_per_scaffold": 1,
}


# =============================================================================
# Utility Functions
# =============================================================================


def load_reward_functions_from_file(file_path: str) -> bool | None:
    """Load reward functions from a Python file."""
    if not file_path or not Path(file_path).exists():
        return None
    try:
        logging.info(f"Loading custom reward functions from {file_path}")
        spec = importlib.util.spec_from_file_location("custom_rewards", file_path)
        custom_rewards = importlib.util.module_from_spec(spec)
        sys.modules["custom_rewards"] = custom_rewards
        spec.loader.exec_module(custom_rewards)
        logging.info("Successfully loaded custom reward functions")
        return True
    except Exception as e:
        logging.error(f"Error loading custom reward functions: {e}")
        return None


def calculate_iters(train_set: Any, batch_size: int, epochs: int) -> int:
    """Calculate total iterations from epochs."""
    num_samples = len(train_set)
    batches_per_epoch = math.ceil(num_samples / batch_size)
    iters = epochs * batches_per_epoch
    logging.info(
        f"Calculated {iters} iterations from {epochs} epochs "
        f"(dataset size: {num_samples}, batch size: {batch_size})"
    )
    return iters


def load_reference_model(args: argparse.Namespace) -> nn.Module:
    """Load reference model for GRPO training."""
    if args.reference_model_path:
        logging.info(f"Loading reference model from {args.reference_model_path}")
        model, _ = load(args.reference_model_path)
    else:
        logging.info("Loading reference model (using main model)")
        model, _ = load(args.model)
    return model.freeze()


# =============================================================================
# Argument Parser
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for GRPO training."""
    parser = argparse.ArgumentParser(
        description="GRPO Training for MLX-LM-LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, help="Model path or HuggingFace repo")
    model_group.add_argument("--load-in-4bits", action="store_true", default=None)
    model_group.add_argument("--load-in-6bits", action="store_true", default=None)
    model_group.add_argument("--load-in-8bits", action="store_true", default=None)
    model_group.add_argument("--reference-model-path", type=str, default=None)

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--train", action="store_true", help="Enable training")
    train_group.add_argument("--test", action="store_true", help="Enable testing")
    train_group.add_argument("--train-type", choices=["lora", "dora", "full"], default=None)
    train_group.add_argument(
        "--force-dora",
        action="store_true",
        default=False,
        help="Force DoRA even with quantized models (dequantizes weights each step)",
    )
    train_group.add_argument("--optimizer", choices=["adam", "adamw", "muon"], default=None)
    train_group.add_argument("--batch-size", type=int, default=None)
    train_group.add_argument("--iters", type=int, default=None)
    train_group.add_argument("--epochs", type=int, default=None)
    train_group.add_argument("--learning-rate", type=float, default=None)
    train_group.add_argument("--gradient-accumulation-steps", type=int, default=None)
    train_group.add_argument("--num-layers", type=int, default=None)
    train_group.add_argument("--max-seq-length", type=int, default=None)
    train_group.add_argument("--grad-checkpoint", action="store_true", default=None)
    train_group.add_argument("--lr-schedule", type=str, default=None)
    train_group.add_argument("--seed", type=int, default=None)

    # LoRA parameters
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora-rank", type=int, default=None)
    lora_group.add_argument("--lora-alpha", type=float, default=None)
    lora_group.add_argument("--lora-dropout", type=float, default=None)

    # GRPO-specific arguments
    grpo_group = parser.add_argument_group("GRPO")
    grpo_group.add_argument("--group-size", type=int, default=None)
    grpo_group.add_argument("--beta", type=float, default=None, help="KL penalty coefficient")
    grpo_group.add_argument("--epsilon", type=float, default=None)
    grpo_group.add_argument("--epsilon-high", type=float, default=None)
    grpo_group.add_argument("--temperature", type=float, default=None)
    grpo_group.add_argument("--max-completion-length", type=int, default=None)
    grpo_group.add_argument(
        "--generation-sub-batch-size",
        type=int,
        default=1,
        help="Generate this many completions at a time to avoid GPU timeout",
    )
    grpo_group.add_argument("--grpo-loss-type", choices=["grpo", "bnpo", "dr_grpo"], default=None)
    grpo_group.add_argument(
        "--importance-sampling-level", choices=["token", "sequence"], default=None
    )

    # Reward functions
    reward_group = parser.add_argument_group("Rewards")
    reward_group.add_argument("--reward-functions", type=str, default=None)
    reward_group.add_argument("--reward-weights", type=str, default=None)
    reward_group.add_argument("--reward-functions-file", type=str, default=None)

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data", type=str, default=None, help="Dataset path")
    data_group.add_argument("--cache-dataset", action="store_true", default=None)
    data_group.add_argument("--no-cache-dataset", action="store_true", default=False)
    data_group.add_argument("--cache-dir", type=str, default=None)
    data_group.add_argument("--force-reload", action="store_true", default=None)
    data_group.add_argument("--shuffle-data", action="store_true", default=None)
    data_group.add_argument("--no-shuffle-data", action="store_true", default=False)
    data_group.add_argument("--balanced-shuffle", action="store_true", default=None)
    data_group.add_argument("--no-balanced-shuffle", action="store_true", default=False)
    data_group.add_argument("--shuffle-seed", type=int, default=None)
    data_group.add_argument("--require-think-tags", action="store_true", default=None)
    data_group.add_argument("--no-require-think-tags", action="store_true", default=False)

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--val-batches", type=int, default=None)
    eval_group.add_argument("--test-batches", type=int, default=None)
    eval_group.add_argument("--steps-per-report", type=int, default=None)
    eval_group.add_argument("--steps-per-eval", type=int, default=None)

    # Checkpoint arguments
    ckpt_group = parser.add_argument_group("Checkpoints")
    ckpt_group.add_argument("--adapter-path", type=str, default=None)
    ckpt_group.add_argument("--resume-adapter-file", type=str, default=None)
    ckpt_group.add_argument("--save-every", type=int, default=None)
    ckpt_group.add_argument("--keep-last-n-checkpoints", type=int, default=None)
    ckpt_group.add_argument("--keep-best-n-checkpoints", type=int, default=None)
    ckpt_group.add_argument("--checkpoint-metric", type=str, default=None)
    ckpt_group.add_argument("--resume", action="store_true", default=False)
    ckpt_group.add_argument("--resume-iteration", type=int, default=None)

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--wandb", type=str, default=None, help="WandB project name")
    log_group.add_argument("--log-rollouts", action="store_true", default=None)
    log_group.add_argument("--no-log-rollouts", action="store_true", default=False)
    log_group.add_argument("--log-rollouts-every-n-steps", type=int, default=None)
    log_group.add_argument("--log-rollouts-to-wandb", action="store_true", default=None)
    log_group.add_argument("--no-log-rollouts-to-wandb", action="store_true", default=False)

    # Monitor arguments
    monitor_group = parser.add_argument_group("Monitor")
    monitor_group.add_argument("--enable-monitor", action="store_true", default=None)
    monitor_group.add_argument("--no-monitor", action="store_true", default=False)
    monitor_group.add_argument("--monitor-kl-warning", type=float, default=None)
    monitor_group.add_argument("--monitor-kl-critical", type=float, default=None)

    # Two-phase generation
    twophase_group = parser.add_argument_group("Two-Phase Generation")
    twophase_group.add_argument("--enforce-thinking", action="store_true", default=None)
    twophase_group.add_argument("--think-start-token", type=str, default=None)
    twophase_group.add_argument("--think-end-token", type=str, default=None)
    twophase_group.add_argument("--continuation-tokens", type=int, default=64)
    twophase_group.add_argument(
        "--two-phase-samples-per-group",
        type=int,
        default=2,
        help="Max samples per group to apply two-phase recovery. -1 = all (default).",
    )
    twophase_group.add_argument(
        "--exam-phase-recovery-ratio",
        type=float,
        default=None,
        help="Ratio of incomplete exam samples (missing </think>) that get phase 2 recovery. "
        "Completion is truncated, '... </think>\\n\\boxed{' is injected, and model "
        "completes the boxed answer. Injected tokens are masked from loss. "
        "Default: 0.5 (50%% of group).",
    )

    # Curriculum scaffolding
    curriculum_group = parser.add_argument_group("Curriculum")
    curriculum_group.add_argument("--curriculum-enabled", action="store_true", default=None)
    curriculum_group.add_argument("--curriculum-start-ratio", type=float, default=None)
    curriculum_group.add_argument("--curriculum-end-ratio", type=float, default=None)
    curriculum_group.add_argument("--curriculum-warmup-iters", type=int, default=None)
    curriculum_group.add_argument("--curriculum-taper-iters", type=int, default=None)

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--fuse", action="store_true", default=None)
    output_group.add_argument("--no-fuse", action="store_true", default=False)

    # Config file
    parser.add_argument("-c", "--config", type=str, default=None, help="YAML config file")

    return parser


# =============================================================================
# Training Functions
# =============================================================================


def _model_is_quantized(model: nn.Module) -> bool:
    """Check if any model layer uses quantized weights."""
    if hasattr(model, "args") and getattr(model.args, "quantization", None):
        return True
    return any(
        isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        for _, m in model.named_modules()
    )


def train_model(
    args: argparse.Namespace,
    model: nn.Module,
    tokenizer: Any,
    reference_model: nn.Module | None,
    train_set: CacheDataset,
    valid_set: CacheDataset,
    training_callback: TrainingCallback | None = None,
) -> None:
    """Run GRPO training."""
    mx.random.seed(args.seed)

    if args.iters is None and args.epochs is not None:
        args.iters = calculate_iters(train_set, args.batch_size, args.epochs)

    # Setup model
    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers but model has {len(model.layers)}"
        )

    if args.train_type == "dora" and _model_is_quantized(model):
        if getattr(args, "force_dora", False):
            print_warning(
                "DoRA with quantized models dequantizes all weights every forward pass, "
                "negating quantization memory savings and reducing speed. "
                "Proceeding anyway due to --force-dora."
            )
        else:
            print_warning(
                "DoRA with quantized models dequantizes all weights every forward pass, "
                "negating quantization memory savings and reducing speed. "
                "Falling back to LoRA. Use --force-dora to override."
            )
            args.train_type = "lora"

    if args.train_type == "full":
        for layer in model.layers[-max(args.num_layers, 0) :]:
            layer.unfreeze()
    elif args.train_type in ["lora", "dora"]:
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.train_type == "dora"),
        )
    else:
        raise ValueError(f"Unknown train-type: {args.train_type}")

    if args.resume_adapter_file:
        print_info(f"Loading adapter weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    # Setup paths
    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    # Setup optimizer
    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    optimizer_config = args.optimizer_config.get(args.optimizer.lower(), {})
    opt_class = {"adam": optim.Adam, "adamw": optim.AdamW, "muon": optim.Muon}[
        args.optimizer.lower()
    ]
    optimizer = opt_class(learning_rate=lr, **optimizer_config)

    # Create V2 type coordinator for type-aware rewards
    type_coordinator = create_v2_coordinator(tokenizer)

    # Register backward-compatible type_aware_strict / type_aware_reward aliases
    # so existing configs (e.g. train.sh) continue to work.
    _v2_adapter = v2_reward_adapter(type_coordinator)
    for _alias in ("type_aware_strict", "type_aware_reward"):
        REWARD_REGISTRY[_alias] = _v2_adapter

    # Setup reward functions
    if args.reward_functions_file:
        load_reward_functions_from_file(args.reward_functions_file)

    reward_funcs = get_default_reward_functions()
    if args.reward_functions:
        func_names = [name.strip() for name in args.reward_functions.split(",")]
        try:
            reward_funcs = [get_reward_function(name) for name in func_names]
            print_success(f"Using reward functions: {', '.join(func_names)}")
        except KeyError as e:
            print_error(f"Error: {e}")
            print_info(f"Available: {list_available_reward_functions()}")
            return

    # Build GRPO training args
    grpo_args = GRPOTrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
        cross_sample_max_completion_length=args.cross_sample_max_completion_length,
        grad_checkpoint=args.grad_checkpoint,
        beta=args.beta,
        group_size=args.group_size,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        reference_model_path=args.reference_model_path,
        temperature=args.temperature,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        reward_weights=(
            [float(x) for x in args.reward_weights.strip("[]").split(",")]
            if args.reward_weights
            else None
        ),
        importance_sampling_level=args.importance_sampling_level,
        grpo_loss_type=args.grpo_loss_type,
        # Dataset options
        cache_dataset=args.cache_dataset,
        cache_dir=args.cache_dir,
        force_reload=args.force_reload,
        cross_sampling_enabled=args.cross_sampling_enabled,
        cross_sampling_ratio=args.cross_sampling_ratio,
        cross_sampling_max_history_tokens=args.cross_sampling_max_history_tokens,
        cross_sampling_seed=args.cross_sampling_seed,
        cross_sampling_truncation_marker=args.cross_sampling_truncation_marker,
        shuffle_data=args.shuffle_data,
        balanced_shuffle=args.balanced_shuffle,
        shuffle_seed=args.shuffle_seed,
        require_think_tags=args.require_think_tags,
        # Crash recovery
        auto_resume_on_crash=args.auto_resume_on_crash,
        max_crash_retries=args.max_crash_retries,
        crash_cooldown_seconds=args.crash_cooldown_seconds,
        # Rollout logging
        log_rollouts=args.log_rollouts,
        log_rollouts_every_n_steps=args.log_rollouts_every_n_steps,
        log_rollouts_to_wandb=args.log_rollouts_to_wandb,
        rollout_log_file=args.rollout_log_file,
        # Checkpoint management
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        keep_best_n_checkpoints=args.keep_best_n_checkpoints,
        checkpoint_metric=args.checkpoint_metric,
        checkpoint_metric_higher_is_better=args.checkpoint_metric_higher_is_better,
        # Training monitor
        enable_monitor=args.enable_monitor,
        monitor_kl_warning=args.monitor_kl_warning,
        monitor_kl_critical=args.monitor_kl_critical,
        monitor_reward_warning=args.monitor_reward_warning,
        monitor_reward_critical=args.monitor_reward_critical,
        monitor_stop_on_critical=args.monitor_stop_on_critical,
        monitor_critical_count=args.monitor_critical_count,
        # Resume training
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_iteration=args.resume_iteration,
        # Selective layer training
        train_layers=args.train_layers,
        thinking_layers=args.thinking_layers,
        answer_layers=args.answer_layers,
        thinking_gradient_weight=args.thinking_gradient_weight,
        answer_gradient_weight=args.answer_gradient_weight,
        # Two-phase generation
        enforce_thinking=args.enforce_thinking,
        think_start_token=args.think_start_token,
        think_end_token=args.think_end_token,
        continuation_tokens=args.continuation_tokens,
        continuation_force_answer_ratio=args.continuation_force_answer_ratio,
        two_phase_samples_per_group=args.two_phase_samples_per_group,
        exam_phase_recovery_ratio=args.exam_phase_recovery_ratio,
        # Smart truncation
        smart_truncation_enabled=args.smart_truncation_enabled,
        max_extreme_tokens=args.max_extreme_tokens,
        truncation_brevity_marker=args.truncation_brevity_marker,
        truncation_keep_start_ratio=args.truncation_keep_start_ratio,
        truncation_keep_end_ratio=args.truncation_keep_end_ratio,
        # SFT anchor
        sft_anchor_enabled=args.sft_anchor_enabled,
        sft_anchor_layers=args.sft_anchor_layers,
        sft_anchor_lr_multiplier=args.sft_anchor_lr_multiplier,
        gradient_alignment_mode=args.gradient_alignment_mode,
        gradient_alignment_weight=args.gradient_alignment_weight,
        # Curriculum scaffolding
        curriculum_enabled=args.curriculum_enabled,
        curriculum_start_ratio=args.curriculum_start_ratio,
        curriculum_end_ratio=args.curriculum_end_ratio,
        curriculum_warmup_iters=args.curriculum_warmup_iters,
        curriculum_taper_iters=args.curriculum_taper_iters,
        curriculum_by_lines=args.curriculum_by_lines,
        curriculum_truncation_mode=args.curriculum_truncation_mode,
        curriculum_preserve_intuition=args.curriculum_preserve_intuition,
        # Multi-curriculum rollout
        multi_curriculum_rollout=args.multi_curriculum_rollout,
        curriculum_scaffold_levels=args.curriculum_scaffold_levels,
        # Scaffold-aware rewards
        scaffold_penalty_weight=args.scaffold_penalty_weight,
        scaffold_penalty_mode=args.scaffold_penalty_mode,
        samples_per_scaffold=args.samples_per_scaffold,
    )

    print_info("Starting GRPO training")
    train_grpo(
        model=model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=valid_set,
        reward_funcs=reward_funcs,
        args=grpo_args,
        training_callback=training_callback,
        type_coordinator=type_coordinator,
    )


def evaluate_model(
    args: argparse.Namespace,
    model: nn.Module,
    tokenizer: Any,
    reference_model: nn.Module | None,
    test_set: CacheDataset,
) -> None:
    """Evaluate GRPO model on test set."""
    print_section("Evaluating GRPO Model")

    test_loss, test_metrics = evaluate_grpo(
        model=model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        dataset=test_set,
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
        reward_funcs=get_default_reward_functions(),
    )

    test_ppl = math.exp(test_loss)
    print(
        f"{Colors.BOLD}Test Results:{Colors.RESET}\n"
        f"  {Colors.YELLOW}Loss:{Colors.RESET} {test_loss:.3f}\n"
        f"  {Colors.YELLOW}Perplexity:{Colors.RESET} {test_ppl:.3f}"
    )

    if test_metrics:
        print(f"\n{Colors.CYAN}GRPO Test Metrics:{Colors.RESET}")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {Colors.WHITE}{metric_name}:{Colors.RESET} {float(metric_value):.3f}")


# =============================================================================
# Main Entry Points
# =============================================================================


def run(args: argparse.Namespace, training_callback: TrainingCallback | None = None) -> None:
    """Main run function."""
    np.random.seed(args.seed)

    if args.wandb is not None:
        training_callback = WandBCallback(
            project_name=args.wandb,
            log_dir=args.adapter_path,
            config=vars(args),
            wrapped_callback=training_callback,
        )

    # Setup quantization
    quantization_config = None
    if args.load_in_4bits:
        quantization_config = {"bits": 4, "group_size": 64}
    elif args.load_in_6bits:
        quantization_config = {"bits": 6, "group_size": 64}
    elif args.load_in_8bits:
        quantization_config = {"bits": 8, "group_size": 64}

    # Load model
    print_info(f"Loading model: {Colors.CYAN}{args.model}{Colors.RESET}")
    model, tokenizer = from_pretrained(model=args.model, quantized_load=quantization_config)

    # Load reference model for GRPO
    reference_model = load_reference_model(args) if args.train else None

    # Load datasets
    print_info("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    # Run training or testing
    if args.test and not args.train:
        if args.adapter_path:
            load_adapters(model, args.adapter_path)
    elif args.train:
        print_section("Training")
        train_model(
            args,
            model,
            tokenizer,
            reference_model,
            CacheDataset(train_set),
            CacheDataset(valid_set),
            training_callback,
        )
    else:
        raise ValueError("Must provide at least one of --train or --test")

    if args.test:
        print_section("Testing")
        evaluate_model(
            args,
            model,
            tokenizer,
            reference_model,
            CacheDataset(test_set),
        )

    # Cleanup
    mx.clear_cache()
    del reference_model

    # Fuse model if requested
    if args.fuse and args.train:
        print_section("Fusing Model")
        fuse_and_save_model(
            model=model,
            tokenizer=tokenizer,
            save_path=args.adapter_path,
        )
        print_success(f"Model fused and saved to {Colors.CYAN}{args.adapter_path}{Colors.RESET}")


def main(args: dict | argparse.Namespace | None = None) -> None:
    """Main entry point for GRPO training."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    print_banner()

    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    elif isinstance(args, dict):
        default_args = vars(build_parser().parse_args([]))
        default_args.update(args)
        args = types.SimpleNamespace(**default_args)

    # Load config file
    if args.config:
        with open(args.config, "r") as f:
            config_args = yaml.load(f, Loader=yaml_loader)
        for k, v in config_args.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)

    # Apply defaults
    for k, v in CONFIG_DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    # Handle LoRA parameters from CLI
    if any(
        [
            getattr(args, "lora_rank", None),
            getattr(args, "lora_alpha", None),
            getattr(args, "lora_dropout", None),
        ]
    ):
        lora_params = args.lora_parameters.copy() if hasattr(args, "lora_parameters") else {}
        if args.lora_rank is not None:
            lora_params["rank"] = args.lora_rank
        if args.lora_alpha is not None:
            lora_params["scale"] = args.lora_alpha
        if args.lora_dropout is not None:
            lora_params["dropout"] = args.lora_dropout
        args.lora_parameters = lora_params

    # Handle boolean toggles
    if getattr(args, "no_cache_dataset", False):
        args.cache_dataset = False
    if getattr(args, "no_log_rollouts", False):
        args.log_rollouts = False
    if getattr(args, "no_log_rollouts_to_wandb", False):
        args.log_rollouts_to_wandb = False
    if getattr(args, "no_monitor", False):
        args.enable_monitor = False
    if getattr(args, "resume", False):
        args.resume_from_checkpoint = True
    if getattr(args, "no_shuffle_data", False):
        args.shuffle_data = False
    if getattr(args, "no_balanced_shuffle", False):
        args.balanced_shuffle = False
    if getattr(args, "no_require_think_tags", False):
        args.require_think_tags = False
    if getattr(args, "no_fuse", False):
        args.fuse = False

    # Print configuration
    print_section("Configuration Summary")
    print(f"{Colors.WHITE}Model:{Colors.RESET} {args.model}")
    print(f"{Colors.WHITE}Training Type:{Colors.RESET} {args.train_type}")
    print(f"{Colors.WHITE}Batch Size:{Colors.RESET} {args.batch_size}")
    print(f"{Colors.WHITE}Learning Rate:{Colors.RESET} {args.learning_rate}")
    print(f"{Colors.WHITE}Optimizer:{Colors.RESET} {args.optimizer}")
    if args.load_in_4bits:
        print(f"{Colors.WHITE}Quantization:{Colors.RESET} 4-bit")
    elif args.load_in_6bits:
        print(f"{Colors.WHITE}Quantization:{Colors.RESET} 6-bit")
    elif args.load_in_8bits:
        print(f"{Colors.WHITE}Quantization:{Colors.RESET} 8-bit")

    run(args)


if __name__ == "__main__":
    main()
