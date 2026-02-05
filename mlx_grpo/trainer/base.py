"""Base training utilities and protocols for GRPO trainer.

This module provides:
- Protocol classes for type-safe extensibility (SOLID: Interface Segregation, Dependency Inversion)
- Base configuration dataclass (SOLID: Single Responsibility)
- Core utilities like gradient checkpointing

Following SOLID principles:
- Single Responsibility: This module handles only base abstractions
- Open/Closed: Protocols allow extension without modification
- Interface Segregation: Small, focused Protocol classes
- Dependency Inversion: Higher-level modules depend on these abstractions
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "BaseTrainingArgs",
    "RewardFunction",
    "TrainingCallback",
    "CheckpointCallback",
    "grad_checkpoint",
    "ModelT",
    "T",
]

# Type variables for generics
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=nn.Module)


# =============================================================================
# Protocol Classes (Structural Subtyping)
# =============================================================================


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions.

    Enables type checking without requiring inheritance.
    Any callable with this signature satisfies the protocol.

    Example:
        @reward("my_reward")
        def custom_reward(
            prompts: list[str],
            completions: list[str],
            answers: list[str],
            types: list[str] | None = None
        ) -> list[float]:
            return [1.0 for _ in completions]
    """

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: list[str] | None = None,
    ) -> list[float]:
        """Compute rewards for completions.

        Args:
            prompts: Input prompts
            completions: Model completions to score
            answers: Ground truth answers
            types: Optional sample types for type-aware rewards

        Returns:
            List of reward scores, one per completion
        """
        ...


@runtime_checkable
class TrainingCallback(Protocol):
    """Protocol for training callbacks.

    Callbacks receive training metrics at regular intervals.
    """

    def on_train_loss_report(self, info: dict[str, Any]) -> None:
        """Called after each training loss report.

        Args:
            info: Dictionary containing:
                - iteration: Current iteration number
                - train_loss: Training loss value
                - learning_rate: Current learning rate
                - tokens_per_second: Training throughput
                - peak_memory: Peak GPU memory in GB
        """
        ...

    def on_val_loss_report(self, info: dict[str, Any]) -> None:
        """Called after each validation loss report.

        Args:
            info: Dictionary containing:
                - iteration: Current iteration number
                - val_loss: Validation loss value
                - val_time: Time taken for validation
        """
        ...


@runtime_checkable
class CheckpointCallback(Protocol):
    """Protocol for checkpoint management callbacks."""

    def on_checkpoint_save(self, path: str, iteration: int, metrics: dict[str, float]) -> None:
        """Called when a checkpoint is saved."""
        ...

    def on_checkpoint_cleanup(self, removed_paths: list[str]) -> None:
        """Called when old checkpoints are cleaned up."""
        ...


# =============================================================================
# Base Configuration
# =============================================================================


@dataclass
class BaseTrainingArgs:
    """Base training arguments - foundation for GRPOTrainingArgs.

    This dataclass provides common training configuration that can be
    extended by specific trainers. Uses dataclass for:
    - Automatic __init__, __repr__, __eq__
    - Field defaults and metadata
    - Easy serialization
    """

    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of gradient accumulation steps."}
    )
    val_batches: int = field(
        default=25,
        metadata={"help": "Number of validation batches, -1 uses entire validation set."},
    )
    steps_per_report: int = field(
        default=10, metadata={"help": "Number of training steps between loss reporting."}
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(default=100, metadata={"help": "Save the model every N steps."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False, metadata={"help": "Use gradient checkpointing to reduce memory use."}
    )


# =============================================================================
# Utility Functions
# =============================================================================


def grad_checkpoint(layer: nn.Module) -> None:
    """Enable gradient checkpointing for a layer type.

    Gradient checkpointing trades compute for memory by not storing
    intermediate activations during forward pass. Instead, activations
    are recomputed during backward pass.

    This function modifies the __call__ method of the layer's type,
    so all instances of that type will use checkpointing.

    Args:
        layer: An instance of the layer type to enable checkpointing for.
               All instances of type(layer) will be affected.

    Example:
        # Enable checkpointing for all transformer layers
        grad_checkpoint(model.layers[0])
    """
    fn = type(layer).__call__

    def checkpointed_fn(model: nn.Module, *args: Any, **kwargs: Any) -> Any:
        def inner_fn(params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


# =============================================================================
# Type Aliases (for cleaner type hints)
# =============================================================================

# Reward function type alias
RewardFunc = Callable[[list[str], list[str], list[str], list[str] | None], list[float]]

# Loss function type alias
LossFunc = Callable[[nn.Module, mx.array, mx.array], tuple[mx.array, mx.array]]
