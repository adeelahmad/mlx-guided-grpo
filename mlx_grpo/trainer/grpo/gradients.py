"""Gradient manipulation utilities for GRPO training.

This module provides gradient-related functions:
- Gradient projection toward SFT direction
- Re-exports from curriculum.py for convenience (compute_gradient_alignment, interpolate_gradients)

SOLID Principles:
- Single Responsibility: Only handles gradient manipulation
- Open/Closed: Can be extended with new gradient strategies
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    pass

# Re-export gradient functions from curriculum.py for convenience
from .curriculum import compute_gradient_alignment, interpolate_gradients

# Re-export SFT anchor loss from loss.py
from .loss import compute_sft_anchor_loss

__all__ = [
    "compute_gradient_alignment",
    "interpolate_gradients",
    "project_gradient_toward_sft",
    "compute_sft_anchor_loss",
]


def project_gradient_toward_sft(
    grpo_grad_flat: dict[str, mx.array],
    sft_grad_flat: dict[str, mx.array],
    strength: float,
) -> dict[str, mx.array]:
    """Project GRPO gradient toward SFT gradient direction.

    Adds a component in the SFT direction to the GRPO gradient.
    result = grpo_grad + strength * (sft_grad normalized)

    This helps guide exploration toward format compliance while still
    allowing GRPO to optimize for reward.

    Args:
        grpo_grad_flat: Flattened GRPO gradients
        sft_grad_flat: Flattened SFT gradients
        strength: How much to pull toward SFT direction

    Returns:
        Modified gradient dict with SFT direction component added
    """
    # Get SFT direction (normalized)
    sft_vec: list[mx.array] = []
    keys_order: list[str] = []

    for key in sft_grad_flat:
        if key in grpo_grad_flat:
            sft_vec.append(sft_grad_flat[key].flatten())
            keys_order.append(key)

    if not sft_vec:
        return grpo_grad_flat

    sft_flat = mx.concatenate(sft_vec)
    sft_norm = mx.sqrt(mx.sum(sft_flat ** 2)) + 1e-8
    sft_direction = sft_flat / sft_norm

    # Add SFT direction component to each GRPO gradient
    result = dict(grpo_grad_flat)
    offset = 0

    for key in keys_order:
        shape = sft_grad_flat[key].shape
        size = sft_grad_flat[key].size
        sft_component = sft_direction[offset:offset + size].reshape(shape)
        result[key] = grpo_grad_flat[key] + strength * sft_component * float(sft_norm)
        offset += size

    return result
