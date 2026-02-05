"""Layer utilities for GRPO training.

This module provides layer-level control for gradient computation:
- Layer specification parsing
- Gradient masking per layer
- Dual-path gradient combination (thinking vs answer)
- Layer freezing utilities

SOLID Principles:
- Single Responsibility: Only handles layer-related operations
- Open/Closed: Functions can be extended without modification
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "parse_layer_spec",
    "create_layer_gradient_mask",
    "combine_dual_gradients",
    "detect_thinking_answer_positions",
    "freeze_model_layers",
]


def parse_layer_spec(spec: str | None, num_layers: int) -> set[int]:
    """Parse layer specification string into a set of layer indices.

    Args:
        spec: Layer specification, e.g., "0-8,20-28" or "0,1,2,3" or "all"
        num_layers: Total number of layers in the model

    Returns:
        Set of layer indices to include

    Examples:
        >>> parse_layer_spec("0-8,20-28", 32)
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 25, 26, 27, 28}
        >>> parse_layer_spec("all", 32)
        {0, 1, 2, ..., 31}
    """
    if spec is None or spec.lower() == "all":
        return set(range(num_layers))

    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            # Range like "0-8"
            start, end = part.split("-")
            start_idx, end_idx = int(start.strip()), int(end.strip())
            layers.update(range(start_idx, end_idx + 1))
        else:
            # Single layer
            layers.add(int(part))

    # Clamp to valid range
    return {layer for layer in layers if 0 <= layer < num_layers}


def create_layer_gradient_mask(
    grad_tree: dict[str, mx.array],
    layer_set: set[int],
    model_prefix: str = "model.layers.",
) -> dict[str, mx.array]:
    """Create a gradient mask that zeros out gradients for layers not in layer_set.

    Args:
        grad_tree: Flattened gradient tree from tree_flatten
        layer_set: Set of layer indices to keep gradients for
        model_prefix: Prefix for layer keys in the gradient tree

    Returns:
        Masked gradient tree with zeros for excluded layers
    """
    masked_grads: dict[str, mx.array] = {}
    layer_pattern = re.compile(rf'{re.escape(model_prefix)}(\d+)\.')

    for key, grad in grad_tree.items():
        match = layer_pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx in layer_set:
                masked_grads[key] = grad
            else:
                # Zero out gradient for this layer
                masked_grads[key] = mx.zeros_like(grad)
        else:
            # Non-layer parameters (embeddings, lm_head, etc.) - keep gradient
            masked_grads[key] = grad

    return masked_grads


def combine_dual_gradients(
    thinking_grads: dict[str, mx.array],
    answer_grads: dict[str, mx.array],
    thinking_layers: set[int],
    answer_layers: set[int],
    thinking_weight: float = 1.0,
    answer_weight: float = 1.0,
    model_prefix: str = "model.layers.",
) -> dict[str, mx.array]:
    """Combine gradients from thinking and answer tokens with layer-specific masking.

    For layers in thinking_layers only: use thinking gradients
    For layers in answer_layers only: use answer gradients
    For layers in both: weighted sum of both gradients
    For layers in neither: zero gradients (no training)

    Args:
        thinking_grads: Gradients from thinking token loss
        answer_grads: Gradients from answer token loss
        thinking_layers: Set of layers for thinking gradients
        answer_layers: Set of layers for answer gradients
        thinking_weight: Weight multiplier for thinking gradients
        answer_weight: Weight multiplier for answer gradients
        model_prefix: Prefix for layer keys

    Returns:
        Combined gradient tree
    """
    combined: dict[str, mx.array] = {}
    layer_pattern = re.compile(rf'{re.escape(model_prefix)}(\d+)\.')

    for key in thinking_grads.keys():
        t_grad = thinking_grads[key]
        a_grad = answer_grads.get(key, mx.zeros_like(t_grad))

        match = layer_pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            in_thinking = layer_idx in thinking_layers
            in_answer = layer_idx in answer_layers

            if in_thinking and in_answer:
                # Layer trains on both - weighted sum
                combined[key] = thinking_weight * t_grad + answer_weight * a_grad
            elif in_thinking:
                # Thinking only
                combined[key] = thinking_weight * t_grad
            elif in_answer:
                # Answer only
                combined[key] = answer_weight * a_grad
            else:
                # Neither - zero gradient (frozen layer)
                combined[key] = mx.zeros_like(t_grad)
        else:
            # Non-layer parameters - use answer gradients (or could be configurable)
            combined[key] = answer_weight * a_grad

    return combined


def detect_thinking_answer_positions(
    completion_texts: list[str],
    tokenizer: Any,
    think_start: str = "<think>",
    think_end: str = "</think>",
) -> tuple[list[mx.array], list[mx.array]]:
    """Detect which token positions are thinking vs answer tokens.

    Args:
        completion_texts: List of completion text strings
        tokenizer: Tokenizer for encoding
        think_start: Start marker for thinking section
        think_end: End marker for thinking section

    Returns:
        Tuple of (thinking_masks, answer_masks) where each is a list of mx.arrays
        with 1s for positions belonging to that type
    """
    thinking_masks: list[mx.array] = []
    answer_masks: list[mx.array] = []

    for text in completion_texts:
        # Tokenize full text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        seq_len = len(tokens)

        if seq_len == 0:
            thinking_masks.append(mx.zeros((1,)))
            answer_masks.append(mx.zeros((1,)))
            continue

        # Find thinking section boundaries in text
        think_start_pos = text.find(think_start)
        think_end_pos = text.find(think_end)

        if think_start_pos == -1 or think_end_pos == -1:
            # No thinking section - all tokens are answer tokens
            thinking_masks.append(mx.zeros((seq_len,)))
            answer_masks.append(mx.ones((seq_len,)))
            continue

        # Create character-level mask first
        char_is_thinking = [False] * len(text)
        if think_start_pos < think_end_pos:
            for i in range(think_start_pos, think_end_pos + len(think_end)):
                if i < len(text):
                    char_is_thinking[i] = True

        # Convert to token-level mask by checking which tokens fall in thinking section
        # This is approximate - we check if the token's text representation is mostly in thinking section
        thinking_mask: list[float] = []
        answer_mask: list[float] = []

        current_pos = 0
        for token_id in tokens:
            token_text = tokenizer.decode([token_id])
            token_len = len(token_text)

            # Check if majority of token is in thinking section
            if current_pos < len(text):
                end_pos = min(current_pos + max(token_len, 1), len(text))
                thinking_chars = sum(
                    1 for i in range(current_pos, end_pos)
                    if i < len(char_is_thinking) and char_is_thinking[i]
                )
                total_chars = end_pos - current_pos

                is_thinking = thinking_chars > total_chars / 2
                thinking_mask.append(1.0 if is_thinking else 0.0)
                answer_mask.append(0.0 if is_thinking else 1.0)
                current_pos = end_pos
            else:
                # Past end of text - treat as answer
                thinking_mask.append(0.0)
                answer_mask.append(1.0)

        thinking_masks.append(mx.array(thinking_mask))
        answer_masks.append(mx.array(answer_mask))

    return thinking_masks, answer_masks


def freeze_model_layers(
    model: nn.Module,
    layers_to_train: set[int],
    verbose: bool = True,
) -> None:
    """Freeze all layers except those in layers_to_train.

    Args:
        model: The model to modify
        layers_to_train: Set of layer indices to keep trainable
        verbose: Whether to print freezing information
    """
    if not hasattr(model, 'layers'):
        if verbose:
            tqdm.write("Warning: Model doesn't have 'layers' attribute, skipping layer freezing")
        return

    total_layers = len(model.layers)
    frozen_count = 0

    for i, layer in enumerate(model.layers):
        if i not in layers_to_train:
            layer.freeze()
            frozen_count += 1

    if verbose:
        trained_layers = sorted(layers_to_train & set(range(total_layers)))
        tqdm.write(f"Layer Selection: Training {len(trained_layers)}/{total_layers} layers")
        if len(trained_layers) <= 20:
            tqdm.write(f"  Training layers: {trained_layers}")
        else:
            tqdm.write(f"  Training layers: {trained_layers[:10]} ... {trained_layers[-10:]}")
        tqdm.write(f"  Frozen layers: {frozen_count}")
