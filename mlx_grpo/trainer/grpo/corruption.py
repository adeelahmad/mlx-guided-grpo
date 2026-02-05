"""Corruption detection and guardrails for GRPO training.

This module provides validation to detect:
- NaN/Inf values in adapter weights (prevents saving corrupt models)
- Corrupted/garbage completions (detects generation failures)

Exit Codes:
- EXIT_CODE_CORRUPTION (42): Fatal corruption detected, immediate exit required

SOLID Principles:
- Single Responsibility: Only handles corruption detection
- Open/Closed: New validation checks can be added without modifying existing ones
"""
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import mlx.core as mx
from tqdm import tqdm

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

__all__ = [
    "EXIT_CODE_CORRUPTION",
    "CorruptionError",
    "AdapterCorruptionError",
    "CompletionCorruptionError",
    "validate_adapter_weights",
    "validate_completions",
    "validate_and_exit_on_corruption",
]

# Exit code for corruption - distinguishes from OOM (137) and general errors (1)
EXIT_CODE_CORRUPTION = 42


class CorruptionError(Exception):
    """Base exception for corruption detection."""

    exit_code = EXIT_CODE_CORRUPTION


class AdapterCorruptionError(CorruptionError):
    """Raised when adapter weights contain invalid values (NaN/Inf)."""

    def __init__(self, message: str, corrupt_keys: List[str]):
        super().__init__(message)
        self.corrupt_keys = corrupt_keys


class CompletionCorruptionError(CorruptionError):
    """Raised when generated completions are corrupted/garbage."""

    def __init__(self, message: str, corrupt_indices: List[int]):
        super().__init__(message)
        self.corrupt_indices = corrupt_indices


def validate_adapter_weights(
    weights: Dict[str, mx.array],
    raise_on_error: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate adapter weights for NaN/Inf values.

    This should be called BEFORE saving any checkpoint to prevent
    persisting a corrupted model that cannot be recovered.

    Args:
        weights: Dictionary of adapter weights to validate
        raise_on_error: If True, raise AdapterCorruptionError on detection

    Returns:
        Tuple of (is_valid, list_of_corrupt_keys)

    Raises:
        AdapterCorruptionError: If corruption detected and raise_on_error=True

    Example:
        >>> weights = dict(tree_flatten(model.trainable_parameters()))
        >>> is_valid, corrupt = validate_adapter_weights(weights)
        >>> if not is_valid:
        ...     # Handle corruption - DO NOT SAVE
    """
    corrupt_keys: List[str] = []

    for key, value in weights.items():
        if not isinstance(value, mx.array):
            continue

        # Force evaluation to detect lazy computation issues
        mx.eval(value)

        # Check for NaN
        has_nan = mx.any(mx.isnan(value)).item()
        if has_nan:
            corrupt_keys.append(f"{key} (contains NaN)")
            continue

        # Check for Inf
        has_inf = mx.any(mx.isinf(value)).item()
        if has_inf:
            corrupt_keys.append(f"{key} (contains Inf)")
            continue

        # Check for extreme values that indicate numerical instability
        max_abs = mx.max(mx.abs(value)).item()
        if max_abs > 1e6:  # Unusually large values for LoRA weights
            corrupt_keys.append(f"{key} (extreme value: {max_abs:.2e})")

    is_valid = len(corrupt_keys) == 0

    if not is_valid and raise_on_error:
        msg = (
            f"CORRUPTION DETECTED: {len(corrupt_keys)} adapter weight(s) contain invalid values:\n"
            + "\n".join(f"  - {k}" for k in corrupt_keys[:10])
        )
        if len(corrupt_keys) > 10:
            msg += f"\n  ... and {len(corrupt_keys) - 10} more"
        raise AdapterCorruptionError(msg, corrupt_keys)

    return is_valid, corrupt_keys


def validate_completions(
    completions: List[str],
    min_length: int = 5,
    max_repetition_ratio: float = 0.8,
    raise_on_error: bool = False,
) -> Tuple[bool, List[int], List[str]]:
    """Validate generated completions for corruption indicators.

    Detects:
    - Empty or extremely short completions
    - Excessive character repetition (e.g., "AAAAAAA...")
    - Control character spam
    - Pure special token sequences
    - Gibberish patterns

    Args:
        completions: List of generated completion strings
        min_length: Minimum acceptable completion length
        max_repetition_ratio: Maximum ratio of repeated characters
        raise_on_error: If True, raise CompletionCorruptionError

    Returns:
        Tuple of (is_valid, corrupt_indices, reasons)

    Example:
        >>> completions = ["<think>This is valid</think>", "AAAAAAAAAA"]
        >>> is_valid, bad_idx, reasons = validate_completions(completions)
    """
    corrupt_indices: List[int] = []
    reasons: List[str] = []

    # Patterns indicating corruption
    control_char_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]{3,}')
    repeated_char_pattern = re.compile(r'(.)\1{20,}')  # Same char 20+ times
    repeated_token_pattern = re.compile(r'(\S{2,})\s*\1{10,}')  # Same token 10+ times
    special_token_only = re.compile(r'^[\s<>/\[\]{}|\\]+$')  # Only special chars

    for i, completion in enumerate(completions):
        issues: List[str] = []

        # Check minimum length
        stripped = completion.strip()
        if len(stripped) < min_length:
            issues.append(f"too short ({len(stripped)} chars)")

        # Check for control character spam
        if control_char_pattern.search(completion):
            issues.append("control character spam")

        # Check for single character repetition
        if repeated_char_pattern.search(completion):
            issues.append("excessive character repetition")

        # Check for token repetition
        if repeated_token_pattern.search(completion):
            issues.append("excessive token repetition")

        # Check character diversity
        if len(stripped) > 10:
            unique_chars = len(set(stripped))
            char_ratio = unique_chars / len(stripped)
            if char_ratio < (1 - max_repetition_ratio):
                issues.append(f"low character diversity ({char_ratio:.2%})")

        # Check if only special tokens
        if special_token_only.match(stripped) and len(stripped) > 0:
            issues.append("only special tokens")

        # Check for encoding issues (replacement character spam)
        if completion.count('\ufffd') > 5:
            issues.append("encoding errors (replacement chars)")

        if issues:
            corrupt_indices.append(i)
            reasons.append(f"idx {i}: {', '.join(issues)}")

    # Determine if this is critical (majority corrupted)
    corruption_ratio = len(corrupt_indices) / len(completions) if completions else 0
    is_valid = corruption_ratio < 0.5  # Allow up to 50% individual failures

    if not is_valid and raise_on_error:
        msg = (
            f"COMPLETION CORRUPTION: {len(corrupt_indices)}/{len(completions)} "
            f"completions appear corrupted ({corruption_ratio:.1%}):\n"
            + "\n".join(f"  - {r}" for r in reasons[:5])
        )
        if len(reasons) > 5:
            msg += f"\n  ... and {len(reasons) - 5} more"
        raise CompletionCorruptionError(msg, corrupt_indices)

    return is_valid, corrupt_indices, reasons


def validate_and_exit_on_corruption(
    weights: Dict[str, mx.array],
    checkpoint_name: str = "checkpoint",
) -> None:
    """Validate weights and exit with EXIT_CODE_CORRUPTION if corrupted.

    This is a convenience wrapper for use in the training loop
    that handles the exit logic.

    Args:
        weights: Adapter weights to validate
        checkpoint_name: Name for error message context
    """
    try:
        validate_adapter_weights(weights, raise_on_error=True)
    except AdapterCorruptionError as e:
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"FATAL: Adapter corruption detected at {checkpoint_name}")
        tqdm.write(f"{'='*80}")
        tqdm.write(str(e))
        tqdm.write(f"\nExiting with code {EXIT_CODE_CORRUPTION} to prevent saving corrupt model.")
        tqdm.write("DO NOT use --resume with the current adapter file.")
        tqdm.write(f"{'='*80}\n")
        sys.exit(EXIT_CODE_CORRUPTION)


def log_completion_warnings(
    completions: List[str],
    iteration: int,
) -> bool:
    """Log warnings for suspicious completions without failing.

    Returns True if completions appear healthy, False if concerning.
    """
    is_valid, corrupt_indices, reasons = validate_completions(
        completions, raise_on_error=False
    )

    if corrupt_indices:
        corruption_ratio = len(corrupt_indices) / len(completions)
        if corruption_ratio > 0.3:  # More than 30% corrupted
            tqdm.write(
                f"\nWarning [iter {iteration}]: {len(corrupt_indices)}/{len(completions)} "
                f"completions appear corrupted ({corruption_ratio:.1%})"
            )
            for reason in reasons[:3]:
                tqdm.write(f"  - {reason}")
            if len(reasons) > 3:
                tqdm.write(f"  ... and {len(reasons) - 3} more")

    return is_valid
