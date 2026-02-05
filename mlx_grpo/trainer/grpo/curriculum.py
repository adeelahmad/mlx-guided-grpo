"""Curriculum learning utilities for GRPO training.

This module provides curriculum-based thinking scaffolding:
- Gradual removal of target thinking
- Smart truncation preserving key content
- Multi-level hierarchical truncation
- Gradient alignment between SFT and GRPO

SOLID Principles:
- Single Responsibility: Only handles curriculum/truncation logic
- Open/Closed: Can be extended with new truncation strategies
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "compute_curriculum_ratio",
    "extract_thinking_content",
    "truncate_thinking_by_ratio",
    "build_curriculum_prefix",
    "hierarchical_truncate_thinking",
    "smart_truncate_completion",
    "compute_gradient_alignment",
    "interpolate_gradients",
]


def compute_curriculum_ratio(
    iteration: int,
    start_ratio: float,
    end_ratio: float,
    warmup_iters: int,
    taper_iters: int,
) -> float:
    """Compute current thinking retention ratio based on curriculum schedule.

    Schedule:
    - Iter 0 to warmup_iters: stay at start_ratio
    - Iter warmup_iters to warmup_iters + taper_iters: linear taper to end_ratio
    - After taper: stay at end_ratio

    Args:
        iteration: Current training iteration
        start_ratio: Initial ratio (typically 1.0 = full thinking)
        end_ratio: Final ratio (typically 0.0 = no thinking)
        warmup_iters: Iterations to stay at start_ratio
        taper_iters: Iterations to taper from start to end

    Returns:
        Ratio in [end_ratio, start_ratio]

    Example:
        >>> compute_curriculum_ratio(50, 1.0, 0.0, 100, 500)
        1.0  # Still in warmup
        >>> compute_curriculum_ratio(350, 1.0, 0.0, 100, 500)
        0.5  # Halfway through taper
    """
    if iteration < warmup_iters:
        return start_ratio

    taper_progress = iteration - warmup_iters
    if taper_progress >= taper_iters:
        return end_ratio

    # Linear interpolation
    progress = taper_progress / max(taper_iters, 1)
    return start_ratio + progress * (end_ratio - start_ratio)


def extract_thinking_content(
    text: str,
    think_start: str = "<think>",
    think_end: str = "</think>",
) -> tuple[str, str, str]:
    """Extract thinking content from a completion.

    Args:
        text: Full completion text
        think_start: Opening think tag
        think_end: Closing think tag

    Returns:
        Tuple of (pre_thinking, thinking_content, post_thinking):
        - pre_thinking: Content before <think> (usually empty)
        - thinking_content: Content inside <think>...</think> (excluding tags)
        - post_thinking: Content after </think> (the answer)
    """
    start_idx = text.find(think_start)
    end_idx = text.find(think_end)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # No valid thinking block found
        return "", "", text

    pre_thinking = text[:start_idx]
    thinking_content = text[start_idx + len(think_start) : end_idx]
    post_thinking = text[end_idx + len(think_end) :]

    return pre_thinking, thinking_content, post_thinking


def truncate_thinking_by_ratio(
    thinking_content: str,
    ratio: float,
    by_lines: bool = True,
    truncation_mode: str = "prefix",
    preserve_intuition: bool = True,
) -> str:
    """Truncate thinking content to retain only the specified ratio.

    Args:
        thinking_content: The content inside think tags
        ratio: How much to retain (1.0 = all, 0.0 = none)
        by_lines: If True, truncate by lines; if False, by characters
        truncation_mode: "prefix" = keep start only, "middle" = keep start + end
        preserve_intuition: If True, always preserve [ANSWER INTUITION: ...] block

    Returns:
        Truncated thinking content
    """
    if ratio >= 1.0:
        return thinking_content
    if ratio <= 0.0:
        return ""

    # Extract and preserve ANSWER INTUITION block if present
    preserved_prefix = ""
    truncatable_content = thinking_content

    if preserve_intuition:
        # Find [ANSWER INTUITION: ...] pattern
        intuition_patterns = [
            r"\[ANSWER INTUITION:[^\]]*\]",  # [ANSWER INTUITION: xxx]
            r"\[ANSWER INTUITION:[^\n]*",  # [ANSWER INTUITION: xxx (no closing bracket)
        ]

        intuition_end_pos = -1
        for pattern in intuition_patterns:
            match = re.search(pattern, thinking_content, re.IGNORECASE)
            if match:
                # Find position after the intuition block
                end_pos = match.end()
                # Include trailing newline if present
                if end_pos < len(thinking_content) and thinking_content[end_pos] == "\n":
                    end_pos += 1
                intuition_end_pos = max(intuition_end_pos, end_pos)

        if intuition_end_pos > 0:
            # Preserve everything up to and including ANSWER INTUITION
            preserved_prefix = thinking_content[:intuition_end_pos]
            truncatable_content = thinking_content[intuition_end_pos:]

            # If nothing left to truncate, return as-is
            if not truncatable_content.strip():
                return thinking_content

    # Now truncate only the truncatable_content
    if truncation_mode == "middle":
        # Middle truncation: keep start + end, remove middle
        return _truncate_middle(preserved_prefix, truncatable_content, ratio, by_lines)
    else:
        # Default "prefix" mode: keep start, truncate end
        return _truncate_prefix(preserved_prefix, truncatable_content, ratio, by_lines)


def _truncate_middle(
    preserved_prefix: str,
    truncatable_content: str,
    ratio: float,
    by_lines: bool,
) -> str:
    """Truncate middle of content, keeping start and end."""
    if by_lines:
        lines = truncatable_content.split("\n")
        total = len(lines)
        if total <= 2:
            return preserved_prefix + truncatable_content

        keep_total = max(2, int(total * ratio))
        keep_start = keep_total // 2
        keep_end = keep_total - keep_start

        if keep_start + keep_end >= total:
            return preserved_prefix + truncatable_content

        start_lines = lines[:keep_start]
        end_lines = lines[-keep_end:] if keep_end > 0 else []

        truncated_parts = start_lines
        truncated_parts.append("\n...\n")
        if end_lines:
            truncated_parts.extend(end_lines)

        return preserved_prefix + "\n".join(truncated_parts)
    else:
        # By characters
        total = len(truncatable_content)
        if total <= 20:
            return preserved_prefix + truncatable_content

        keep_total = max(10, int(total * ratio))
        keep_start = keep_total // 2
        keep_end = keep_total - keep_start

        if keep_start + keep_end >= total:
            return preserved_prefix + truncatable_content

        start_text = truncatable_content[:keep_start]
        end_text = truncatable_content[-keep_end:] if keep_end > 0 else ""

        return preserved_prefix + start_text + "\n...\n" + end_text


def _truncate_prefix(
    preserved_prefix: str,
    truncatable_content: str,
    ratio: float,
    by_lines: bool,
) -> str:
    """Truncate from end, keeping start (prefix mode)."""
    if by_lines:
        lines = truncatable_content.split("\n")
        keep_count = max(1, int(len(lines) * ratio))
        return preserved_prefix + "\n".join(lines[:keep_count])
    else:
        keep_count = max(1, int(len(truncatable_content) * ratio))
        return preserved_prefix + truncatable_content[:keep_count]


def build_curriculum_prefix(
    target_completion: str,
    ratio: float,
    think_start: str = "<think>",
    think_end: str = "</think>",
    by_lines: bool = True,
    truncation_mode: str = "prefix",
    preserve_intuition: bool = True,
) -> str:
    """Build a prefix for curriculum learning by truncating target thinking.

    IMPORTANT: Scaffolding ONLY provides content BEFORE </think>.
    The model MUST ALWAYS generate:
    - The closing </think> tag
    - The answer (e.g., \\boxed{X})

    This ensures the model learns the complete output format.

    Args:
        target_completion: Full target completion with thinking
        ratio: How much thinking to retain (1.0 = full, 0.0 = none)
        think_start: Opening think tag
        think_end: Closing think tag
        by_lines: Truncate by lines or characters
        truncation_mode: "prefix" or "middle"
        preserve_intuition: Always preserve [ANSWER INTUITION: ...] block

    Returns:
        Prefix string that the model should continue from
    """
    pre, thinking, post = extract_thinking_content(target_completion, think_start, think_end)

    if not thinking:
        # No thinking found, return empty (model generates everything)
        return ""

    if ratio <= 0.0:
        # No thinking prefix - model must generate everything
        return ""

    truncated_thinking = truncate_thinking_by_ratio(
        thinking, ratio, by_lines, truncation_mode, preserve_intuition
    )

    # ALWAYS return partial prefix - model must generate </think> and answer
    return f"{pre}{think_start}{truncated_thinking}"


def hierarchical_truncate_thinking(
    thinking_content: str,
    target_tokens: int,
    tokenizer: Any,
    keep_start_ratio: float = 0.3,
    keep_end_ratio: float = 0.5,
    brevity_marker: str = "[truncated due to brevity]",
) -> tuple[str, bool]:
    """Hierarchically truncate thinking content to fit within target token count.

    Strategy (from coarsest to finest):
    1. First try removing paragraphs from middle
    2. Then try removing lines from middle
    3. Finally remove words from middle

    Always preserves beginning (context setup) and end (conclusion/key insights).

    Args:
        thinking_content: The content inside <think>...</think> tags
        target_tokens: Target number of tokens for thinking section
        tokenizer: Tokenizer for counting tokens
        keep_start_ratio: Ratio of content to keep from start (0.0-1.0)
        keep_end_ratio: Ratio of content to keep from end (0.0-1.0)
        brevity_marker: Marker to insert at truncation point

    Returns:
        Tuple of (truncated_content, was_truncated)
    """
    # Count current tokens
    current_tokens = len(tokenizer.encode(thinking_content))

    if current_tokens <= target_tokens:
        return thinking_content, False

    # Calculate how much to keep
    total_keep_ratio = keep_start_ratio + keep_end_ratio
    if total_keep_ratio > 1.0:
        # Normalize
        keep_start_ratio = keep_start_ratio / total_keep_ratio
        keep_end_ratio = keep_end_ratio / total_keep_ratio

    # Try Level 1: Paragraph-level truncation
    paragraphs = thinking_content.split("\n\n")
    if len(paragraphs) >= 3:
        truncated = _truncate_at_level(
            paragraphs,
            "\n\n",
            target_tokens,
            tokenizer,
            keep_start_ratio,
            keep_end_ratio,
            brevity_marker,
        )
        if truncated:
            return truncated, True

    # Try Level 2: Line-level truncation
    lines = thinking_content.split("\n")
    if len(lines) >= 3:
        truncated = _truncate_at_level(
            lines, "\n", target_tokens, tokenizer, keep_start_ratio, keep_end_ratio, brevity_marker
        )
        if truncated:
            return truncated, True

    # Level 3: Word-level truncation (always works)
    words = thinking_content.split()
    if len(words) >= 3:
        truncated = _truncate_at_level(
            words, " ", target_tokens, tokenizer, keep_start_ratio, keep_end_ratio, brevity_marker
        )
        if truncated:
            return truncated, True

    # Fallback: Just cut at token boundary
    tokens = tokenizer.encode(thinking_content)
    if len(tokens) > target_tokens:
        marker_tokens = len(tokenizer.encode(f"\n{brevity_marker}\n"))
        available = target_tokens - marker_tokens
        if available < 10:
            available = target_tokens
            keep_start = int(available * keep_start_ratio)
            keep_end = available - keep_start
            start_tokens = tokens[:keep_start]
            end_tokens = tokens[-keep_end:] if keep_end > 0 else []
            return tokenizer.decode(start_tokens + end_tokens), True

        keep_start = int(available * keep_start_ratio)
        keep_end = available - keep_start

        start_tokens = tokens[:keep_start]
        end_tokens = tokens[-keep_end:] if keep_end > 0 else []

        start_text = tokenizer.decode(start_tokens)
        end_text = tokenizer.decode(end_tokens) if end_tokens else ""

        return f"{start_text}\n{brevity_marker}\n{end_text}", True

    return thinking_content, False


def _truncate_at_level(
    units: list[str],
    join_str: str,
    target_tokens: int,
    tokenizer: Any,
    keep_start_ratio: float,
    keep_end_ratio: float,
    brevity_marker: str,
) -> str | None:
    """Helper to truncate at a specific level (paragraphs, lines, or words).

    Uses binary search to find optimal number of units to keep.

    Returns:
        Truncated string if successful, None if can't meet target at this level.
    """
    total_units = len(units)
    if total_units < 3:
        return None

    # Binary search for optimal number of units to keep
    min_units = 2  # At least 1 start + 1 end
    max_units = total_units

    best_result: str | None = None
    best_token_count = 0

    while min_units <= max_units:
        mid_units = (min_units + max_units) // 2

        # Calculate start and end counts
        keep_start_count = max(1, int(mid_units * keep_start_ratio))
        keep_end_count = max(1, mid_units - keep_start_count)

        # Ensure we don't exceed total units
        if keep_start_count + keep_end_count > total_units:
            keep_start_count = total_units // 2
            keep_end_count = total_units - keep_start_count

        # Build candidate
        start_units = units[:keep_start_count]
        end_units = units[-keep_end_count:] if keep_end_count > 0 else []

        # Check for overlap
        if keep_start_count + keep_end_count >= total_units:
            # No truncation needed at this level
            candidate = join_str.join(units)
        else:
            candidate = (
                join_str.join(start_units) + f"\n{brevity_marker}\n" + join_str.join(end_units)
            )

        token_count = len(tokenizer.encode(candidate))

        if token_count <= target_tokens:
            if token_count > best_token_count:
                # This is a better fit (closer to target without exceeding)
                best_result = candidate
                best_token_count = token_count
            # Try to fit more
            min_units = mid_units + 1
        else:
            # Need fewer units
            max_units = mid_units - 1

    return best_result


def smart_truncate_completion(
    completion_text: str,
    target_tokens: int,
    tokenizer: Any,
    think_start: str = "<think>",
    think_end: str = "</think>",
    keep_start_ratio: float = 0.3,
    keep_end_ratio: float = 0.5,
    brevity_marker: str = "[truncated due to brevity]",
) -> tuple[str, bool]:
    """Smart truncate a completion by truncating the thinking section's middle.

    Preserves:
    - Content before <think>
    - Beginning of thinking (context setup)
    - End of thinking (conclusion/insights)
    - Content after </think> (answer)

    Args:
        completion_text: Full completion text with <think>...</think>
        target_tokens: Target total token count
        tokenizer: Tokenizer for counting
        think_start: Opening think tag
        think_end: Closing think tag
        keep_start_ratio: Ratio of thinking to keep from start
        keep_end_ratio: Ratio of thinking to keep from end
        brevity_marker: Marker to insert at truncation point

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    # Find thinking section
    start_idx = completion_text.find(think_start)
    end_idx = completion_text.find(think_end)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # No valid thinking section - do simple truncation
        tokens = tokenizer.encode(completion_text)
        if len(tokens) > target_tokens:
            return tokenizer.decode(tokens[:target_tokens]), True
        return completion_text, False

    # Split into parts
    pre_think = completion_text[:start_idx]
    thinking = completion_text[start_idx + len(think_start) : end_idx]
    post_think = completion_text[end_idx:]  # Includes </think> and answer

    # Count non-thinking tokens
    pre_tokens = len(tokenizer.encode(pre_think + think_start))
    post_tokens = len(tokenizer.encode(post_think))

    # Available tokens for thinking
    available_for_thinking = target_tokens - pre_tokens - post_tokens

    if available_for_thinking <= 10:
        # Not enough room - minimal thinking
        return f"{pre_think}{think_start}\n{brevity_marker}\n{post_think}", True

    # Truncate thinking content
    truncated_thinking, was_truncated = hierarchical_truncate_thinking(
        thinking,
        available_for_thinking,
        tokenizer,
        keep_start_ratio,
        keep_end_ratio,
        brevity_marker,
    )

    # Reconstruct
    result = f"{pre_think}{think_start}{truncated_thinking}{post_think}"

    # Final check and adjustment
    result_tokens = len(tokenizer.encode(result))
    if result_tokens > target_tokens:
        # Need more aggressive truncation
        adjustment = result_tokens - target_tokens + 10
        truncated_thinking, _ = hierarchical_truncate_thinking(
            thinking,
            max(10, available_for_thinking - adjustment),
            tokenizer,
            keep_start_ratio,
            keep_end_ratio,
            brevity_marker,
        )
        result = f"{pre_think}{think_start}{truncated_thinking}{post_think}"
        was_truncated = True

    return result, was_truncated


def compute_gradient_alignment(
    sft_grad_flat: dict[str, mx.array],
    grpo_grad_flat: dict[str, mx.array],
) -> dict[str, float]:
    """Compute alignment metrics between SFT and GRPO gradients.

    Args:
        sft_grad_flat: Flattened SFT gradients
        grpo_grad_flat: Flattened GRPO gradients

    Returns:
        Dict with:
        - cosine_similarity: How aligned the gradient directions are
        - sft_norm: L2 norm of SFT gradient
        - grpo_norm: L2 norm of GRPO gradient
        - kl_divergence: Approximate KL between gradient "distributions"
    """
    # Flatten both gradient dicts to vectors
    sft_vec = []
    grpo_vec = []

    for key in sft_grad_flat:
        if key in grpo_grad_flat:
            sft_vec.append(sft_grad_flat[key].flatten())
            grpo_vec.append(grpo_grad_flat[key].flatten())

    if not sft_vec:
        return {"cosine_similarity": 0.0, "sft_norm": 0.0, "grpo_norm": 0.0, "kl_divergence": 0.0}

    sft_flat = mx.concatenate(sft_vec)
    grpo_flat = mx.concatenate(grpo_vec)

    # Compute norms
    sft_norm = mx.sqrt(mx.sum(sft_flat**2))
    grpo_norm = mx.sqrt(mx.sum(grpo_flat**2))

    # Cosine similarity
    dot_product = mx.sum(sft_flat * grpo_flat)
    cosine_sim = dot_product / (sft_norm * grpo_norm + 1e-8)

    # KL divergence approximation: treat normalized abs gradients as distributions
    sft_dist = mx.abs(sft_flat) / (mx.sum(mx.abs(sft_flat)) + 1e-8)
    grpo_dist = mx.abs(grpo_flat) / (mx.sum(mx.abs(grpo_flat)) + 1e-8)
    # KL(grpo || sft) - how much grpo diverges from sft
    kl_div = mx.sum(grpo_dist * mx.log((grpo_dist + 1e-8) / (sft_dist + 1e-8)))

    return {
        "cosine_similarity": float(cosine_sim),
        "sft_norm": float(sft_norm),
        "grpo_norm": float(grpo_norm),
        "kl_divergence": float(kl_div),
    }


def interpolate_gradients(
    sft_grad_flat: dict[str, mx.array],
    grpo_grad_flat: dict[str, mx.array],
    alpha: float,
) -> dict[str, mx.array]:
    """Interpolate between SFT and GRPO gradients.

    result = alpha * sft_grad + (1 - alpha) * grpo_grad

    This pulls GRPO exploration toward SFT format direction.

    Args:
        sft_grad_flat: Flattened SFT gradients
        grpo_grad_flat: Flattened GRPO gradients
        alpha: Interpolation weight (0 = pure GRPO, 1 = pure SFT)

    Returns:
        Interpolated gradient dict
    """
    result = {}
    for key in grpo_grad_flat:
        if key in sft_grad_flat:
            result[key] = alpha * sft_grad_flat[key] + (1 - alpha) * grpo_grad_flat[key]
        else:
            result[key] = grpo_grad_flat[key]
    return result
