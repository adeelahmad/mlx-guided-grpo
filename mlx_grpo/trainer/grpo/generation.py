"""Generation utilities for GRPO training.

This module provides:
- GRPO-specific batch generation with two-phase recovery
- Smart truncation for completions
- Curriculum-aware generation with scaffolding

SOLID Principles:
- Single Responsibility: Only handles text generation
- Open/Closed: Generation strategies can be extended
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import batch_generate
from tqdm import tqdm

from .curriculum import (
    build_curriculum_prefix,
    smart_truncate_completion,
)

if TYPE_CHECKING:
    from typing import Any
    import mlx.nn as nn

__all__ = [
    "generate_grpo",
]


def generate_grpo(
    model: nn.Module,
    tokenizer: Any,
    prompt_tokens: list[Any],
    max_tokens: int,
    group_size: int,
    temperature: float,
    batch_size: int,
    end_token: str,
    enforce_thinking: bool = False,
    think_start: str = "<think>",
    think_end: str = "</think>",
    answer_end: str = "</answer>",
    continuation_tokens: int = 256,
    continuation_force_answer_ratio: float = 1.0,
    curriculum_prefixes: list[str] | None = None,
    target_completions: list[str] | None = None,
    multi_curriculum_rollout: bool = False,
    curriculum_scaffold_levels: list[float] | None = None,
    curriculum_truncation_mode: str = "prefix",
    curriculum_preserve_intuition: bool = True,
    type_info: list[Any] | None = None,
    cross_sample_max_tokens: int | None = None,
    # Smart truncation parameters
    smart_truncation_enabled: bool = False,
    max_extreme_tokens: int = 1024,
    truncation_brevity_marker: str = "[truncated due to brevity]",
    truncation_keep_start_ratio: float = 0.3,
    truncation_keep_end_ratio: float = 0.5,
    generation_sub_batch_size: int = 1,  # Generate this many at a time to avoid GPU timeout
) -> tuple[list[mx.array], list[str], list[int], list[bool], list[float], list[int]]:
    """Generate completions with optional two-phase recovery for incomplete outputs.

    Two-phase generation only triggers when:
    - enforce_thinking=True AND
    - A completion has <think> but missing </think>, OR
    - A completion is missing </answer>

    Smart Truncation Mode (smart_truncation_enabled=True):
    - Phase 1 generates up to max_extreme_tokens (not max_tokens)
    - If model naturally closes </think> and exceeds max_tokens:
      - Truncate thinking middle (preserve start + end)
      - Insert brevity_marker to teach conciseness
      - No Phase 2 needed (cleaner gradients)
    - If model doesn't close </think>:
      - Fall back to original two-phase recovery

    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        prompt_tokens: List of prompt token sequences
        max_tokens: Maximum completion tokens
        group_size: Number of completions per prompt
        temperature: Sampling temperature
        batch_size: Batch size for generation
        end_token: End token to stop generation
        enforce_thinking: Enable two-phase generation for incomplete outputs
        think_start: Start marker for thinking section
        think_end: End marker for thinking section
        answer_end: End marker for answer section
        continuation_tokens: Max tokens for continuation phase
        continuation_force_answer_ratio: Ratio of incomplete completions that force answer
        curriculum_prefixes: Optional prefix strings for curriculum learning
        target_completions: Target completions for scaffolding
        multi_curriculum_rollout: Use different scaffold levels for each completion
        curriculum_scaffold_levels: Custom scaffold ratios
        curriculum_truncation_mode: "prefix" or "middle"
        curriculum_preserve_intuition: Preserve [ANSWER INTUITION: ...] blocks
        type_info: Type info dicts per prompt
        cross_sample_max_tokens: Max tokens for cross-sampled examples
        smart_truncation_enabled: Enable smart truncation mode
        max_extreme_tokens: Max tokens before forcing closure
        truncation_brevity_marker: Marker for truncation
        truncation_keep_start_ratio: Ratio to keep from start
        truncation_keep_end_ratio: Ratio to keep from end

    Returns:
        Tuple of:
        - all_completions: List of completion token arrays
        - all_completion_texts: List of completion strings
        - batch_indices: Prompt index for each completion
        - two_phase_flags: Whether each used two-phase recovery
        - all_scaffold_ratios: Scaffold ratio for each completion
        - all_scaffold_token_counts: Number of scaffold/injected tokens to mask from loss for each completion
    """
    was_training = model.training
    model.eval()
    try:
        all_completions: list[mx.array] = []
        all_completion_texts: list[str] = []
        batch_indices: list[int] = []
        all_scaffold_ratios: list[float] = []
        all_scaffold_token_counts: list[int] = []  # Track tokens to mask from loss

        # Track which outputs need continuation
        incomplete_indices: list[int] = []
        incomplete_prompts: list[Any] = []
        incomplete_prefixes: list[str] = []
        incomplete_targets: list[str | None] = []
        incomplete_injected_counts: list[int] = []  # Injected tokens to mask

        total_samples = len(prompt_tokens)

        use_eos_token = False
        if end_token:
            try:
                tokenizer.add_eos_token(end_token)
                use_eos_token = True
            except ValueError:
                use_eos_token = False

        sampler = make_sampler(
            temperature,
            top_p=0.95,
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=20,
        )

        # Setup multi-curriculum rollout scaffold levels
        if multi_curriculum_rollout and curriculum_scaffold_levels is None:
            if group_size > 1:
                curriculum_scaffold_levels = [1.0 - k / (group_size - 1) for k in range(group_size)]
            else:
                curriculum_scaffold_levels = [0.0]

        effective_cross_max = cross_sample_max_tokens if cross_sample_max_tokens is not None else max_tokens

        # Phase 1: Generate all completions
        for i in range(0, total_samples, batch_size):
            current_batch_size = min(batch_size, total_samples - i)
            batch_prompts = prompt_tokens[i:i + current_batch_size]

            batched_prompts: list[Any] = []
            batched_indices: list[int] = []
            batched_prefix_texts: list[str] = []
            batched_targets: list[str | None] = []
            batched_scaffold_ratios: list[float] = []
            batched_original_prompts: list[Any] = []

            # Determine effective max_tokens for this batch
            batch_has_cross_sampled = False
            for j in range(current_batch_size):
                prompt_idx = i + j
                if type_info is not None and prompt_idx < len(type_info):
                    prompt_type_info = type_info[prompt_idx]
                    if isinstance(prompt_type_info, dict) and prompt_type_info.get("cross_sampled", False):
                        batch_has_cross_sampled = True
                        break

            if smart_truncation_enabled:
                batch_max_tokens = max_extreme_tokens
            else:
                batch_max_tokens = effective_cross_max if batch_has_cross_sampled else max_tokens

            for j, prompt in enumerate(batch_prompts):
                prompt_idx = i + j

                target = None
                if target_completions is not None and prompt_idx < len(target_completions):
                    target = target_completions[prompt_idx]

                for k in range(group_size):
                    prefix_text = ""
                    scaffold_ratio = 0.0

                    # Check if exam-type sample
                    is_exam = False
                    if type_info is not None and prompt_idx < len(type_info):
                        prompt_type_info = type_info[prompt_idx]
                        if isinstance(prompt_type_info, dict):
                            is_exam = prompt_type_info.get("is_exam", False)
                        elif isinstance(prompt_type_info, str):
                            is_exam = "exam" in prompt_type_info.lower()

                    # Skip scaffolding for exam samples
                    if not is_exam:
                        if multi_curriculum_rollout and target:
                            scaffold_ratio = curriculum_scaffold_levels[k % len(curriculum_scaffold_levels)]
                            prefix_text = build_curriculum_prefix(
                                target_completion=target,
                                ratio=scaffold_ratio,
                                think_start=think_start,
                                think_end=think_end,
                                by_lines=True,
                                truncation_mode=curriculum_truncation_mode,
                                preserve_intuition=curriculum_preserve_intuition,
                            )
                        elif curriculum_prefixes is not None and prompt_idx < len(curriculum_prefixes):
                            prefix_text = curriculum_prefixes[prompt_idx]
                            if think_end in prefix_text:
                                scaffold_ratio = 1.0

                    if prefix_text:
                        prefix_tokens = tokenizer.encode(prefix_text)
                        combined_prompt = list(prompt) + prefix_tokens
                        batched_prompts.append(combined_prompt)
                    else:
                        batched_prompts.append(prompt)

                    batched_indices.append(prompt_idx)
                    batched_prefix_texts.append(prefix_text)
                    batched_targets.append(target)
                    batched_scaffold_ratios.append(scaffold_ratio)
                    batched_original_prompts.append(prompt)

            mx.synchronize()
            mx.clear_cache()

            # Sub-batch generation to avoid GPU timeout
            all_result_texts: list[str] = []
            sub_batch_sz = generation_sub_batch_size if generation_sub_batch_size > 0 else len(batched_prompts)

            for sub_start in range(0, len(batched_prompts), sub_batch_sz):
                sub_end = min(sub_start + sub_batch_sz, len(batched_prompts))
                sub_prompts = batched_prompts[sub_start:sub_end]

                sub_results = batch_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=sub_prompts,
                    max_tokens=batch_max_tokens,
                    sampler=sampler,
                    verbose=True,
                )

                all_result_texts.extend(sub_results.texts)
                del sub_results
                mx.synchronize()
                mx.clear_cache()  # Clear after each sub-batch

            for idx, completion_text in enumerate(all_result_texts):
                prefix_text = batched_prefix_texts[idx]
                target = batched_targets[idx]
                scaffold_ratio = batched_scaffold_ratios[idx]
                full_completion_text = prefix_text + completion_text

                # Track scaffold token count (tokens to mask from loss)
                scaffold_token_count = len(tokenizer.encode(prefix_text)) if prefix_text else 0

                completion_ids = tokenizer.encode(full_completion_text)

                # Smart truncation handling
                smart_truncated = False
                if smart_truncation_enabled:
                    has_natural_think_end = think_end in full_completion_text

                    if has_natural_think_end and len(completion_ids) > max_tokens:
                        full_completion_text, smart_truncated = smart_truncate_completion(
                            completion_text=full_completion_text,
                            target_tokens=max_tokens,
                            tokenizer=tokenizer,
                            think_start=think_start,
                            think_end=think_end,
                            keep_start_ratio=truncation_keep_start_ratio,
                            keep_end_ratio=truncation_keep_end_ratio,
                            brevity_marker=truncation_brevity_marker,
                        )
                        completion_ids = tokenizer.encode(full_completion_text)
                        if smart_truncated:
                            tqdm.write(f"    [SMART TRUNCATE] idx={len(all_completions)}: Truncated thinking middle")

                # Truncate to max_tokens
                if len(completion_ids) > max_tokens:
                    completion_ids = completion_ids[:max_tokens]
                    full_completion_text = tokenizer.decode(completion_ids)

                if not use_eos_token and end_token:
                    end_sequence = tokenizer.encode(end_token)
                    if (
                        len(completion_ids) >= len(end_sequence)
                        and completion_ids[-len(end_sequence):] == end_sequence
                    ):
                        completion_ids = completion_ids[:-len(end_sequence)]

                completion_ids_arr = mx.array(completion_ids)
                completion_ids_arr = mx.stop_gradient(completion_ids_arr)
                mx.eval(completion_ids_arr)  # Evaluate immediately to prevent lazy accumulation
                all_completions.append(completion_ids_arr)
                all_completion_texts.append(full_completion_text)
                batch_indices.append(batched_indices[idx])
                all_scaffold_ratios.append(scaffold_ratio)
                all_scaffold_token_counts.append(scaffold_token_count)

                # Check for two-phase recovery
                if enforce_thinking and not smart_truncated:
                    is_incomplete, fixed_prefix, injected_count = _check_incomplete_completion(
                        full_completion_text=full_completion_text,
                        scaffold_ratio=scaffold_ratio,
                        target=target,
                        type_info=type_info,
                        prompt_idx=batched_indices[idx],
                        think_start=think_start,
                        think_end=think_end,
                        answer_end=answer_end,
                        continuation_force_answer_ratio=continuation_force_answer_ratio,
                        tokenizer=tokenizer,
                    )

                    if is_incomplete and fixed_prefix:
                        current_idx = len(all_completions) - 1
                        incomplete_indices.append(current_idx)
                        incomplete_prompts.append(batched_original_prompts[idx])
                        incomplete_prefixes.append(fixed_prefix)
                        incomplete_targets.append(target)
                        incomplete_injected_counts.append(injected_count)

            del all_result_texts
            mx.eval(all_completions[-len(batched_prompts):])
            mx.clear_cache()

        # Phase 2: Batch continuation for incomplete outputs
        if incomplete_indices:
            tqdm.write(f"  Two-phase recovery: {len(incomplete_indices)} incomplete outputs")
            _run_phase2_continuation(
                incomplete_indices=incomplete_indices,
                incomplete_prompts=incomplete_prompts,
                incomplete_prefixes=incomplete_prefixes,
                all_completions=all_completions,
                all_completion_texts=all_completion_texts,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                temperature=temperature,
                continuation_tokens=continuation_tokens,
                max_tokens=max_tokens,
                end_token=end_token,
                use_eos_token=use_eos_token,
            )
            # Update scaffold token counts with injected tokens from two-phase recovery
            for i, global_idx in enumerate(incomplete_indices):
                all_scaffold_token_counts[global_idx] += incomplete_injected_counts[i]
            tqdm.write(f"  Two-phase recovery complete")
            mx.clear_cache()  # Clear after phase 2

        # Build recovery flags
        two_phase_flags = [False] * len(all_completions)
        for idx in incomplete_indices:
            two_phase_flags[idx] = True

        if not all_completions:
            raise ValueError("No valid completions generated.")

        return all_completions, all_completion_texts, batch_indices, two_phase_flags, all_scaffold_ratios, all_scaffold_token_counts
    finally:
        mx.clear_cache()
        if was_training:
            model.train()


def _check_incomplete_completion(
    full_completion_text: str,
    scaffold_ratio: float,
    target: str | None,
    type_info: list[Any] | None,
    prompt_idx: int,
    think_start: str,
    think_end: str,
    answer_end: str,
    continuation_force_answer_ratio: float,
    tokenizer: Any = None,
) -> tuple[bool, str | None, int]:
    """Check if a completion needs two-phase recovery.

    Returns:
        Tuple of (is_incomplete, fixed_prefix, injected_token_count)
        - injected_token_count: number of tokens injected (not generated by model) that should be masked
    """
    import random
    import re

    # Check type info
    is_exam = False
    if type_info is not None and prompt_idx < len(type_info):
        prompt_type_info = type_info[prompt_idx]
        if isinstance(prompt_type_info, dict):
            is_exam = prompt_type_info.get("is_exam", False)
        elif isinstance(prompt_type_info, str):
            is_exam = "exam" in prompt_type_info.lower()

    has_think_start = think_start in full_completion_text
    has_think_end = think_end in full_completion_text
    has_answer_end = answer_end in full_completion_text

    # Exam sample handling
    if is_exam:
        has_boxed_after_think = False
        if has_think_end:
            think_end_pos = full_completion_text.find(think_end)
            after_think = full_completion_text[think_end_pos + len(think_end):]
            has_boxed_after_think = '\\boxed{' in after_think

        if has_think_end and not has_boxed_after_think:
            think_end_pos = full_completion_text.find(think_end)
            injected_text = "\n\n\\boxed{"
            fixed_prefix = full_completion_text[:think_end_pos + len(think_end)] + injected_text
            injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
            return True, fixed_prefix, injected_count

        elif has_think_start and not has_think_end:
            think_start_pos = full_completion_text.find(think_start)
            model_thinking = full_completion_text[think_start_pos + len(think_start):].rstrip()
            prefix_before_think = full_completion_text[:think_start_pos] if think_start_pos > 0 else ""

            # Truncate thinking to leave room for closing tags (reserve ~20 tokens worth)
            # This prevents Phase 2 truncation from cutting off the tags we're adding
            max_thinking_chars = max(100, len(model_thinking) - 80)  # Leave ~80 chars for tags
            if len(model_thinking) > max_thinking_chars:
                model_thinking = model_thinking[:max_thinking_chars].rstrip() + "\n...[truncated]"

            # Injected content: closing tags (the model_thinking is from model, not injected)
            injected_text = f"\n{think_end}\n\n\\boxed{{"
            fixed_prefix = f"{prefix_before_think}{think_start}{model_thinking}{injected_text}"
            injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
            return True, fixed_prefix, injected_count

        elif not has_think_start:
            injected_text = f"{think_start}\n{think_end}\n\n\\boxed{{"
            fixed_prefix = injected_text
            injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
            return True, fixed_prefix, injected_count

        return False, None, 0

    # Non-exam handling
    if has_think_start and not has_think_end and scaffold_ratio < 1.0:
        force_answer = random.random() < continuation_force_answer_ratio

        if force_answer:
            think_start_pos = full_completion_text.find(think_start)
            model_thinking = full_completion_text[think_start_pos + len(think_start):].rstrip()
            prefix_before_think = full_completion_text[:think_start_pos] if think_start_pos > 0 else ""

            # Truncate thinking to leave room for closing tags (reserve ~80 chars for tags)
            max_thinking_chars = max(100, len(model_thinking) - 80)
            if len(model_thinking) > max_thinking_chars:
                model_thinking = model_thinking[:max_thinking_chars].rstrip() + "\n...[truncated]"

            # Extract intuition from target
            bridge_parts = []
            if target:
                target_think_start = target.find(think_start)
                target_think_end = target.find(think_end)

                if target_think_start != -1 and target_think_end != -1:
                    target_thinking = target[target_think_start + len(think_start):target_think_end]
                    intuition_match = re.search(r'\[ANSWER INTUITION:[^\]]*\]', target_thinking)
                    if intuition_match:
                        bridge_parts.append(intuition_match.group(0))

            # Calculate injected content (closing tags + optional bridge from target)
            if model_thinking and bridge_parts:
                bridge = "\n\n...\n\n" + "\n".join(bridge_parts)
                injected_text = f"{bridge}\n{think_end}\n\\boxed{{"
                fixed_prefix = f"{prefix_before_think}{think_start}{model_thinking}{injected_text}"
            elif model_thinking:
                injected_text = f"\n{think_end}\n\\boxed{{"
                fixed_prefix = f"{prefix_before_think}{think_start}{model_thinking}{injected_text}"
            else:
                injected_text = f"\n{think_end}\n\\boxed{{"
                fixed_prefix = f"{prefix_before_think}{think_start}{injected_text}"

            injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
            return True, fixed_prefix, injected_count
        else:
            # Natural continuation - but still close thinking properly
            think_start_pos = full_completion_text.find(think_start)
            model_thinking = full_completion_text[think_start_pos + len(think_start):].rstrip()
            prefix_before_think = full_completion_text[:think_start_pos] if think_start_pos > 0 else ""

            # Truncate thinking if too long, add brevity marker
            if len(model_thinking) > 500:
                truncated_thinking = model_thinking[:400] + "\n\n...[Truncated for brevity]...\n"
            else:
                truncated_thinking = model_thinking

            injected_text = f"\n{think_end}\n\\boxed{{"
            fixed_prefix = f"{prefix_before_think}{think_start}{truncated_thinking}{injected_text}"
            injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
            return True, fixed_prefix, injected_count

    elif has_think_end and not has_answer_end and scaffold_ratio < 1.0:
        think_end_pos = full_completion_text.find(think_end)
        injected_text = "\n" + r"\boxed{"
        fixed_prefix = full_completion_text[:think_end_pos + len(think_end)] + injected_text
        injected_count = len(tokenizer.encode(injected_text)) if tokenizer else 0
        return True, fixed_prefix, injected_count

    return False, None, 0


def _run_phase2_continuation(
    incomplete_indices: list[int],
    incomplete_prompts: list[Any],
    incomplete_prefixes: list[str],
    all_completions: list[mx.array],
    all_completion_texts: list[str],
    model: nn.Module,
    tokenizer: Any,
    batch_size: int,
    temperature: float,
    continuation_tokens: int,
    max_tokens: int,
    end_token: str,
    use_eos_token: bool,
) -> None:
    """Run Phase 2 continuation for incomplete completions."""
    continuation_prompts = []
    for orig_prompt, fixed_prefix in zip(incomplete_prompts, incomplete_prefixes):
        if isinstance(orig_prompt, mx.array):
            prompt_text = tokenizer.decode(orig_prompt.tolist())
        elif isinstance(orig_prompt, list):
            prompt_text = tokenizer.decode(orig_prompt)
        else:
            prompt_text = str(orig_prompt)

        continuation_prompt = prompt_text + fixed_prefix
        continuation_prompts.append(tokenizer.encode(continuation_prompt))

    continuation_sampler = make_sampler(
        temperature,
        top_p=0.95,
        min_p=0.0,
        min_tokens_to_keep=1,
        top_k=20,
    )

    for cont_batch_start in range(0, len(continuation_prompts), batch_size):
        cont_batch_end = min(cont_batch_start + batch_size, len(continuation_prompts))
        cont_batch_prompts = continuation_prompts[cont_batch_start:cont_batch_end]
        cont_batch_indices = incomplete_indices[cont_batch_start:cont_batch_end]
        cont_batch_prefixes = incomplete_prefixes[cont_batch_start:cont_batch_end]

        mx.synchronize()
        mx.clear_cache()

        continuation_results = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=cont_batch_prompts,
            max_tokens=continuation_tokens,
            sampler=continuation_sampler,
            verbose=True,
        )

        mx.synchronize()
        mx.clear_cache()  # Clear generation cache immediately

        for local_idx, cont_text in enumerate(continuation_results.texts):
            global_idx = cont_batch_indices[local_idx]
            fixed_prefix = cont_batch_prefixes[local_idx]

            combined_text = fixed_prefix + cont_text
            completion_ids = tokenizer.encode(combined_text)

            # Smart truncation: preserve the ending (which has </think>\n\boxed{answer})
            # Truncate from the MIDDLE of thinking, not from the end
            if len(completion_ids) > max_tokens:
                # Find where </think> is in the tokens
                think_end_text = "</think>"
                boxed_text = "\\boxed{"

                # Try to find boundary between thinking and answer
                think_end_pos = combined_text.find(think_end_text)
                if think_end_pos != -1:
                    # Preserve everything from </think> onwards
                    end_content = combined_text[think_end_pos:]
                    start_content = combined_text[:think_end_pos]

                    # Calculate how much of start_content we can keep
                    end_tokens = tokenizer.encode(end_content)
                    available_for_start = max_tokens - len(end_tokens) - 10  # Buffer

                    if available_for_start > 50:  # Only truncate if we have reasonable room
                        start_tokens = tokenizer.encode(start_content)
                        if len(start_tokens) > available_for_start:
                            truncated_start = tokenizer.decode(start_tokens[:available_for_start])
                            combined_text = truncated_start + "\n...[truncated]\n" + end_content
                            completion_ids = tokenizer.encode(combined_text)
                    else:
                        # Fall back to simple truncation but keep last 30 tokens
                        keep_end = min(30, len(completion_ids) // 4)
                        keep_start = max_tokens - keep_end
                        completion_ids = completion_ids[:keep_start] + completion_ids[-keep_end:]
                        combined_text = tokenizer.decode(completion_ids)
                else:
                    # No </think> found, use simple truncation
                    completion_ids = completion_ids[:max_tokens]
                    combined_text = tokenizer.decode(completion_ids)

            if not use_eos_token and end_token:
                end_sequence = tokenizer.encode(end_token)
                if (
                    len(completion_ids) >= len(end_sequence)
                    and completion_ids[-len(end_sequence):] == end_sequence
                ):
                    completion_ids = completion_ids[:-len(end_sequence)]

            completion_arr = mx.stop_gradient(mx.array(completion_ids))
            mx.eval(completion_arr)  # Evaluate immediately
            all_completions[global_idx] = completion_arr
            all_completion_texts[global_idx] = combined_text

        del continuation_results
        mx.clear_cache()
