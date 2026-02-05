"""
Cross-Sampling Module
=====================

Implements cross-sampling for GRPO training, where samples are paired
to create conversation history, allowing models to learn from 
contextual "background noise" during training.

Cross-sampling structure:
    Original Sample A: prompt_A → completion_A
    Original Sample B: prompt_B → completion_B
    
    After cross-sampling, Sample B becomes:
    messages = [
        {"role": "user", "content": prompt_A},
        {"role": "assistant", "content": completion_A},  # History (may be truncated)
        {"role": "user", "content": prompt_B},           # Never truncate
        # Model generates completion_B
    ]

Features:
    - Efficient length-based pair matching using index
    - Smart truncation of <think> sections (middle cut, preserve ends)
    - Configurable truncation marker
    - Metadata tracking for analysis
    - Deterministic pairing with seed

Usage:
    from .cross_sampling import CrossSampler, CrossSamplingConfig
    
    config = CrossSamplingConfig(
        enabled=True,
        ratio=0.3,  # 30% of samples get cross-sampled
        max_history_tokens=512,
    )
    
    sampler = CrossSampler(config, tokenizer, length_index)
    modified_data, metadata = sampler.apply(data)
"""

import random
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class CrossSamplingConfig:
    """Configuration for cross-sampling."""
    
    enabled: bool = False
    ratio: float = 0.0  # 0.0 - 1.0, percentage of samples to cross-sample
    max_history_tokens: int = 512
    seed: int = 42
    
    # Truncation settings
    truncation_marker: str = "\n[...reasoning truncated for brevity...]\n"
    preserve_start_lines: int = 3  # Keep first N lines of thinking
    preserve_end_lines: int = 1    # Keep last N lines of thinking
    
    # Thinking tag patterns
    think_open_tag: str = "<think>"
    think_close_tag: str = "</think>"


@dataclass
class CrossSampleMetadata:
    """Metadata for a cross-sampled pair."""
    
    sample_idx: int
    paired_with_idx: int
    history_tokens_original: int
    history_tokens_truncated: int
    was_truncated: bool
    truncation_ratio: float  # How much was removed


def truncate_thinking(
    text: str,
    max_tokens: int,
    tokenizer: Any,
    config: CrossSamplingConfig,
) -> Tuple[str, bool, int, int]:
    """
    Truncate the thinking section of a completion while preserving structure.
    
    Strategy:
    1. Find <think>...</think> section
    2. Preserve first N and last M lines
    3. Cut from the middle, insert marker
    4. Never truncate content outside <think> tags
    
    Args:
        text: Full completion text (including thinking)
        max_tokens: Maximum tokens allowed for this text
        tokenizer: Tokenizer for counting tokens
        config: Cross-sampling configuration
    
    Returns:
        Tuple of (truncated_text, was_truncated, original_tokens, final_tokens)
    """
    if not text:
        return text, False, 0, 0
    
    original_tokens = len(tokenizer.encode(text))
    
    if original_tokens <= max_tokens:
        return text, False, original_tokens, original_tokens
    
    # Find thinking section
    think_pattern = re.compile(
        re.escape(config.think_open_tag) + r"(.*?)" + re.escape(config.think_close_tag),
        re.DOTALL
    )
    
    match = think_pattern.search(text)
    
    if not match:
        # No thinking section - truncate from end (but try to keep answer)
        # Find answer section if present
        answer_match = re.search(r"(<answer>.*?</answer>)", text, re.DOTALL)
        if answer_match:
            answer_part = answer_match.group(1)
            pre_answer = text[:answer_match.start()]
            # Truncate pre-answer part
            while len(tokenizer.encode(pre_answer + answer_part)) > max_tokens and len(pre_answer) > 100:
                pre_answer = pre_answer[:int(len(pre_answer) * 0.8)]
            truncated = pre_answer + config.truncation_marker + answer_part
        else:
            # Just truncate from end
            truncated = text
            while len(tokenizer.encode(truncated)) > max_tokens and len(truncated) > 100:
                truncated = truncated[:int(len(truncated) * 0.9)]
            truncated += config.truncation_marker
        
        final_tokens = len(tokenizer.encode(truncated))
        return truncated, True, original_tokens, final_tokens
    
    # Extract parts
    before_think = text[:match.start()]
    thinking_content = match.group(1)
    after_think = text[match.end():]
    
    # Split thinking into lines
    thinking_lines = thinking_content.split('\n')
    
    # Calculate how much we need to remove
    non_thinking = before_think + config.think_open_tag + config.think_close_tag + after_think
    non_thinking_tokens = len(tokenizer.encode(non_thinking))
    marker_tokens = len(tokenizer.encode(config.truncation_marker))
    
    available_for_thinking = max_tokens - non_thinking_tokens - marker_tokens
    
    if available_for_thinking <= 0:
        # Can't fit any thinking - just use marker
        truncated = (
            before_think + 
            config.think_open_tag + 
            config.truncation_marker + 
            config.think_close_tag + 
            after_think
        )
        final_tokens = len(tokenizer.encode(truncated))
        return truncated, True, original_tokens, final_tokens
    
    # Preserve start and end lines
    n_preserve_start = min(config.preserve_start_lines, len(thinking_lines) // 2)
    n_preserve_end = min(config.preserve_end_lines, len(thinking_lines) // 2)
    
    start_lines = thinking_lines[:n_preserve_start]
    end_lines = thinking_lines[-n_preserve_end:] if n_preserve_end > 0 else []
    
    # Build truncated thinking
    start_text = '\n'.join(start_lines)
    end_text = '\n'.join(end_lines)
    
    # Check if we need to truncate further
    test_thinking = start_text + config.truncation_marker + end_text
    test_full = before_think + config.think_open_tag + test_thinking + config.think_close_tag + after_think
    
    if len(tokenizer.encode(test_full)) <= max_tokens:
        # This fits
        final_tokens = len(tokenizer.encode(test_full))
        return test_full, True, original_tokens, final_tokens
    
    # Need to trim the start/end sections too
    while len(tokenizer.encode(test_full)) > max_tokens:
        if len(start_text) > len(end_text) and len(start_text) > 50:
            # Trim start
            start_text = start_text[:int(len(start_text) * 0.7)]
        elif len(end_text) > 50:
            # Trim end
            end_text = end_text[int(len(end_text) * 0.3):]
        else:
            # Can't trim more
            break
        
        test_thinking = start_text + config.truncation_marker + end_text
        test_full = before_think + config.think_open_tag + test_thinking + config.think_close_tag + after_think
    
    final_tokens = len(tokenizer.encode(test_full))
    return test_full, True, original_tokens, final_tokens


class CrossSampler:
    """
    Applies cross-sampling to a dataset.
    
    Cross-sampling pairs samples together, using one sample's prompt+completion
    as conversation history for another sample.
    """
    
    def __init__(
        self,
        config: CrossSamplingConfig,
        tokenizer: Any,
        length_index: Optional[Dict[int, List[int]]] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.length_index = length_index
        
        # Set random seed for reproducibility
        self.rng = random.Random(config.seed)
        
        # Stats
        self.stats = {
            'total_samples': 0,
            'cross_sampled': 0,
            'truncated': 0,
            'total_tokens_saved': 0,
        }
    
    def _find_pair(
        self,
        idx: int,
        data: List[Tuple],
        used_indices: set,
        prompt_token_idx: int = 0,
    ) -> Optional[int]:
        """
        Find a suitable pairing sample for cross-sampling.
        
        Uses length index if available for efficient matching.
        
        Args:
            idx: Index of sample to find pair for
            data: Full dataset
            used_indices: Set of already-paired indices
            prompt_token_idx: Index of prompt tokens in tuple
        
        Returns:
            Index of paired sample, or None if no suitable pair found
        """
        # Get target length
        if isinstance(data[idx], tuple):
            target_tokens = data[idx][prompt_token_idx]
        else:
            target_tokens = data[idx]
        target_length = len(target_tokens) if hasattr(target_tokens, '__len__') else 0
        
        # Try to find a sample that fits within max_history_tokens
        # We want: len(history_prompt) + len(history_completion) <= max_history_tokens
        
        candidates = []
        
        if self.length_index is not None:
            # Use length index for efficient lookup
            bucket_size = 64
            target_bucket = target_length // bucket_size
            
            # Check nearby buckets
            for offset in range(5):
                for bucket in [target_bucket - offset, target_bucket + offset]:
                    if bucket in self.length_index and bucket >= 0:
                        for candidate_idx in self.length_index[bucket]:
                            if candidate_idx != idx and candidate_idx not in used_indices:
                                candidates.append(candidate_idx)
                                if len(candidates) >= 20:
                                    break
                    if len(candidates) >= 20:
                        break
                if len(candidates) >= 20:
                    break
        else:
            # Fall back to random sampling
            all_indices = [i for i in range(len(data)) if i != idx and i not in used_indices]
            candidates = self.rng.sample(all_indices, min(20, len(all_indices)))
        
        if not candidates:
            return None
        
        # Pick randomly from candidates
        return self.rng.choice(candidates)
    
    def apply(
        self,
        data: List[Dict[str, Any]],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        system_key: str = "system",
    ) -> Tuple[List[Dict[str, Any]], List[Optional[CrossSampleMetadata]]]:
        """
        Apply cross-sampling to raw data (before tokenization).
        
        Args:
            data: List of raw samples (dicts with prompt, answer, etc.)
            prompt_key: Key for prompt in dict
            answer_key: Key for answer/completion in dict
            system_key: Key for system message in dict
        
        Returns:
            Tuple of (modified_data, metadata_list)
            metadata_list[i] is CrossSampleMetadata if sample i was cross-sampled, else None
        """
        if not self.config.enabled or self.config.ratio <= 0:
            return data, [None] * len(data)
        
        self.stats['total_samples'] = len(data)
        
        # Determine which samples to cross-sample
        n_to_sample = int(len(data) * self.config.ratio)
        indices_to_sample = set(self.rng.sample(range(len(data)), n_to_sample))
        
        logger.info(f"Cross-sampling {n_to_sample}/{len(data)} samples ({self.config.ratio*100:.1f}%)")
        
        # Track used pairs to avoid duplicates
        used_as_history = set()
        
        # Result containers
        modified_data = []
        metadata_list = []
        
        for idx, sample in enumerate(data):
            if idx not in indices_to_sample:
                # Not selected for cross-sampling - keep original
                modified_data.append(sample.copy())
                metadata_list.append(None)
                continue
            
            # Find a pair
            pair_idx = self._find_pair(idx, data, used_as_history)
            
            if pair_idx is None:
                # No suitable pair found - keep original
                modified_data.append(sample.copy())
                metadata_list.append(None)
                continue
            
            used_as_history.add(pair_idx)
            pair_sample = data[pair_idx]
            
            # Get history content (from paired sample)
            history_prompt = str(pair_sample.get(prompt_key, ""))
            history_completion = str(pair_sample.get(answer_key, ""))
            
            # Truncate history completion if needed
            original_tokens = len(self.tokenizer.encode(history_completion))
            
            truncated_completion, was_truncated, orig_toks, final_toks = truncate_thinking(
                text=history_completion,
                max_tokens=self.config.max_history_tokens,
                tokenizer=self.tokenizer,
                config=self.config,
            )
            
            if was_truncated:
                self.stats['truncated'] += 1
                self.stats['total_tokens_saved'] += (orig_toks - final_toks)
            
            # Build the cross-sampled messages
            # The modified sample will have history prepended
            modified_sample = sample.copy()
            
            # Store original prompt
            original_prompt = str(sample.get(prompt_key, ""))
            
            # Create conversation history structure
            # This will be used when applying chat template
            modified_sample['_cross_sampled'] = True
            modified_sample['_cross_sample_history'] = [
                {"role": "user", "content": history_prompt},
                {"role": "assistant", "content": truncated_completion},
            ]
            modified_sample['_cross_sample_source_idx'] = pair_idx
            modified_sample['_original_prompt'] = original_prompt
            
            # Create metadata
            meta = CrossSampleMetadata(
                sample_idx=idx,
                paired_with_idx=pair_idx,
                history_tokens_original=orig_toks,
                history_tokens_truncated=final_toks,
                was_truncated=was_truncated,
                truncation_ratio=(orig_toks - final_toks) / orig_toks if orig_toks > 0 else 0,
            )
            
            modified_data.append(modified_sample)
            metadata_list.append(meta)
            self.stats['cross_sampled'] += 1
        
        logger.info(
            f"Cross-sampling complete: {self.stats['cross_sampled']} samples modified, "
            f"{self.stats['truncated']} truncated, "
            f"{self.stats['total_tokens_saved']} tokens saved from truncation"
        )
        
        return modified_data, metadata_list
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cross-sampling statistics."""
        return self.stats.copy()


def apply_cross_sample_to_messages(
    sample: Dict[str, Any],
    system_message: Optional[str] = None,
    prompt_key: str = "prompt",
) -> List[Dict[str, str]]:
    """
    Convert a potentially cross-sampled sample to chat messages format.
    
    Args:
        sample: Sample dict (may have _cross_sample_history)
        system_message: Optional system message to prepend
        prompt_key: Key for the current prompt
    
    Returns:
        List of message dicts for chat template
    """
    messages = []
    
    # Add system message if provided
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add cross-sample history if present
    if sample.get('_cross_sampled') and '_cross_sample_history' in sample:
        messages.extend(sample['_cross_sample_history'])
    
    # Add current prompt
    current_prompt = sample.get('_original_prompt', sample.get(prompt_key, ""))
    messages.append({"role": "user", "content": str(current_prompt)})
    
    return messages
