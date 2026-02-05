"""Dataset utilities for GRPO training.

This module provides dataset classes and loading utilities focused on GRPO training.

Classes:
    GRPODataset: Main dataset class for GRPO training with prompt-answer pairs
    CacheDataset: Wrapper for lazy processing and caching
    ConcatenatedDataset: Combines multiple datasets

Functions:
    load_dataset: Main entry point for loading datasets
    create_dataset: Factory function for creating appropriate dataset type
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import random
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

__all__ = [
    "GRPODataset",
    "CacheDataset",
    "ConcatenatedDataset",
    "create_dataset",
    "load_dataset",
    "load_local_dataset",
    "load_hf_dataset",
]


# =============================================================================
# GRPO Dataset
# =============================================================================


class GRPODataset:
    """Dataset for GRPO training with prompt-answer pairs.

    Each sample contains:
    - Tokenized prompt
    - Tokenized answer (ground truth)
    - Raw prompt string
    - Raw answer string
    - Type metadata (for type-aware rewards)

    Args:
        data: List of dictionaries with prompt/answer pairs
        tokenizer: HuggingFace tokenizer
        prompt_key: Key for prompt in data dict
        answer_key: Key for answer in data dict
        system_key: Key for system message in data dict
        type_key: Key for sample type in data dict
        ground_truth_key: Key for ground truth (exam-style datasets)
        require_think_tags: Skip samples without <think> tags
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        system_key: str = "system",
        type_key: str = "type",
        ground_truth_key: str = "ground_truth",
        text_completion_key: str | None = None,
        require_think_tags: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self._data: list[tuple] = []
        skipped_count = 0
        exam_count = 0

        for item in data:
            prompt_str = str(item[prompt_key])
            answer_str = str(item.get(answer_key, ""))
            raw_type = item.get(type_key, None)
            ground_truth = item.get(ground_truth_key, None)
            possible_boxed_answers = item.get("possible_boxed_answers", None)

            # Build metadata dict for type_info
            if isinstance(raw_type, dict):
                type_info = raw_type
                if possible_boxed_answers is not None and "possible_boxed_answers" not in type_info:
                    type_info["possible_boxed_answers"] = possible_boxed_answers
            else:
                type_info = {
                    "type": raw_type,
                    "ground_truth": ground_truth,
                }
                if possible_boxed_answers is not None:
                    type_info["possible_boxed_answers"] = possible_boxed_answers

            # Check if this is an exam-type sample
            is_exam_type = (
                type_info.get("type") == "exam"
                or ground_truth is not None
                or item.get("is_exam", False)
            )

            if is_exam_type:
                type_info["is_exam"] = True
                exam_count += 1

            # Skip samples without think tags if required (but NOT for exam type)
            if require_think_tags and not is_exam_type:
                if "<think>" not in answer_str or "</think>" not in answer_str:
                    skipped_count += 1
                    continue

            # Tokenize
            if text_completion_key is None:
                default_system_str = self._get_system_prompt(is_exam_type)
                system_str = item.get(system_key, default_system_str)

                messages = [
                    {"role": "system", "content": system_str},
                    {"role": "user", "content": prompt_str},
                ]
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            else:
                prompt_tokens = tokenizer.encode(str(item[text_completion_key]))

            answer_tokens = tokenizer.encode(answer_str) if answer_str else []
            self._data.append((prompt_tokens, answer_tokens, prompt_str, answer_str, type_info))

        if skipped_count > 0:
            logging.info(f"[GRPODataset] Skipped {skipped_count} samples without <think> tags")
        if exam_count > 0:
            logging.info(f"[GRPODataset] Loaded {exam_count} exam-type samples")

        # Shuffle data
        if shuffle and self._data:
            random.seed(seed)
            random.shuffle(self._data)
            logging.info(f"[GRPODataset] Shuffled {len(self._data)} samples (seed={seed})")

    def _get_system_prompt(self, is_exam: bool) -> str:
        """Get default system prompt."""
        return """I'm NeuralAI, an AI assistant. In this environment, I analyze problems carefully.

<think>
[In this section, I will perform analytical thinking about the problem]
- I will explain my thoughts clearly
- I will use markdown tables for calculations when helpful
- I will identify constraints and verify logical consistency
- I will consider alternative approaches
</think>

1. Work through the problem in <think>...</think> tags
2. Provide your final answer as \\boxed{answer}
3. Add a brief explanation"""

    def __getitem__(self, idx: int) -> tuple:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)

    def process(self, d: Any) -> Any:
        return d


# =============================================================================
# Utility Dataset Classes
# =============================================================================


class ConcatenatedDataset:
    """Combines multiple datasets into one."""

    def __init__(self, data: list[Any]):
        self._data = data
        self._len = sum(len(d) for d in self._data)

    def __getitem__(self, idx: int) -> dict:
        for data_idx, data in enumerate(self._data):
            j = idx - len(data)
            if j < 0:
                break
            idx = j
        datum = data[idx]
        datum["_dataset"] = data_idx
        return datum

    def process(self, d: dict) -> Any:
        return self._data[d["_dataset"]].process(d)

    def __len__(self) -> int:
        return self._len


class CacheDataset:
    """Wrapper that lazily processes and caches dataset items."""

    def __init__(self, data: Any):
        self._data = data
        self._proc_data: list[Any | None] = [None] * len(data)

    def itemlen(self, idx: int) -> int:
        return len(self._data[idx])

    def __getitem__(self, idx: int) -> Any:
        if self._proc_data[idx] is None:
            self._proc_data[idx] = self._data.process(self._data[idx])
        return self._proc_data[idx]

    def __len__(self) -> int:
        return len(self._data)


# =============================================================================
# Dataset Factory
# =============================================================================


def create_dataset(
    data: list[dict],
    tokenizer: PreTrainedTokenizer,
    config: Any,
) -> GRPODataset:
    """Create a dataset from raw data.

    Args:
        data: List of data dictionaries
        tokenizer: HuggingFace tokenizer
        config: Configuration object with dataset options

    Returns:
        GRPODataset instance
    """
    prompt_feature = getattr(config, "prompt_feature", "prompt")
    answer_feature = getattr(config, "answer_feature", "answer")
    system_feature = getattr(config, "system_feature", "system")
    type_feature = getattr(config, "type_feature", "type")

    sample = data[0]

    if prompt_feature not in sample:
        raise ValueError(
            f"Dataset must contain '{prompt_feature}' field. "
            f"Found fields: {list(sample.keys())}"
        )

    # Get GRPO options from config
    require_think_tags = getattr(config, "require_think_tags", True)
    shuffle_data = getattr(config, "shuffle_data", True)
    shuffle_seed = getattr(config, "shuffle_seed", 42)

    # Apply cross-sampling if enabled
    cross_sampling_enabled = getattr(config, "cross_sampling_enabled", False)
    if cross_sampling_enabled:
        from .cross_sampling import CrossSampler, CrossSamplingConfig

        cross_config = CrossSamplingConfig(
            enabled=True,
            ratio=getattr(config, "cross_sampling_ratio", 0.3),
            max_history_tokens=getattr(config, "cross_sampling_max_history_tokens", 512),
            seed=getattr(config, "cross_sampling_seed", 42),
            truncation_marker=getattr(
                config, "cross_sampling_truncation_marker", "\n[...truncated...]\n"
            ),
        )
        cross_sampler = CrossSampler(cross_config, tokenizer)
        data, _ = cross_sampler.apply(
            data,
            prompt_key=prompt_feature,
            answer_key=answer_feature,
            system_key=system_feature,
        )
        logging.info(f"[Cross-sampling] Applied to {cross_sampler.stats['cross_sampled']} samples")

    return GRPODataset(
        data=data,
        tokenizer=tokenizer,
        prompt_key=prompt_feature,
        answer_key=answer_feature,
        system_key=system_feature,
        type_key=type_feature,
        require_think_tags=require_think_tags,
        shuffle=shuffle_data,
        seed=shuffle_seed,
    )


# =============================================================================
# Dataset Loading Functions
# =============================================================================


def load_local_dataset(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    config: Any,
    cache_dataset: bool = True,
    cache_dir: str | None = None,
    force_reload: bool = False,
) -> tuple[Any, Any, Any]:
    """Load local dataset with optional caching.

    Args:
        data_path: Path to dataset directory
        tokenizer: HuggingFace tokenizer
        config: Configuration object
        cache_dataset: Whether to cache processed data
        cache_dir: Custom cache directory
        force_reload: Force reload ignoring cache

    Returns:
        Tuple of (train, valid, test) datasets
    """

    def get_cache_path(subset_name: str) -> Path:
        if cache_dir:
            cache_base = Path(cache_dir)
        else:
            cache_base = data_path / ".cache"
        cache_base.mkdir(parents=True, exist_ok=True)

        config_str = f"{getattr(tokenizer, 'name_or_path', 'unknown')}_grpo"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return cache_base / f"{subset_name}_{config_hash}.pkl"

    def get_file_hash(path: Path) -> str:
        if not path.exists():
            return ""
        stat = path.stat()
        return f"{stat.st_mtime}_{stat.st_size}"

    def load_subset(path: Path, subset_name: str) -> Any:
        if not path.exists():
            return []

        cache_path = get_cache_path(subset_name)
        file_hash = get_file_hash(path)

        # Try to load from cache
        if cache_dataset and not force_reload and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("file_hash") == file_hash:
                    logging.info(
                        f"  Loaded {subset_name} from cache ({len(cached['data'])} samples)"
                    )
                    return cached["data"]
                else:
                    logging.info(f"  Cache invalidated for {subset_name} (file changed)")
            except Exception as e:
                logging.warning(f"  Cache load failed for {subset_name}: {e}")

        # Load and process fresh
        logging.info(f"  Processing {subset_name}...")
        with open(path, "r") as fid:
            raw_data = [json.loads(line) for line in fid]

        dataset = create_dataset(raw_data, tokenizer, config)

        # Save to cache
        if cache_dataset:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump({"file_hash": file_hash, "data": dataset}, f)
                logging.info(f"  Cached {subset_name} ({len(dataset)} samples)")
            except Exception as e:
                logging.warning(f"  Cache save failed for {subset_name}: {e}")

        return dataset

    logging.info(f"Loading dataset from {data_path}" + (" (with caching)" if cache_dataset else ""))
    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl", n) for n in names]
    return train, valid, test


def load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    config: Any,
) -> tuple[Any, Any, Any]:
    """Load dataset from HuggingFace Hub.

    Args:
        data_id: HuggingFace dataset identifier
        tokenizer: HuggingFace tokenizer
        config: Configuration object

    Returns:
        Tuple of (train, valid, test) datasets
    """
    from datasets import exceptions
    from datasets import load_dataset as hf_load

    try:
        dataset = hf_load(data_id)
        names = ("train", "valid", "test")
        train, valid, test = [
            create_dataset(dataset[n], tokenizer, config) if n in dataset.keys() else []
            for n in names
        ]
    except exceptions.DatasetNotFoundError:
        raise ValueError(f"HuggingFace dataset not found: {data_id}")

    return train, valid, test


def load_custom_hf_dataset(
    args: Any,
    tokenizer: PreTrainedTokenizer,
) -> tuple[Any, Any, Any]:
    """Load custom HuggingFace dataset with specified configuration.

    Args:
        args: Arguments object with hf_dataset configuration
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (train, valid, test) datasets
    """
    import datasets

    def create_hf_dataset(dataset_name: str, config: Any, split: str, hf_config: dict) -> Any:
        ds = datasets.load_dataset(dataset_name, split=split, **hf_config)
        return create_dataset(ds, tokenizer, config)

    dataset_collection = args.hf_dataset
    if isinstance(dataset_collection, dict):
        dataset_collection = [dataset_collection]

    collection = []
    for ds in dataset_collection:
        ds_path = ds["path"]
        logging.info(f"Loading HuggingFace dataset {ds_path}")
        config = types.SimpleNamespace(**ds)
        hf_config = ds.get("config", {})

        if args.train:
            train_split = ds.get("train_split", "train[:80%]")
            valid_split = ds.get("valid_split", "train[-10%:]")
            train = create_hf_dataset(ds_path, config, train_split, hf_config)
            valid = create_hf_dataset(ds_path, config, valid_split, hf_config)
        else:
            train, valid = [], []

        if args.test:
            test_split = ds.get("test_split")
            test = create_hf_dataset(ds_path, config, test_split, hf_config)
        else:
            test = []

        collection.append((train, valid, test))

    if len(collection) == 1:
        return collection[0]

    return tuple(map(ConcatenatedDataset, zip(*collection)))


def load_dataset(
    args: Any,
    tokenizer: PreTrainedTokenizer,
) -> tuple[Any, Any, Any]:
    """Main entry point for loading datasets.

    Args:
        args: Arguments object with dataset configuration
        tokenizer: HuggingFace tokenizer

    Returns:
        Tuple of (train, valid, test) datasets

    Raises:
        ValueError: If required datasets are missing
    """
    cache_dataset = getattr(args, "cache_dataset", True)
    cache_dir = getattr(args, "cache_dir", None)
    force_reload = getattr(args, "force_reload", False)

    if getattr(args, "hf_dataset", False):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(
                data_path,
                tokenizer,
                args,
                cache_dataset=cache_dataset,
                cache_dir=cache_dir,
                force_reload=force_reload,
            )
        else:
            logging.info(f"Loading HuggingFace dataset {args.data}")
            train, valid, test = load_hf_dataset(args.data, tokenizer, args)

    if args.train and len(train) == 0:
        raise ValueError("Training set not found or empty.")
    if args.train and len(valid) == 0:
        raise ValueError("Validation set not found or empty.")
    if args.test and len(test) == 0:
        raise ValueError("Test set not found or empty.")

    return train, valid, test
