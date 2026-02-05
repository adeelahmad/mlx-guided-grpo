"""
Type-Aware Dataset Loader
==========================

Intelligent dataset loading with automatic type detection and configuration.

Features:
- Auto-detects dataset types from samples
- Applies type-specific preprocessing
- Supports mixed-type datasets
- Dynamic configuration merging

Usage:
    # Auto-detect and load
    dataset = load_typed_dataset("data.jsonl", tokenizer)

    # Get aggregated config
    config = dataset.get_aggregated_config()
    rewards = config.reward_config
    generation = config.generation_strategy
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .registry import (
    detect_dataset_type,
    get_type_handler,
    RewardConfig,
    GenerationStrategy,
    CurriculumConfig,
    compose_rewards,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

__all__ = [
    "TypedDataset",
    "load_typed_dataset",
    "DatasetConfig",
]

logger = logging.getLogger(__name__)


# =============================================================================
# AGGREGATED CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Aggregated configuration from all types in dataset.

    Attributes:
        reward_config: Combined reward configuration
        generation_strategy: Merged generation strategy
        curriculum_config: Merged curriculum configuration
        type_distribution: Distribution of types in dataset
        dominant_type: Most common type
    """
    reward_config: RewardConfig
    generation_strategy: GenerationStrategy
    curriculum_config: CurriculumConfig
    type_distribution: dict[str, int]
    dominant_type: str


# =============================================================================
# TYPED DATASET
# =============================================================================

class TypedDataset:
    """Dataset with automatic type detection and configuration.

    This wraps the standard GRPODataset but adds:
    - Automatic type detection per sample
    - Type-specific preprocessing
    - Aggregated configuration
    - Statistics and reporting
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        auto_detect: bool = True,
        default_type: str = "default",
        **kwargs
    ):
        """
        Args:
            data: List of data samples
            tokenizer: HuggingFace tokenizer
            auto_detect: Enable automatic type detection
            default_type: Fallback type if detection fails
            **kwargs: Additional args for GRPODataset
        """
        self.tokenizer = tokenizer
        self.auto_detect = auto_detect
        self.default_type = default_type

        # Type statistics
        self.type_counts: Counter = Counter()
        self.type_handlers: dict[str, Any] = {}

        # Process samples
        self.processed_data = self._process_samples(data)

        logger.info(f"Loaded {len(self.processed_data)} samples")
        logger.info(f"Type distribution: {dict(self.type_counts)}")

    def _process_samples(
        self,
        data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process samples with type detection and preprocessing."""
        processed = []

        for i, sample in enumerate(data):
            # Detect type
            if self.auto_detect:
                detected_type = detect_dataset_type(sample)
                if detected_type is None:
                    detected_type = self.default_type
                    logger.debug(
                        f"Sample {i}: Could not detect type, "
                        f"using default '{self.default_type}'"
                    )
            else:
                # Use explicit type or default
                detected_type = sample.get("type", self.default_type)

            # Track statistics
            self.type_counts[detected_type] += 1

            # Get handler (cached)
            if detected_type not in self.type_handlers:
                try:
                    self.type_handlers[detected_type] = get_type_handler(detected_type)
                except KeyError:
                    logger.warning(
                        f"Unknown type '{detected_type}', using default handler"
                    )
                    detected_type = self.default_type
                    self.type_handlers[detected_type] = get_type_handler(detected_type)

            handler = self.type_handlers[detected_type]

            # Validate sample
            if not handler.validate_sample(sample):
                logger.warning(
                    f"Sample {i}: Failed validation for type '{detected_type}'"
                )

            # Preprocess
            processed_sample = handler.preprocess_sample(sample.copy())

            # Ensure type field is set
            processed_sample["type"] = detected_type

            processed.append(processed_sample)

        return processed

    def get_aggregated_config(
        self,
        strategy: str = "dominant"
    ) -> DatasetConfig:
        """Get aggregated configuration from all types.

        Args:
            strategy: Aggregation strategy:
                - "dominant": Use most common type's config
                - "merge": Combine configs from all types
                - "weighted": Weight by type frequency

        Returns:
            Aggregated dataset configuration
        """
        if not self.type_counts:
            # Empty dataset, use default
            handler = get_type_handler(self.default_type)
            return DatasetConfig(
                reward_config=handler.get_reward_config(),
                generation_strategy=handler.get_generation_strategy(),
                curriculum_config=handler.get_curriculum_config(),
                type_distribution={},
                dominant_type=self.default_type,
            )

        # Find dominant type
        dominant_type = self.type_counts.most_common(1)[0][0]

        if strategy == "dominant":
            # Use dominant type's config
            handler = self.type_handlers[dominant_type]
            return DatasetConfig(
                reward_config=handler.get_reward_config(),
                generation_strategy=handler.get_generation_strategy(),
                curriculum_config=handler.get_curriculum_config(),
                type_distribution=dict(self.type_counts),
                dominant_type=dominant_type,
            )

        elif strategy == "merge":
            # Merge all configs
            reward_configs = [
                h.get_reward_config()
                for h in self.type_handlers.values()
            ]
            merged_rewards = compose_rewards(*reward_configs)

            # For generation/curriculum, use dominant
            handler = self.type_handlers[dominant_type]
            generation = handler.get_generation_strategy()
            curriculum = handler.get_curriculum_config()

            return DatasetConfig(
                reward_config=merged_rewards,
                generation_strategy=generation,
                curriculum_config=curriculum,
                type_distribution=dict(self.type_counts),
                dominant_type=dominant_type,
            )

        elif strategy == "weighted":
            # Weight configs by frequency
            total = sum(self.type_counts.values())

            # Collect all reward functions with weights
            all_rewards: defaultdict = defaultdict(float)

            for type_name, count in self.type_counts.items():
                handler = self.type_handlers[type_name]
                config = handler.get_reward_config()
                type_weight = count / total

                for func, weight in zip(config.functions, config.normalized_weights):
                    all_rewards[func] += weight * type_weight

            # Create weighted config
            weighted_rewards = RewardConfig(
                functions=tuple(all_rewards.keys()),
                weights=tuple(all_rewards.values()),
            )

            # Use dominant for generation/curriculum
            handler = self.type_handlers[dominant_type]

            return DatasetConfig(
                reward_config=weighted_rewards,
                generation_strategy=handler.get_generation_strategy(),
                curriculum_config=handler.get_curriculum_config(),
                type_distribution=dict(self.type_counts),
                dominant_type=dominant_type,
            )

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def get_samples_by_type(self, type_name: str) -> list[dict[str, Any]]:
        """Get all samples of a specific type."""
        return [s for s in self.processed_data if s["type"] == type_name]

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.processed_data[idx]


# =============================================================================
# CONVENIENCE LOADER
# =============================================================================

def load_typed_dataset(
    data_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    auto_detect: bool = True,
    default_type: str = "default",
    max_samples: int | None = None,
    **kwargs
) -> TypedDataset:
    """Load dataset with automatic type detection.

    Args:
        data_path: Path to JSONL file
        tokenizer: HuggingFace tokenizer
        auto_detect: Enable automatic type detection
        default_type: Fallback type
        max_samples: Limit number of samples (for testing)
        **kwargs: Additional args for TypedDataset

    Returns:
        TypedDataset instance

    Example:
        dataset = load_typed_dataset(
            "data/mixed_dataset.jsonl",
            tokenizer,
            auto_detect=True
        )

        config = dataset.get_aggregated_config()
        print(f"Dominant type: {config.dominant_type}")
        print(f"Rewards: {config.reward_config.functions}")
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info(f"Loading typed dataset from {data_path}")

    # Load JSONL
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_samples and len(data) >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue

    logger.info(f"Loaded {len(data)} raw samples")

    # Create typed dataset
    return TypedDataset(
        data=data,
        tokenizer=tokenizer,
        auto_detect=auto_detect,
        default_type=default_type,
        **kwargs
    )


# =============================================================================
# DATASET MERGER
# =============================================================================

def merge_typed_datasets(
    *datasets: TypedDataset,
    shuffle: bool = True,
    seed: int = 42,
) -> TypedDataset:
    """Merge multiple typed datasets.

    Args:
        *datasets: TypedDataset instances to merge
        shuffle: Shuffle merged data
        seed: Random seed for shuffling

    Returns:
        Combined TypedDataset
    """
    if not datasets:
        raise ValueError("Must provide at least one dataset")

    # Combine data
    all_data = []
    for ds in datasets:
        all_data.extend(ds.processed_data)

    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(all_data)

    # Use first dataset's tokenizer and settings
    return TypedDataset(
        data=all_data,
        tokenizer=datasets[0].tokenizer,
        auto_detect=datasets[0].auto_detect,
        default_type=datasets[0].default_type,
    )
