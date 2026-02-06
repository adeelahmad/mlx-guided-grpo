"""
Base Dataset Loader - SOLID Architecture
=========================================

Abstract base class for type-specific dataset loading.

Responsibilities (SRP):
- Load data from files
- Validate samples
- Type-specific preprocessing
- Tokenization strategies

Design Patterns:
- Template Method: load() orchestrates the workflow
- Observer Pattern: Hooks for data augmentation
- Strategy Pattern: Type-specific preprocessing
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["BaseDatasetLoader", "DataSample", "DatasetHooks", "LoadedDataset"]

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DataSample:
    """Structured data sample.

    Attributes:
        prompt_tokens: Tokenized prompt
        answer_tokens: Tokenized answer
        prompt_text: Raw prompt string
        answer_text: Raw answer string
        type_info: Type metadata (dict, str, etc.)
        metadata: Additional info
    """
    prompt_tokens: list[int]
    answer_tokens: list[int]
    prompt_text: str
    answer_text: str
    type_info: Any
    metadata: dict[str, Any]


@dataclass
class LoadedDataset:
    """Dataset loading result.

    Attributes:
        samples: List of processed samples
        metadata: Dataset-level metadata (stats, type distribution, etc.)
        type_name: Detected type name
    """
    samples: list[DataSample]
    metadata: dict[str, Any]
    type_name: str


# =============================================================================
# OBSERVER HOOKS
# =============================================================================

class DatasetHooks:
    """Observer hooks for dataset loading lifecycle."""

    def before_load(self, path: Path) -> None:
        """Called before loading file."""
        pass

    def after_load(self, raw_data: list[dict]) -> list[dict]:
        """Called after loading raw data.

        Can filter, augment, or transform data.
        """
        return raw_data

    def before_preprocess(self, sample: dict) -> dict:
        """Called before preprocessing each sample."""
        return sample

    def after_preprocess(self, sample: DataSample) -> DataSample:
        """Called after preprocessing each sample."""
        return sample

    def on_validation_failure(self, sample: dict, reason: str) -> None:
        """Called when sample validation fails."""
        pass

    def on_load_complete(self, dataset: LoadedDataset) -> LoadedDataset:
        """Called after entire dataset is loaded."""
        return dataset


# =============================================================================
# BASE DATASET LOADER
# =============================================================================

class BaseDatasetLoader(ABC):
    """Abstract base class for type-specific dataset loaders.

    Subclasses must implement:
    - validate_sample(): Check if sample is valid for this type
    - preprocess_sample(): Type-specific preprocessing
    - get_system_prompt(): Type-specific system message

    Usage:
        class ToolCallLoader(BaseDatasetLoader):
            def validate_sample(self, sample):
                return "function" in sample.get("answer", "")

            def preprocess_sample(self, sample):
                # Clean up function calls
                sample["answer"] = normalize_function(sample["answer"])
                return sample
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize dataset loader.

        Args:
            tokenizer: HuggingFace tokenizer
            hooks: Observer hooks for lifecycle events
            event_bus: Optional event bus for publishing lifecycle events
        """
        self.tokenizer = tokenizer
        self.hooks = hooks or DatasetHooks()
        self.event_bus = event_bus
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # =========================================================================
    # ABSTRACT METHODS (Must implement)
    # =========================================================================

    @abstractmethod
    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate if sample is appropriate for this type.

        Args:
            sample: Raw data sample

        Returns:
            (is_valid, reason_if_invalid)
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess_sample(self, sample: dict) -> dict:
        """Apply type-specific preprocessing to sample.

        Args:
            sample: Raw data sample

        Returns:
            Preprocessed sample
        """
        raise NotImplementedError

    @abstractmethod
    def get_system_prompt(self, sample: dict) -> str:
        """Get type-specific system prompt.

        Args:
            sample: Data sample

        Returns:
            System message string
        """
        raise NotImplementedError

    # =========================================================================
    # TEMPLATE METHOD
    # =========================================================================

    def load(
        self,
        path: str | Path,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> LoadedDataset:
        """Load dataset (template method).

        Orchestrates the loading workflow:
        1. Call before_load hook
        2. Load raw data from file
        3. Call after_load hook
        4. Validate and preprocess each sample
        5. Tokenize samples
        6. Call on_load_complete hook

        Args:
            path: Path to data file (JSONL)
            max_samples: Limit number of samples
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling

        Returns:
            LoadedDataset with processed samples
        """
        path = Path(path)

        # Hook: before load
        self.hooks.before_load(path)

        # Load raw data
        self._logger.info(f"Loading data from {path}")
        raw_data = self._load_file(path, max_samples)

        # Hook: after load
        raw_data = self.hooks.after_load(raw_data)

        # Shuffle if requested
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(raw_data)
            self._logger.info(f"Shuffled {len(raw_data)} samples (seed={seed})")

        # Process samples
        samples = []
        skipped = 0

        for i, raw_sample in enumerate(raw_data):
            try:
                # Hook: before preprocess
                raw_sample = self.hooks.before_preprocess(raw_sample)

                # Validate
                is_valid, reason = self.validate_sample(raw_sample)
                if not is_valid:
                    skipped += 1
                    self.hooks.on_validation_failure(raw_sample, reason or "Unknown")
                    self._publish_event("sample.validated", {
                        "valid": False, "reason": reason,
                    })
                    continue

                self._publish_event("sample.validated", {"valid": True})

                # Preprocess
                processed = self.preprocess_sample(raw_sample.copy())

                # Tokenize
                data_sample = self._tokenize_sample(processed)

                # Hook: after preprocess
                data_sample = self.hooks.after_preprocess(data_sample)

                samples.append(data_sample)

            except Exception as e:
                self._logger.error(f"Error processing sample {i}: {e}", exc_info=True)
                skipped += 1

        # Build metadata
        metadata = {
            "total_loaded": len(raw_data),
            "valid_samples": len(samples),
            "skipped_samples": skipped,
            "shuffle": shuffle,
            "seed": seed if shuffle else None,
        }

        # Create dataset
        dataset = LoadedDataset(
            samples=samples,
            metadata=metadata,
            type_name=self.get_type_name(),
        )

        # Hook: load complete
        dataset = self.hooks.on_load_complete(dataset)

        self._logger.info(
            f"Loaded {len(samples)} samples, skipped {skipped}"
        )

        self._publish_event("sample.loaded", {
            "type": self.get_type_name(),
            "valid": len(samples),
            "skipped": skipped,
        })

        return dataset

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _load_file(
        self,
        path: Path,
        max_samples: Optional[int] = None,
    ) -> list[dict]:
        """Load raw data from JSONL file."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        data = []
        with open(path, 'r', encoding='utf-8') as f:
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
                    self._logger.warning(f"Line {line_num}: Invalid JSON - {e}")

        return data

    def _tokenize_sample(self, sample: dict) -> DataSample:
        """Tokenize a preprocessed sample."""
        prompt_text = sample.get("prompt", "")
        answer_text = sample.get("answer", "")
        type_info = sample.get("type", None)

        # Get system prompt
        system_text = self.get_system_prompt(sample)

        # Build messages for chat template
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_text},
        ]

        # Tokenize prompt with chat template
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Tokenize answer
        answer_tokens = self.tokenizer.encode(answer_text) if answer_text else []

        # Extract metadata
        metadata = {
            k: v for k, v in sample.items()
            if k not in ("prompt", "answer", "type")
        }

        return DataSample(
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            prompt_text=prompt_text,
            answer_text=answer_text,
            type_info=type_info,
            metadata=metadata,
        )

    def get_type_name(self) -> str:
        """Get type name for this loader.

        Override to customize. Default: class name without 'Loader'.
        """
        class_name = self.__class__.__name__
        if class_name.endswith('Loader'):
            return class_name[:-6].lower()
        return class_name.lower()

    def _publish_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Publish event if event bus is configured."""
        if self.event_bus is not None:
            from ..events import Event
            self.event_bus.publish(Event(name=event_name, data=data))

    def __call__(
        self,
        path: str | Path,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> LoadedDataset:
        """Make loader callable."""
        return self.load(path, max_samples, shuffle, seed)
