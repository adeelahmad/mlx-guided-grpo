"""
Dataset Caching Module
======================

Provides efficient caching for GRPO datasets to avoid repeated
processing of large datasets.

Features:
- Content-based hashing for cache invalidation
- Config-based hashing (tokenizer, features, etc.)
- File modification time tracking
- Length index for efficient cross-sampling
- Support for both raw and processed data

Usage:
    from .dataset_cache import DatasetCache

    cache = DatasetCache(
        data_path=Path("data/train.jsonl"),
        cache_dir=None,  # Uses data_path/.cache by default
        config=args,
    )

    if cache.is_valid():
        data, length_index = cache.load()
    else:
        data = process_data(...)
        length_index = build_length_index(data)
        cache.save(data, length_index)
"""

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata stored alongside cached data."""

    # Source file info
    source_path: str
    source_mtime: float
    source_size: int
    source_hash: str  # First 64KB content hash

    # Config info
    config_hash: str

    # Cache info
    cache_version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    num_samples: int = 0

    # Cross-sampling support
    has_length_index: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheMetadata":
        return cls(**d)


def compute_file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Compute hash of file's first chunk for quick comparison."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            chunk = f.read(chunk_size)
            hasher.update(chunk)
    except Exception as e:
        logger.warning(f"Could not hash file {path}: {e}")
        return ""
    return hasher.hexdigest()[:16]


def compute_config_hash(config: Any, tokenizer: Any = None) -> str:
    """
    Compute hash of relevant config options.

    Includes:
    - Tokenizer vocab size and name
    - Feature keys
    - Any processing options
    """
    hasher = hashlib.sha256()

    # Tokenizer info
    if tokenizer is not None:
        try:
            hasher.update(str(tokenizer.vocab_size).encode())
            if hasattr(tokenizer, "name_or_path"):
                hasher.update(str(tokenizer.name_or_path).encode())
        except Exception:
            pass

    # Config options that affect processing
    config_keys = [
        "prompt_feature",
        "answer_feature",
        "system_feature",
        "type_feature",
        "train_mode",
        "text_feature",
        "chat_feature",
        "completion_feature",
        "max_seq_length",
        # Cross-sampling config
        "cross_sampling_enabled",
        "cross_sampling_ratio",
        "cross_sampling_max_history_tokens",
    ]

    for key in config_keys:
        value = getattr(config, key, None)
        if value is not None:
            hasher.update(f"{key}={value}".encode())

    return hasher.hexdigest()[:16]


class DatasetCache:
    """
    Manages caching of processed datasets.

    Cache structure:
        {cache_dir}/
            {dataset_name}_{hash}.pkl       - Processed data
            {dataset_name}_{hash}.meta.json - Metadata
            {dataset_name}_{hash}.idx.pkl   - Length index (for cross-sampling)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        config: Any,
        tokenizer: Any = None,
        cache_dir: Optional[Union[str, Path]] = None,
        enabled: bool = True,
    ):
        self.data_path = Path(data_path)
        self.config = config
        self.tokenizer = tokenizer
        self.enabled = enabled

        # Determine cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            # Default: .cache in same directory as data
            self.cache_dir = self.data_path.parent / ".cache"

        # Compute hashes
        self._source_hash = ""
        self._config_hash = ""
        self._cache_key = ""

        if self.enabled and self.data_path.exists():
            self._compute_hashes()

    def _compute_hashes(self):
        """Compute all necessary hashes."""
        self._source_hash = compute_file_hash(self.data_path)
        self._config_hash = compute_config_hash(self.config, self.tokenizer)

        # Cache key combines source and config
        combined = f"{self._source_hash}_{self._config_hash}"
        self._cache_key = hashlib.sha256(combined.encode()).hexdigest()[:12]

    @property
    def cache_name(self) -> str:
        """Base name for cache files."""
        stem = self.data_path.stem  # e.g., "train" from "train.jsonl"
        return f"{stem}_{self._cache_key}"

    @property
    def data_cache_path(self) -> Path:
        return self.cache_dir / f"{self.cache_name}.pkl"

    @property
    def meta_cache_path(self) -> Path:
        return self.cache_dir / f"{self.cache_name}.meta.json"

    @property
    def index_cache_path(self) -> Path:
        return self.cache_dir / f"{self.cache_name}.idx.pkl"

    def is_valid(self) -> bool:
        """Check if cache exists and is valid."""
        if not self.enabled:
            return False

        # Check all required files exist
        if not self.data_cache_path.exists():
            logger.info(f"Cache miss: data file not found")
            return False

        if not self.meta_cache_path.exists():
            logger.info(f"Cache miss: metadata not found")
            return False

        # Load and verify metadata
        try:
            with open(self.meta_cache_path, "r") as f:
                meta_dict = json.load(f)
            meta = CacheMetadata.from_dict(meta_dict)

            # Check source file hasn't changed
            if not self.data_path.exists():
                logger.info(f"Cache miss: source file deleted")
                return False

            current_mtime = self.data_path.stat().st_mtime
            current_size = self.data_path.stat().st_size

            if abs(meta.source_mtime - current_mtime) > 1:  # 1 second tolerance
                logger.info(f"Cache miss: source file modified (mtime)")
                return False

            if meta.source_size != current_size:
                logger.info(f"Cache miss: source file size changed")
                return False

            # Verify content hash for extra safety
            current_hash = compute_file_hash(self.data_path)
            if meta.source_hash != current_hash:
                logger.info(f"Cache miss: source content changed")
                return False

            # Verify config hash
            if meta.config_hash != self._config_hash:
                logger.info(f"Cache miss: config changed")
                return False

            logger.info(f"Cache hit: {self.cache_name}")
            return True

        except Exception as e:
            logger.warning(f"Cache validation error: {e}")
            return False

    def load(self) -> Tuple[List[Any], Optional[Dict[int, List[int]]]]:
        """
        Load cached data and length index.

        Returns:
            Tuple of (processed_data, length_index)
            length_index may be None if not cached
        """
        if not self.enabled:
            raise RuntimeError("Cache is disabled")

        logger.info(f"Loading from cache: {self.data_cache_path}")

        # Load processed data
        with open(self.data_cache_path, "rb") as f:
            data = pickle.load(f)

        # Load length index if available
        length_index = None
        if self.index_cache_path.exists():
            try:
                with open(self.index_cache_path, "rb") as f:
                    length_index = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load length index: {e}")

        return data, length_index

    def save(
        self,
        data: List[Any],
        length_index: Optional[Dict[int, List[int]]] = None,
        raw_data: Optional[List[Dict]] = None,
    ):
        """
        Save processed data to cache.

        Args:
            data: Processed dataset (list of tuples/items)
            length_index: Optional length-to-indices mapping for cross-sampling
            raw_data: Optional raw data (before processing) for debugging
        """
        if not self.enabled:
            return

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to cache: {self.data_cache_path}")

        # Save processed data
        with open(self.data_cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save length index if provided
        if length_index is not None:
            with open(self.index_cache_path, "wb") as f:
                pickle.dump(length_index, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata
        stat = self.data_path.stat()
        meta = CacheMetadata(
            source_path=str(self.data_path),
            source_mtime=stat.st_mtime,
            source_size=stat.st_size,
            source_hash=self._source_hash,
            config_hash=self._config_hash,
            num_samples=len(data),
            has_length_index=length_index is not None,
        )

        with open(self.meta_cache_path, "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        logger.info(f"Cache saved: {len(data)} samples")

    def invalidate(self):
        """Remove cached files."""
        for path in [self.data_cache_path, self.meta_cache_path, self.index_cache_path]:
            if path.exists():
                path.unlink()
                logger.info(f"Removed cache file: {path}")

    @staticmethod
    def clear_cache_dir(cache_dir: Union[str, Path]):
        """Remove all cache files in a directory."""
        cache_dir = Path(cache_dir)
        if cache_dir.exists():
            for f in cache_dir.glob("*.pkl"):
                f.unlink()
            for f in cache_dir.glob("*.meta.json"):
                f.unlink()
            logger.info(f"Cleared cache directory: {cache_dir}")


def build_length_index(
    data: List[Tuple],
    prompt_token_idx: int = 0,
    bucket_size: int = 64,
) -> Dict[int, List[int]]:
    """
    Build a length-based index for efficient cross-sampling pair matching.

    Groups samples by their prompt token length into buckets for O(1) lookups.

    Args:
        data: List of processed samples (tuples where first element is tokens)
        prompt_token_idx: Index of prompt tokens in the tuple
        bucket_size: Size of length buckets

    Returns:
        Dict mapping bucket_id -> list of sample indices
    """
    length_index: Dict[int, List[int]] = {}

    for idx, item in enumerate(data):
        if isinstance(item, tuple):
            tokens = item[prompt_token_idx]
        else:
            tokens = item

        length = len(tokens) if hasattr(tokens, "__len__") else 0
        bucket = length // bucket_size

        if bucket not in length_index:
            length_index[bucket] = []
        length_index[bucket].append(idx)

    # Log distribution
    total = len(data)
    num_buckets = len(length_index)
    avg_per_bucket = total / num_buckets if num_buckets > 0 else 0
    logger.info(
        f"Length index: {total} samples in {num_buckets} buckets (avg {avg_per_bucket:.1f}/bucket)"
    )

    return length_index


def get_samples_by_length(
    length_index: Dict[int, List[int]],
    target_length: int,
    bucket_size: int = 64,
    max_results: int = 10,
) -> List[int]:
    """
    Get sample indices with similar length.

    Args:
        length_index: Bucket -> indices mapping
        target_length: Desired token length
        bucket_size: Size of length buckets
        max_results: Maximum indices to return

    Returns:
        List of sample indices
    """
    target_bucket = target_length // bucket_size
    results = []

    # Search nearby buckets
    for offset in range(10):  # Check up to 10 buckets away
        for bucket in [target_bucket + offset, target_bucket - offset]:
            if bucket in length_index:
                results.extend(length_index[bucket])
                if len(results) >= max_results:
                    return results[:max_results]

    return results
