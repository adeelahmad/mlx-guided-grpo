"""Checkpoint management for GRPO training.

This module provides checkpoint lifecycle management:
- Automatic cleanup of old checkpoints
- Best checkpoint tracking by metric
- Disk space management

SOLID Principles:
- Single Responsibility: Only handles checkpoint file management
- Open/Closed: Can be extended with different cleanup strategies
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from typing import Optional

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """Manages checkpoint files to prevent disk from filling up.

    Supports two retention strategies:
    - keep_last_n: Keep the N most recent checkpoints
    - keep_best_n: Keep the N best checkpoints by metric

    These strategies can be combined (union of both sets is kept).

    Attributes:
        adapter_dir: Directory containing checkpoint files
        keep_last_n: Number of recent checkpoints to keep (0 = unlimited)
        keep_best_n: Number of best checkpoints to keep (0 = disabled)
        metric_name: Name of the metric for ranking (e.g., "val_loss")
        higher_is_better: Whether higher metric values are better

    Example:
        >>> manager = CheckpointManager(
        ...     adapter_dir=Path("./adapters"),
        ...     keep_last_n=3,
        ...     keep_best_n=2,
        ...     metric_name="val_loss",
        ...     higher_is_better=False,
        ... )
        >>> manager.register_checkpoint(100, Path("./adapters/0000100_adapters.safetensors"), 0.5)
    """

    # Pattern to match checkpoint files: 0000100_adapters.safetensors
    CHECKPOINT_PATTERN = re.compile(r'^(\d{7})_adapters\.safetensors$')

    def __init__(
        self,
        adapter_dir: Path,
        keep_last_n: int = 0,
        keep_best_n: int = 0,
        metric_name: str = "val_loss",
        higher_is_better: bool = False,
    ) -> None:
        """Initialize the checkpoint manager.

        Args:
            adapter_dir: Directory containing checkpoint files
            keep_last_n: Number of recent checkpoints to keep (0 = unlimited)
            keep_best_n: Number of best checkpoints to keep (0 = disabled)
            metric_name: Name of the metric for ranking
            higher_is_better: Whether higher metric values are better
        """
        self.adapter_dir = Path(adapter_dir)
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

        # Track checkpoints: {iteration: (path, metric_value)}
        self.checkpoints: dict[int, tuple[Path, float | None]] = {}

    def register_checkpoint(
        self,
        iteration: int,
        path: Path,
        metric_value: float | None = None,
    ) -> None:
        """Register a new checkpoint and clean up old ones.

        Args:
            iteration: Training iteration number
            path: Path to the checkpoint file
            metric_value: Optional metric value for ranking
        """
        self.checkpoints[iteration] = (path, metric_value)
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove old checkpoints based on keep_last_n and keep_best_n."""
        if not self.checkpoints:
            return

        if self.keep_last_n == 0 and self.keep_best_n == 0:
            return  # Keep all checkpoints

        # Get all iterations sorted
        all_iters = sorted(self.checkpoints.keys())

        # Determine which to keep
        keep_iters: set[int] = set()

        # Keep last N
        if self.keep_last_n > 0:
            keep_iters.update(all_iters[-self.keep_last_n:])

        # Keep best N by metric
        if self.keep_best_n > 0:
            # Filter checkpoints that have metric values
            with_metrics = [
                (it, path, val)
                for it, (path, val) in self.checkpoints.items()
                if val is not None
            ]
            if with_metrics:
                # Sort by metric
                sorted_by_metric = sorted(
                    with_metrics,
                    key=lambda x: x[2],  # type: ignore[arg-type]
                    reverse=self.higher_is_better,
                )
                best_iters = [it for it, _, _ in sorted_by_metric[:self.keep_best_n]]
                keep_iters.update(best_iters)

        # Delete checkpoints not in keep set
        to_delete = set(all_iters) - keep_iters
        for it in to_delete:
            path, _ = self.checkpoints[it]
            if path.exists():
                try:
                    path.unlink()
                    tqdm.write(f"  Deleted old checkpoint: {path.name}")
                except Exception as e:
                    tqdm.write(f"  Warning: Failed to delete {path.name}: {e}")
            del self.checkpoints[it]

    def scan_existing(self) -> None:
        """Scan adapter directory for existing checkpoints.

        This should be called after initialization to discover
        checkpoints from a previous training run.
        """
        if not self.adapter_dir.exists():
            return

        for f in self.adapter_dir.iterdir():
            match = self.CHECKPOINT_PATTERN.match(f.name)
            if match:
                iteration = int(match.group(1))
                self.checkpoints[iteration] = (f, None)

    def get_best_checkpoint(self) -> tuple[int, Path, float] | None:
        """Get the best checkpoint by metric.

        Returns:
            Tuple of (iteration, path, metric_value) or None if no
            checkpoints have metric values.
        """
        with_metrics = [
            (it, path, val)
            for it, (path, val) in self.checkpoints.items()
            if val is not None
        ]
        if not with_metrics:
            return None
        sorted_by_metric = sorted(
            with_metrics,
            key=lambda x: x[2],  # type: ignore[arg-type]
            reverse=self.higher_is_better,
        )
        return sorted_by_metric[0]

    def get_latest_checkpoint(self) -> tuple[int, Path] | None:
        """Get the most recent checkpoint by iteration.

        Returns:
            Tuple of (iteration, path) or None if no checkpoints.
        """
        if not self.checkpoints:
            return None
        latest_iter = max(self.checkpoints.keys())
        path, _ = self.checkpoints[latest_iter]
        return latest_iter, path
