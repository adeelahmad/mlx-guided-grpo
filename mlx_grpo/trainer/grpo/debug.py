"""Debug and crash tracing utilities for GRPO training.

This module provides debugging tools for MLX training:
- Memory monitoring
- Safe array evaluation with crash tracing
- Iteration tracking

Enable with environment variables:
    DEBUG=1 python train.py           # Basic debug output
    DEBUG=1 DEBUG_MEMORY=1 python ... # Verbose memory tracking

SOLID Principles:
- Single Responsibility: Only handles debugging and tracing
- Open/Closed: Can be extended with new debug modes via environment
"""

from __future__ import annotations

import gc
import os
import sys
import traceback
from datetime import datetime
from typing import Any

import mlx.core as mx
from tqdm import tqdm

__all__ = [
    "DEBUG",
    "DEBUG_MEMORY",
    "mem_stats",
    "dbg",
    "dbg_mem",
    "safe_eval",
    "safe_clear",
    "iter_start",
    "iter_end",
    "get_current_iteration",
    "get_eval_history",
]

# Environment-based debug flags
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
DEBUG_MEMORY = os.environ.get("DEBUG_MEMORY", "").lower() in ("1", "true", "yes")

# Module-level state for crash tracing
_eval_history: list[tuple[str, datetime, float]] = []
_current_iteration: int = 0
_last_checkpoint: str = "init"


def mem_stats() -> tuple[float, float, float]:
    """Get memory statistics.

    Returns:
        Tuple of (active_gb, peak_gb, cache_gb)
    """
    try:
        return (
            mx.metal.get_active_memory() / 1e9,
            mx.metal.get_peak_memory() / 1e9,
            mx.metal.get_cache_memory() / 1e9,
        )
    except Exception:
        return (0.0, 0.0, 0.0)


def dbg(msg: str) -> None:
    """Print debug message if DEBUG enabled."""
    if DEBUG:
        tqdm.write(f"[DBG] {msg}")


def dbg_mem(checkpoint: str) -> None:
    """Log memory at checkpoint if DEBUG_MEMORY enabled."""
    if DEBUG_MEMORY:
        active, peak, cache = mem_stats()
        tqdm.write(
            f"[MEM] {checkpoint}: active={active:.2f}GB peak={peak:.2f}GB cache={cache:.2f}GB"
        )


def safe_eval(*args: Any, checkpoint: str = "unknown") -> None:
    """Evaluate MLX arrays with crash tracing.

    CRITICAL: This is the fix for memory leaks.
    Every lazy MLX operation must be evaluated before mx.clear_cache().

    Args:
        *args: Arrays, dicts, lists, or nn.Module instances to evaluate
        checkpoint: Name of checkpoint for crash tracing
    """
    global _last_checkpoint, _eval_history
    _last_checkpoint = checkpoint

    # Extract arrays from mixed inputs
    arrays: list[mx.array] = []

    def _extract(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, mx.array):
            arrays.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _extract(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _extract(v)
        elif hasattr(obj, "parameters"):
            # nn.Module - extract parameters
            try:
                params = obj.parameters()
                _extract(params)
            except Exception:
                pass

    for arg in args:
        _extract(arg)

    if not arrays:
        return

    active_before = mem_stats()[0] if DEBUG_MEMORY else 0

    try:
        mx.eval(*arrays)

        active_after, peak, _ = mem_stats() if DEBUG_MEMORY else (0, 0, 0)
        _eval_history.append((checkpoint, datetime.now(), active_after))
        if len(_eval_history) > 100:
            _eval_history.pop(0)

        if DEBUG:
            delta = active_after - active_before if DEBUG_MEMORY else 0
            mem_str = f" mem={active_after:.2f}GB (Δ{delta:+.3f})" if DEBUG_MEMORY else ""
            tqdm.write(f"[EVAL] {checkpoint} n={len(arrays)}{mem_str}")

    except Exception as e:
        # CRASH - Print detailed trace
        tqdm.write("\n" + "=" * 80)
        tqdm.write("MLX CRASH DETECTED")
        tqdm.write("=" * 80)
        tqdm.write(f"Checkpoint: {checkpoint}")
        tqdm.write(f"Iteration: {_current_iteration}")
        tqdm.write(f"Time: {datetime.now().isoformat()}")
        active, peak, cache = mem_stats()
        tqdm.write(f"Memory: active={active:.2f}GB peak={peak:.2f}GB cache={cache:.2f}GB")
        tqdm.write(f"Arrays: {len(arrays)}")
        tqdm.write(f"\nError: {type(e).__name__}: {e}")

        tqdm.write("\n--- EVAL HISTORY (last 30) ---")
        for loc, ts, mem in _eval_history[-30:]:
            tqdm.write(f"  [{ts.strftime('%H:%M:%S')}] {loc} ({mem:.2f}GB)")

        tqdm.write("\n--- ARRAY INFO ---")
        for i, arr in enumerate(arrays[:10]):
            try:
                tqdm.write(
                    f"  [{i}] shape={arr.shape} dtype={arr.dtype} size={arr.nbytes / 1e6:.1f}MB"
                )
            except Exception:
                tqdm.write(f"  [{i}] <invalid>")

        tqdm.write("\n--- FULL TRACEBACK ---")
        traceback.print_exc()
        tqdm.write("=" * 80)
        raise


def safe_clear(checkpoint: str = "unknown") -> None:
    """Clear cache with logging."""
    mx.clear_cache()
    gc.collect()
    if DEBUG:
        tqdm.write(f"[CLEAR] {checkpoint}")


def iter_start(iteration: int) -> None:
    """Mark iteration start for debugging."""
    global _current_iteration
    _current_iteration = iteration
    if DEBUG:
        active, peak, _ = mem_stats()
        tqdm.write(f"\n[ITER {iteration}] >>> START active={active:.2f}GB peak={peak:.2f}GB")


def iter_end(iteration: int) -> None:
    """Mark iteration end for debugging."""
    if DEBUG:
        active, peak, cache = mem_stats()
        tqdm.write(
            f"[ITER {iteration}] <<< END active={active:.2f}GB peak={peak:.2f}GB cache={cache:.2f}GB"
        )


def get_current_iteration() -> int:
    """Get the current iteration number."""
    return _current_iteration


def get_eval_history() -> list[tuple[str, datetime, float]]:
    """Get the evaluation history for debugging."""
    return _eval_history.copy()
