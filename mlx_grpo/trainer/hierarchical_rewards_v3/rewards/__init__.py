"""
Reward Level Implementations
============================

Individual reward functions organized by level.

Levels:
    - Foundation: Structure and format requirements
    - Correctness: Factual accuracy and answer matching
    - Quality: Reasoning depth and coherence
    - Polish: Style and presentation

Usage:
    from hierarchical_rewards_v3.rewards import hierarchical_reward, quick_score

    # Get full score with diagnostics
    score, diagnostics = hierarchical_reward(response, expected, question)

    # Quick scoring
    score = quick_score(response, expected, question)

    # Batch scoring for GRPO
    batch_result = batch_hierarchical_reward(responses, expected, question)
"""

from .foundation import compute_foundation_reward
from .correctness import compute_correctness_reward
from .quality import compute_quality_reward
from .polish import compute_polish_reward

from .aggregator import (
    hierarchical_reward,
    batch_hierarchical_reward,
    quick_score,
    detailed_analysis,
    compute_soft_gate,
    ensure_ranking_signal,
    GateState,
)

__all__ = [
    # Individual level rewards
    "compute_foundation_reward",
    "compute_correctness_reward",
    "compute_quality_reward",
    "compute_polish_reward",
    # Main aggregator
    "hierarchical_reward",
    "batch_hierarchical_reward",
    # Convenience functions
    "quick_score",
    "detailed_analysis",
    # Utilities
    "compute_soft_gate",
    "ensure_ranking_signal",
    "GateState",
]
