"""
Hierarchical Rewards v3
=======================

A production-grade, multi-hierarchical reward system for GRPO training
with soft gating, anti-gaming mechanisms, and guaranteed gradient flow.

Architecture:
    Four-level hierarchy with cascading soft gates:

    1. Foundation (10%) - Structure and format requirements
       - Think tags, answer sections, completeness
       - Soft threshold: 0.4, floor: 0.15

    2. Correctness (45%) - Factual accuracy
       - Multi-method verification (numerical, exact, semantic, compression)
       - Soft threshold: 0.10, floor: 0.10

    3. Quality (30%) - Reasoning depth and coherence
       - Structural complexity, information density, reference chains
       - Soft threshold: 0.08, floor: 0.08

    4. Polish (15%) - Style and presentation
       - Format, readability, efficiency, consistency
       - Soft threshold: 0.05, floor: 0.05

Key Features:
    - Soft gating: Sigmoid-based gates with floors ensure gradients always flow
    - Anti-gaming: Information-theoretic measures resist exploitation
    - Calibrated for constraints: Optimized for 450 token limit, group size 2

Usage:
    Basic scoring:

        from hierarchical_rewards_v3 import hierarchical_reward

        score, diagnostics = hierarchical_reward(
            response="<think>...</think><answer>42</answer>",
            expected="42",
            question="What is 6 * 7?"
        )

    Batch scoring for GRPO:

        from hierarchical_rewards_v3 import batch_hierarchical_reward

        result = batch_hierarchical_reward(
            responses=[response1, response2],
            expected="42",
            question="What is 6 * 7?"
        )
        scores = result.scores

    Quick scoring:

        from hierarchical_rewards_v3 import quick_score
        score = quick_score(response, expected, question)

    Detailed analysis:

        from hierarchical_rewards_v3 import detailed_analysis
        analysis = detailed_analysis(response, expected, question, verbose=True)

Configuration:
    Use RewardConfig to customize thresholds, weights, and gates:

        from hierarchical_rewards_v3 import RewardConfig, GateConfig

        config = RewardConfig(
            level_weights={
                'foundation': 0.10,
                'correctness': 0.45,
                'quality': 0.30,
                'polish': 0.15
            },
            foundation_gate=GateConfig(threshold=0.4, floor=0.15, steepness=10.0)
        )

        score, diagnostics = hierarchical_reward(response, expected, question, config=config)

Author: Hierarchical Rewards System
Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "Hierarchical Rewards System"

# Core data structures
from .core.base import (
    AntiGamingResult,
    BatchResult,
    ComponentResult,
    DiagnosticInfo,
    LevelResult,
    RewardResult,
)

# Core configuration
from .core.config import (
    GateConfig,
    RewardConfig,
    get_default_config,
)

# Registry
from .core.registry import get_reward_function as get_reward
from .core.registry import list_reward_functions as list_rewards
from .core.registry import register_reward_function as register_reward

# Main reward functions
from .rewards import (  # Main aggregator; Individual level rewards; Utilities
    GateState,
    batch_hierarchical_reward,
    compute_correctness_reward,
    compute_foundation_reward,
    compute_polish_reward,
    compute_quality_reward,
    compute_soft_gate,
    detailed_analysis,
    ensure_ranking_signal,
    hierarchical_reward,
    quick_score,
)

# Utility modules (for advanced users)
from .utils import information_theory, structural_analysis, text_processing

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Configuration
    "RewardConfig",
    "GateConfig",
    "get_default_config",
    # Data structures
    "RewardResult",
    "LevelResult",
    "ComponentResult",
    "DiagnosticInfo",
    "AntiGamingResult",
    "BatchResult",
    # Registry
    "register_reward",
    "get_reward",
    "list_rewards",
    # Main functions
    "hierarchical_reward",
    "batch_hierarchical_reward",
    "quick_score",
    "detailed_analysis",
    # Level rewards
    "compute_foundation_reward",
    "compute_correctness_reward",
    "compute_quality_reward",
    "compute_polish_reward",
    # Utilities
    "compute_soft_gate",
    "ensure_ranking_signal",
    "GateState",
    # Utility modules
    "text_processing",
    "information_theory",
    "structural_analysis",
]
