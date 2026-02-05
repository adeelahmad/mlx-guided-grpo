"""
Hierarchical Reward Aggregator
==============================

The main orchestration module that:
1. Computes each reward level
2. Applies soft gating with cascading
3. Integrates anti-gaming penalties
4. Ensures gradient flow
5. Produces final scores optimized for GRPO

FIXED: batch_hierarchical_reward now correctly handles:
- Multiple responses with a single question/answer (original GRPO use case)
- Multiple responses with individual questions/answers per response (batch use case)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.base import (
    AntiGamingResult,
    BatchResult,
    ComponentResult,
    DiagnosticInfo,
    LevelResult,
    RewardResult,
)
from ..core.config import GateConfig, RewardConfig, get_default_config
from .correctness import compute_correctness_reward
from .foundation import compute_foundation_reward
from .polish import compute_polish_reward
from .quality import compute_quality_reward


@dataclass
class GateState:
    """Tracks gate values through the hierarchy."""

    foundation_gate: float = 1.0
    correctness_gate: float = 1.0
    quality_gate: float = 1.0
    polish_gate: float = 1.0
    cumulative_gate: float = 1.0


def sigmoid(x: float, steepness: float = 10.0) -> float:
    """Numerically stable sigmoid function."""
    if x * steepness > 500:
        return 1.0
    if x * steepness < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-steepness * x))


def compute_soft_gate(
    score: float, gate_config: GateConfig, upstream_gate: float = 1.0
) -> Tuple[float, float]:
    """
    Compute soft gate value with guaranteed gradient flow.

    Returns:
        Tuple of (gate_value, gated_score)
    """
    raw_gate = sigmoid(score - gate_config.threshold, gate_config.steepness)
    gate_value = gate_config.floor + (1.0 - gate_config.floor) * raw_gate
    effective_gate = gate_value * upstream_gate
    gated_score = score * effective_gate
    return gate_value, gated_score


def compute_anti_gaming_penalty(
    response: str, level_results: Dict[str, LevelResult], config: RewardConfig
) -> "SimpleAntiGamingResult":
    """Compute anti-gaming penalties based on suspicious patterns."""
    from ..utils.information_theory import calculate_compression_ratio as compression_ratio
    from ..utils.information_theory import calculate_entropy as char_entropy
    from ..utils.structural_analysis import detect_clones
    from ..utils.text_processing import compute_ngram_repetition, unique_token_ratio

    @dataclass
    class SimpleAntiGamingResult:
        penalty: float = 0.0
        flags: List[str] = field(default_factory=list)
        details: Dict[str, Any] = field(default_factory=dict)

    # Handle None/empty response
    if not response:
        return SimpleAntiGamingResult(penalty=0.0, flags=["empty_response"], details={})

    # Ensure response is a string
    if isinstance(response, list):
        response = " ".join(str(r) for r in response if r)
    elif not isinstance(response, str):
        response = str(response)

    penalties = {}
    flags = []
    details = {}

    # 1. Repetition penalty
    trigram_rep = compute_ngram_repetition(response, n=3)
    if trigram_rep > 0.3:
        penalty = min(0.3, (trigram_rep - 0.3) * 0.5)
        penalties["repetition"] = penalty
        flags.append(f"high_trigram_repetition:{trigram_rep:.2f}")
    details["trigram_repetition"] = trigram_rep

    # 2. Clone detection
    clones = detect_clones(response, min_length=30)
    if clones:
        clone_penalty = min(0.25, len(clones) * 0.05)
        penalties["cloning"] = clone_penalty
        flags.append(f"detected_clones:{len(clones)}")
    details["clone_count"] = len(clones) if clones else 0

    # 3. Low entropy penalty
    entropy = char_entropy(response)
    if entropy < 3.0:
        penalty = min(0.2, (3.0 - entropy) * 0.1)
        penalties["low_entropy"] = penalty
        flags.append(f"low_entropy:{entropy:.2f}")
    details["char_entropy"] = entropy

    # 4. Compression ratio check
    comp_ratio = compression_ratio(response)
    if comp_ratio < 0.3:
        penalty = min(0.2, (0.3 - comp_ratio) * 0.5)
        penalties["compressible"] = penalty
        flags.append(f"high_compression:{comp_ratio:.2f}")
    details["compression_ratio"] = comp_ratio

    # 5. Token diversity
    unique_ratio = unique_token_ratio(response)
    if unique_ratio < 0.3:
        penalty = min(0.15, (0.3 - unique_ratio) * 0.3)
        penalties["low_diversity"] = penalty
        flags.append(f"low_token_diversity:{unique_ratio:.2f}")
    details["unique_token_ratio"] = unique_ratio

    # 6. Suspicious score patterns
    foundation_default = LevelResult(
        level="foundation",
        raw_score=0.5,
        gated_score=0.5,
        gate_value=1.0,
        passed_soft_gate=True,
    )
    quality_default = LevelResult(
        level="quality",
        raw_score=0.5,
        gated_score=0.5,
        gate_value=1.0,
        passed_soft_gate=True,
    )
    foundation_score = level_results.get("foundation", foundation_default).raw_score
    quality_score = level_results.get("quality", quality_default).raw_score

    if quality_score > 0.8 and foundation_score < 0.4:
        penalties["score_inversion"] = 0.1
        flags.append("suspicious_score_pattern")

    total_penalty = min(0.5, sum(penalties.values()))

    return SimpleAntiGamingResult(
        penalty=total_penalty,
        flags=flags,
        details={"individual_penalties": penalties, "metrics": details},
    )


def ensure_ranking_signal(scores: List[float], min_spread: float = 0.05) -> List[float]:
    """Ensure scores have sufficient spread for GRPO ranking."""
    if len(scores) < 2:
        return scores

    min_score = min(scores)
    max_score = max(scores)
    spread = max_score - min_score

    if spread < min_spread:
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        adjusted = scores.copy()
        increment = min_spread / len(scores)
        for rank, idx in enumerate(sorted_indices):
            adjusted[idx] = scores[idx] + rank * increment

        adj_min = min(adjusted)
        adj_max = max(adjusted)
        if adj_max > adj_min:
            adjusted = [
                min_score + (s - adj_min) * spread / (adj_max - adj_min + 1e-8) for s in adjusted
            ]
        return adjusted

    return scores


def _ensure_string(value: Any) -> str:
    """Ensure value is a string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None)
    if not isinstance(value, str):
        return str(value)
    return value


def hierarchical_reward(
    response: str,
    expected: str,
    question: str,
    config: Optional[RewardConfig] = None,
    **kwargs,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute hierarchical reward with soft gating.

    This is the main entry point for single-response scoring.
    """
    if config is None:
        config = get_default_config()

    # Ensure inputs are strings
    response = _ensure_string(response)
    expected = _ensure_string(expected)
    question = _ensure_string(question)

    level_results: Dict[str, LevelResult] = {}
    gate_state = GateState()

    # ============================================
    # Level 1: Foundation
    # ============================================
    foundation_result = compute_foundation_reward(
        response=response, expected=expected, question=question, config=config
    )
    foundation_gate, foundation_gated = compute_soft_gate(
        foundation_result.score, config.foundation_gate
    )
    gate_state.foundation_gate = foundation_gate
    gate_state.cumulative_gate *= foundation_gate

    foundation_components = {c.name: c for c in foundation_result.components}
    level_results["foundation"] = LevelResult(
        level="foundation",
        raw_score=foundation_result.score,
        gated_score=foundation_gated,
        gate_value=foundation_gate,
        passed_soft_gate=foundation_gate > 0.5,
        components=foundation_components,
        upstream_gate=1.0,
        diagnostics=foundation_result.details,
    )

    # ============================================
    # Level 2: Correctness
    # ============================================
    correctness_result = compute_correctness_reward(
        response=response, expected=expected, question=question, config=config
    )
    correctness_gate, correctness_gated = compute_soft_gate(
        correctness_result.score, config.correctness_gate, gate_state.foundation_gate
    )
    gate_state.correctness_gate = correctness_gate
    gate_state.cumulative_gate *= correctness_gate

    correctness_components = {c.name: c for c in correctness_result.components}
    level_results["correctness"] = LevelResult(
        level="correctness",
        raw_score=correctness_result.score,
        gated_score=correctness_gated,
        gate_value=correctness_gate,
        passed_soft_gate=correctness_gate > 0.5,
        components=correctness_components,
        upstream_gate=gate_state.foundation_gate,
        diagnostics=correctness_result.details,
    )

    # ============================================
    # Level 3: Quality
    # ============================================
    quality_result = compute_quality_reward(
        response=response, expected=expected, question=question, config=config
    )
    quality_gate, quality_gated = compute_soft_gate(
        quality_result.score, config.quality_gate, gate_state.correctness_gate
    )
    gate_state.quality_gate = quality_gate
    gate_state.cumulative_gate *= quality_gate

    quality_components = {c.name: c for c in quality_result.components}
    level_results["quality"] = LevelResult(
        level="quality",
        raw_score=quality_result.score,
        gated_score=quality_gated,
        gate_value=quality_gate,
        passed_soft_gate=quality_gate > 0.5,
        components=quality_components,
        upstream_gate=gate_state.correctness_gate,
        diagnostics=quality_result.details,
    )

    # ============================================
    # Level 4: Polish
    # ============================================
    polish_result = compute_polish_reward(
        response=response, expected=expected, question=question, config=config
    )
    polish_gate, polish_gated = compute_soft_gate(
        polish_result.score, config.polish_gate, gate_state.quality_gate
    )
    gate_state.polish_gate = polish_gate
    gate_state.cumulative_gate *= polish_gate

    polish_components = {c.name: c for c in polish_result.components}
    level_results["polish"] = LevelResult(
        level="polish",
        raw_score=polish_result.score,
        gated_score=polish_gated,
        gate_value=polish_gate,
        passed_soft_gate=polish_gate > 0.5,
        components=polish_components,
        upstream_gate=gate_state.quality_gate,
        diagnostics=polish_result.details,
    )

    # ============================================
    # Anti-Gaming Check
    # ============================================
    anti_gaming = compute_anti_gaming_penalty(
        response=response, level_results=level_results, config=config
    )

    # ============================================
    # Final Score Computation
    # ============================================
    weighted_sum = sum(
        level_results[level].gated_score * config.level_weights[level]
        for level in ["foundation", "correctness", "quality", "polish"]
    )
    total_weight = sum(config.level_weights.values())
    pre_penalty_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    final_score = pre_penalty_score * (1.0 - anti_gaming.penalty)
    final_score = max(config.min_score, final_score)
    final_score = config.min_score + (config.max_score - config.min_score) * final_score

    # ============================================
    # Build Diagnostics
    # ============================================
    diagnostics = {
        "final_score": final_score,
        "pre_penalty_score": pre_penalty_score,
        "anti_gaming": {
            "penalty": anti_gaming.penalty,
            "flags": anti_gaming.flags,
            "details": anti_gaming.details,
        },
        "gates": {
            "foundation": gate_state.foundation_gate,
            "correctness": gate_state.correctness_gate,
            "quality": gate_state.quality_gate,
            "polish": gate_state.polish_gate,
            "cumulative": gate_state.cumulative_gate,
        },
        "levels": {
            name: {
                "raw_score": result.raw_score,
                "gated_score": result.gated_score,
                "gate_value": result.gate_value,
                "weight": config.level_weights[name],
                "weighted_contribution": result.gated_score * config.level_weights[name],
                "components": (
                    [
                        {"name": c.name, "score": c.raw_score, "weight": c.weight}
                        for c in result.components.values()
                    ]
                    if result.components
                    else []
                ),
            }
            for name, result in level_results.items()
        },
        "config": {
            "level_weights": config.level_weights,
            "gate_thresholds": {
                "foundation": config.foundation_gate.threshold,
                "correctness": config.correctness_gate.threshold,
                "quality": config.quality_gate.threshold,
                "polish": config.polish_gate.threshold,
            },
        },
    }

    return final_score, diagnostics


def batch_hierarchical_reward(
    responses: List[str],
    expected: Union[str, List[str]],
    question: Union[str, List[str]],
    config: Optional[RewardConfig] = None,
    ensure_ranking: bool = True,
    **kwargs,
) -> "SimpleBatchResult":
    """
    Compute hierarchical rewards for a batch of responses.

    FIXED: Now correctly handles both:
    - Single expected/question for all responses (original GRPO use case)
    - List of expected/questions matching responses (batch training use case)

    Args:
        responses: List of model responses
        expected: Expected answer (single str OR list matching responses)
        question: Original question (single str OR list matching responses)
        config: Reward configuration
        ensure_ranking: Whether to ensure score differentiation

    Returns:
        SimpleBatchResult with scores and diagnostics
    """
    if config is None:
        config = get_default_config()

    # Normalize expected to list
    if isinstance(expected, str):
        expected_list = [expected] * len(responses)
    elif isinstance(expected, list):
        if len(expected) != len(responses):
            raise ValueError(
                f"Length mismatch: {len(responses)} responses but {len(expected)} expected answers"
            )
        expected_list = expected
    else:
        expected_list = [str(expected) if expected else ""] * len(responses)

    # Normalize question to list
    if isinstance(question, str):
        question_list = [question] * len(responses)
    elif isinstance(question, list):
        if len(question) != len(responses):
            raise ValueError(
                f"Length mismatch: {len(responses)} responses but {len(question)} questions"
            )
        question_list = question
    else:
        question_list = [str(question) if question else ""] * len(responses)

    scores = []
    all_diagnostics = []

    # Compute individual scores - FIXED: iterate over all three lists
    for i, (response, exp, q) in enumerate(zip(responses, expected_list, question_list)):
        score, diagnostics = hierarchical_reward(
            response=response,
            expected=exp,
            question=q,
            config=config,
            response_index=i,
            **kwargs,
        )
        scores.append(score)
        all_diagnostics.append(diagnostics)

    # Ensure ranking signal if requested
    if ensure_ranking and len(scores) > 1:
        original_scores = scores.copy()
        scores = ensure_ranking_signal(scores, min_spread=0.05)

        for i, diag in enumerate(all_diagnostics):
            diag["ranking_adjustment"] = {
                "original": original_scores[i],
                "adjusted": scores[i],
                "spread_ensured": original_scores != scores,
            }

    # Compute batch statistics
    mean_score = sum(scores) / len(scores) if scores else 0
    std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0

    @dataclass
    class SimpleBatchResult:
        scores: List[float]
        diagnostics: List[Dict[str, Any]]
        batch_stats: Dict[str, Any]

    batch_stats = {
        "mean": mean_score,
        "min": min(scores) if scores else 0,
        "max": max(scores) if scores else 0,
        "spread": max(scores) - min(scores) if scores else 0,
        "std": std_score,
    }

    return SimpleBatchResult(scores=scores, diagnostics=all_diagnostics, batch_stats=batch_stats)


# ============================================
# Convenience functions
# ============================================


def quick_score(response: str, expected: str, question: str = "") -> float:
    """Get just the final score without diagnostics."""
    score, _ = hierarchical_reward(response, expected, question)
    return score


def detailed_analysis(
    response: str, expected: str, question: str = "", verbose: bool = False
) -> Dict[str, Any]:
    """Get comprehensive analysis with formatted output."""
    score, diagnostics = hierarchical_reward(response, expected, question)

    analysis = {
        "score": score,
        "grade": _score_to_grade(score),
        "summary": {
            "foundation": diagnostics["levels"]["foundation"]["raw_score"],
            "correctness": diagnostics["levels"]["correctness"]["raw_score"],
            "quality": diagnostics["levels"]["quality"]["raw_score"],
            "polish": diagnostics["levels"]["polish"]["raw_score"],
        },
        "gates": diagnostics["gates"],
        "anti_gaming": diagnostics["anti_gaming"]["penalty"],
        "issues": diagnostics["anti_gaming"]["flags"],
    }

    if verbose:
        analysis["full_diagnostics"] = diagnostics

    return analysis


def _score_to_grade(score: float) -> str:
    """Convert score to letter grade."""
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"


__all__ = [
    "hierarchical_reward",
    "batch_hierarchical_reward",
    "quick_score",
    "detailed_analysis",
    "compute_soft_gate",
    "ensure_ranking_signal",
    "GateState",
]
