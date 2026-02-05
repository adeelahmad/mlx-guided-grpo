"""
Quality Level Rewards
=====================

Level 3: Reasoning quality, efficiency, and coherence.

This level evaluates:
    - Reasoning depth and structure
    - Step clarity and progression
    - Efficiency (concise but complete)
    - Coherence (ideas connect logically)

Design Philosophy:
    - Reward genuine complexity, not superficial markers
    - Use structural analysis from code analysis
    - Information-theoretic measures resist gaming
    - Balance between depth and efficiency
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ..core.base import LevelResult
from ..core.config import get_config
from ..utils.information_theory import (
    calculate_content_quality_score,
    calculate_information_density,
    calculate_redundancy,
)
from ..utils.structural_analysis import (
    analyze_reasoning_structure,
    analyze_reference_chain,
    calculate_cognitive_complexity,
    calculate_cyclomatic_complexity,
    calculate_structural_quality,
)
from ..utils.text_processing import (
    extract_sections,
    get_unique_token_ratio,
    normalize_text,
)

logger = logging.getLogger(__name__)


# ============================================================================
# REASONING QUALITY
# ============================================================================


def reasoning_quality_reward(completion: str) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate quality of reasoning in completion.

    Uses structural analysis to assess:
        - Step count and clarity
        - Reasoning depth
        - Branching (handles alternatives)
        - Presence of premises and conclusion

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "structure": {},
        "cognitive_complexity": 0.0,
        "structural_quality": 0.0,
    }

    if not completion:
        return 0.0, diagnostics

    # Extract thinking section
    thinking, _ = extract_sections(completion)

    # Analyze on thinking section primarily, fall back to whole completion
    text_to_analyze = thinking if thinking else completion

    # Structural analysis
    structure = analyze_reasoning_structure(text_to_analyze)
    diagnostics["structure"] = structure

    # Cognitive complexity
    complexity = calculate_cognitive_complexity(text_to_analyze)
    diagnostics["cognitive_complexity"] = complexity

    # Structural quality score
    struct_score, struct_details = calculate_structural_quality(text_to_analyze)
    diagnostics["structural_quality"] = struct_score
    diagnostics["structural_details"] = struct_details

    # Combine into final score
    # We want:
    # - Meaningful step count (not too few, not too many)
    # - Reasonable depth
    # - Some branching for complex problems
    # - Clear conclusion

    # Step count score (target: 3-7 steps)
    step_count = structure["step_count"]
    if step_count == 0:
        step_score = 0.0
    elif step_count < 2:
        step_score = 0.3 * step_count
    elif step_count <= 7:
        step_score = 0.5 + 0.5 * min(1.0, step_count / 5)
    else:
        # Too many steps - might be padding
        step_score = max(0.5, 1.0 - (step_count - 7) * 0.05)

    # Depth score (target: 1-4 levels)
    depth = structure["depth"]
    if depth == 0:
        depth_score = 0.0
    elif depth <= 3:
        depth_score = 0.4 + 0.2 * depth
    else:
        depth_score = max(0.7, 1.0 - (depth - 3) * 0.1)

    # Branch score (some branching is good)
    branches = structure["branches"]
    if branches == 0:
        branch_score = 0.5  # Linear is OK but not ideal
    elif branches <= 3:
        branch_score = 0.6 + 0.15 * branches
    else:
        branch_score = max(0.7, 1.0 - (branches - 3) * 0.05)

    # Conclusion score
    conclusion_score = 1.0 if structure["has_conclusion"] else 0.4
    premise_score = 1.0 if structure["has_premises"] else 0.5

    # Combine
    final_score = (
        0.25 * step_score
        + 0.20 * depth_score
        + 0.15 * branch_score
        + 0.15 * conclusion_score
        + 0.10 * premise_score
        + 0.15 * complexity
    )

    diagnostics["component_scores"] = {
        "steps": step_score,
        "depth": depth_score,
        "branches": branch_score,
        "conclusion": conclusion_score,
        "premises": premise_score,
        "complexity": complexity,
    }

    return final_score, diagnostics


# ============================================================================
# EFFICIENCY
# ============================================================================


def efficiency_reward(
    completion: str,
    expected: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate efficiency of completion.

    Good efficiency means:
        - Concise but complete
        - High information density
        - Low redundancy
        - Appropriate length for problem complexity

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "total_length": 0,
        "thinking_length": 0,
        "answer_length": 0,
        "information_density": 0.0,
        "redundancy": 0.0,
        "unique_ratio": 0.0,
    }

    if not completion:
        return 0.0, diagnostics

    # Length analysis
    thinking, answer = extract_sections(completion)

    total_words = len(completion.split())
    thinking_words = len(thinking.split()) if thinking else 0
    answer_words = len(answer.split()) if answer else 0

    diagnostics["total_length"] = total_words
    diagnostics["thinking_length"] = thinking_words
    diagnostics["answer_length"] = answer_words

    # Information density (higher = better)
    info_density = calculate_information_density(thinking if thinking else completion)
    diagnostics["information_density"] = info_density

    # Redundancy (lower = better)
    redundancy = calculate_redundancy(thinking if thinking else completion)
    diagnostics["redundancy"] = redundancy

    # Unique token ratio
    unique_ratio = get_unique_token_ratio(completion)
    diagnostics["unique_ratio"] = unique_ratio

    # Estimate problem complexity from expected answer
    expected_complexity = len(expected.split()) if expected else 10

    # Target length based on problem complexity
    # Simple problem (short answer): ~50-100 words thinking
    # Complex problem (long answer): ~100-200 words thinking
    target_thinking = min(200, max(50, expected_complexity * 10))

    # Length efficiency score
    if thinking_words == 0:
        length_score = 0.2
    elif thinking_words < target_thinking * 0.5:
        # Too short
        length_score = 0.3 + 0.4 * (thinking_words / target_thinking)
    elif thinking_words <= target_thinking * 1.5:
        # Good range
        length_score = 0.8 + 0.2 * (1 - abs(thinking_words - target_thinking) / target_thinking)
    else:
        # Too long
        excess = (thinking_words - target_thinking * 1.5) / target_thinking
        length_score = max(0.3, 0.8 - excess * 0.2)

    # Information density score
    # Target: 0.5-0.8 (not too low/repetitive, not too high/random)
    if info_density < 0.3:
        density_score = info_density / 0.3 * 0.5
    elif info_density <= 0.8:
        density_score = 0.5 + 0.5 * (info_density - 0.3) / 0.5
    else:
        density_score = max(0.7, 1.0 - (info_density - 0.8) * 0.5)

    # Redundancy score (lower redundancy = higher score)
    redundancy_score = max(0, 1.0 - redundancy)

    # Combine scores
    final_score = 0.35 * length_score + 0.35 * density_score + 0.30 * redundancy_score

    diagnostics["component_scores"] = {
        "length": length_score,
        "density": density_score,
        "redundancy": redundancy_score,
    }

    return final_score, diagnostics


# ============================================================================
# COHERENCE
# ============================================================================


def coherence_reward(completion: str) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate coherence of reasoning.

    Good coherence means:
        - Ideas connect logically
        - References to previous statements
        - Smooth transitions
        - No contradictions

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "reference_chain": {},
        "transition_quality": 0.0,
        "content_quality": 0.0,
    }

    if not completion:
        return 0.0, diagnostics

    # Extract thinking
    thinking, _ = extract_sections(completion)
    text = thinking if thinking else completion

    # Reference chain analysis
    ref_analysis = analyze_reference_chain(text)
    diagnostics["reference_chain"] = ref_analysis

    # Content quality (information-theoretic)
    content_score, content_metrics = calculate_content_quality_score(text)
    diagnostics["content_quality"] = content_score
    diagnostics["content_metrics"] = content_metrics

    # Transition analysis
    transition_score = analyze_transitions(text)
    diagnostics["transition_quality"] = transition_score

    # Combine scores
    final_score = (
        0.35 * ref_analysis["coherence_score"] + 0.35 * content_score + 0.30 * transition_score
    )

    diagnostics["component_scores"] = {
        "reference": ref_analysis["coherence_score"],
        "content": content_score,
        "transition": transition_score,
    }

    return final_score, diagnostics


def analyze_transitions(text: str) -> float:
    """
    Analyze transition quality between sentences/paragraphs.

    Returns:
        Score in [0, 1]
    """
    if not text:
        return 0.0

    import re

    # Split into sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return 0.5  # Single sentence - neutral

    # Transition words/phrases
    good_transitions = [
        r"\b(therefore|thus|hence|so|consequently)\b",
        r"\b(first|second|third|next|then|finally)\b",
        r"\b(because|since|as|given)\b",
        r"\b(however|but|although|despite)\b",
        r"\b(additionally|furthermore|moreover|also)\b",
        r"\b(this|that|these|those)\s+(means?|implies?|shows?|suggests?)\b",
        r"\b(in other words|that is|namely)\b",
    ]

    # Count transitions
    transition_count = 0
    for sentence in sentences[1:]:  # Skip first sentence
        for pattern in good_transitions:
            if re.search(pattern, sentence, re.IGNORECASE):
                transition_count += 1
                break

    # Score based on transition density
    # Target: ~50-80% of sentences have transitions
    density = transition_count / max(1, len(sentences) - 1)

    if density < 0.2:
        score = 0.3 + density * 1.5
    elif density <= 0.8:
        score = 0.6 + 0.4 * (density - 0.2) / 0.6
    else:
        # Too many transitions might be templated
        score = max(0.7, 1.0 - (density - 0.8))

    return score


# ============================================================================
# DEPTH REWARD
# ============================================================================


def depth_reward(completion: str) -> Tuple[float, Dict[str, Any]]:
    """
    Specifically evaluate reasoning depth.

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "cyclomatic": 0,
        "reasoning_depth": 0,
        "detail_level": 0.0,
    }

    if not completion:
        return 0.0, diagnostics

    thinking, _ = extract_sections(completion)
    text = thinking if thinking else completion

    # Cyclomatic complexity
    cyclomatic = calculate_cyclomatic_complexity(text)
    diagnostics["cyclomatic"] = cyclomatic

    # Reasoning depth
    structure = analyze_reasoning_structure(text)
    diagnostics["reasoning_depth"] = structure["depth"]

    # Detail level (based on explanation patterns)
    detail_score = analyze_detail_level(text)
    diagnostics["detail_level"] = detail_score

    # Combine
    # Cyclomatic: target 2-5
    if cyclomatic < 2:
        cyc_score = cyclomatic / 2
    elif cyclomatic <= 6:
        cyc_score = 0.7 + 0.3 * min(1, (cyclomatic - 2) / 4)
    else:
        cyc_score = max(0.5, 1.0 - (cyclomatic - 6) * 0.1)

    # Depth: target 2-4
    depth = structure["depth"]
    if depth < 2:
        depth_score = depth / 2 * 0.5
    elif depth <= 4:
        depth_score = 0.5 + 0.5 * (depth - 1) / 3
    else:
        depth_score = max(0.6, 1.0 - (depth - 4) * 0.1)

    final_score = 0.35 * cyc_score + 0.35 * depth_score + 0.30 * detail_score

    diagnostics["component_scores"] = {
        "cyclomatic": cyc_score,
        "depth": depth_score,
        "detail": detail_score,
    }

    return final_score, diagnostics


def analyze_detail_level(text: str) -> float:
    """
    Analyze level of detail in explanations.

    Returns:
        Score in [0, 1]
    """
    if not text:
        return 0.0

    import re

    # Patterns indicating detailed explanation
    detail_patterns = [
        (r"\b(because|since|as)\b.*\b(this|that|it)\b", 1.0),  # Causal explanation
        (r"\b(for example|e\.g\.|such as)\b", 0.8),  # Examples
        (r"\b(means that|implies that|indicates)\b", 0.8),  # Implications
        (r"\b(let me|I will|we can|we need to)\b", 0.6),  # Process description
        (r"\b(specifically|in particular|notably)\b", 0.7),  # Specificity
        (r"\b\d+[.,:]\d+\b", 0.5),  # Precise numbers
    ]

    total_score = 0.0
    max_score = 0.0

    for pattern, weight in detail_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            total_score += weight * min(3, len(matches)) / 3
        max_score += weight

    if max_score == 0:
        return 0.0

    return total_score / max_score


# ============================================================================
# COMBINED QUALITY
# ============================================================================


def create_quality_result(
    completion: str,
    expected: str,
    gate_config: Optional[Any] = None,
    upstream_gate: float = 1.0,
) -> LevelResult:
    """
    Create complete LevelResult for quality level.
    """
    config = get_config()
    gate = gate_config or config.quality_gate

    # Get component scores
    reasoning_score, reasoning_diag = reasoning_quality_reward(completion)
    efficiency_score, efficiency_diag = efficiency_reward(completion, expected)
    coherence_score, coherence_diag = coherence_reward(completion)
    depth_score, depth_diag = depth_reward(completion)

    # Compute raw level score
    weights = {
        "reasoning": config.reasoning_depth_weight,
        "efficiency": config.efficiency_weight,
        "coherence": config.reasoning_coherence_weight,
        "depth": config.step_clarity_weight,
    }
    weight_sum = sum(weights.values())

    raw_score = (
        weights["reasoning"] * reasoning_score
        + weights["efficiency"] * efficiency_score
        + weights["coherence"] * coherence_score
        + weights["depth"] * depth_score
    ) / weight_sum

    # Apply soft gate
    gate_value = gate.compute_gate(raw_score)
    gated_score = gate.apply_gate(raw_score, upstream_gate)

    # Create result
    result = LevelResult(
        level="quality",
        raw_score=raw_score,
        gated_score=gated_score,
        gate_value=gate_value,
        passed_soft_gate=gate_value > 0.5,
        upstream_gate=upstream_gate,
        diagnostics={
            "reasoning": reasoning_diag,
            "efficiency": efficiency_diag,
            "coherence": coherence_diag,
            "depth": depth_diag,
        },
    )

    # Add components
    result.add_component("reasoning", reasoning_score, weights["reasoning"], reasoning_diag)
    result.add_component("efficiency", efficiency_score, weights["efficiency"], efficiency_diag)
    result.add_component("coherence", coherence_score, weights["coherence"], coherence_diag)
    result.add_component("depth", depth_score, weights["depth"], depth_diag)

    return result


def compute_quality_reward(
    response: str, expected: str, question: str, config: Any, **kwargs
) -> "RewardResult":
    """
    Compute quality reward with RewardResult interface.

    Args:
        response: Model response
        expected: Expected answer
        question: Original question
        config: RewardConfig
        **kwargs: Additional arguments

    Returns:
        RewardResult with score and components
    """
    from ..core.base import ComponentResult, RewardResult
    from ..core.config import get_config

    cfg = config if config else get_config()

    # Get component scores
    reasoning_score, reasoning_diag = reasoning_quality_reward(response)
    efficiency_score, efficiency_diag = efficiency_reward(response, expected)
    coherence_score, coherence_diag = coherence_reward(response)
    depth_score, depth_diag = depth_reward(response)

    # Compute raw level score
    weights = {
        "reasoning": cfg.reasoning_depth_weight,
        "efficiency": cfg.efficiency_weight,
        "coherence": cfg.reasoning_coherence_weight,
        "depth": cfg.step_clarity_weight,
    }
    weight_sum = sum(weights.values())

    raw_score = (
        weights["reasoning"] * reasoning_score
        + weights["efficiency"] * efficiency_score
        + weights["coherence"] * coherence_score
        + weights["depth"] * depth_score
    ) / weight_sum

    # Build components
    components = [
        ComponentResult(
            name="reasoning",
            raw_score=reasoning_score,
            weight=weights["reasoning"],
            weighted_score=reasoning_score * weights["reasoning"],
            diagnostics=reasoning_diag,
        ),
        ComponentResult(
            name="efficiency",
            raw_score=efficiency_score,
            weight=weights["efficiency"],
            weighted_score=efficiency_score * weights["efficiency"],
            diagnostics=efficiency_diag,
        ),
        ComponentResult(
            name="coherence",
            raw_score=coherence_score,
            weight=weights["coherence"],
            weighted_score=coherence_score * weights["coherence"],
            diagnostics=coherence_diag,
        ),
        ComponentResult(
            name="depth",
            raw_score=depth_score,
            weight=weights["depth"],
            weighted_score=depth_score * weights["depth"],
            diagnostics=depth_diag,
        ),
    ]

    # Define simple result class for inter-level communication
    from dataclasses import dataclass
    from dataclasses import field as dataclass_field
    from typing import Any as AnyType
    from typing import Dict as DictType
    from typing import List as ListType

    @dataclass
    class SimpleRewardResult:
        """Simple reward result for inter-level communication."""

        score: float
        components: ListType[ComponentResult] = dataclass_field(default_factory=list)
        details: DictType[str, AnyType] = dataclass_field(default_factory=dict)

    return SimpleRewardResult(
        score=raw_score,
        components=components,
        details={
            "reasoning": reasoning_diag,
            "efficiency": efficiency_diag,
            "coherence": coherence_diag,
            "depth": depth_diag,
        },
    )
