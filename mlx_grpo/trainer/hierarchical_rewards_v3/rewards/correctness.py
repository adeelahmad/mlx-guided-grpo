"""
Correctness Level Rewards
=========================

Level 2: Factual accuracy and answer matching.

This level evaluates:
    - Answer matches expected answer
    - Numerical accuracy
    - Semantic similarity
    - Key information presence

Design Philosophy:
    - Multiple verification methods (not just string match)
    - Tolerance for format differences
    - Information-theoretic similarity (resists gaming)
    - Partial credit for partial correctness
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.base import LevelResult
from ..core.config import get_config
from ..utils.information_theory import (
    compression_based_similarity,
    normalized_compression_distance,
)
from ..utils.text_processing import (
    calculate_text_similarity,
    extract_answer_content,
    extract_numbers,
    extract_sections,
    fuzzy_match,
    normalize_number,
    normalize_text,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================


def extract_final_answer(completion: str) -> str:
    """
    Extract the final answer from completion.

    Tries multiple methods:
        1. Answer section after </think>
        2. Boxed answer
        3. Explicit answer markers
        4. Last significant line
    """
    if not completion:
        return ""

    # First try: answer section
    _, answer_section = extract_sections(completion)

    if answer_section:
        # Extract content from answer section
        answer = extract_answer_content(answer_section)
        if answer:
            return answer

    # Second try: extract from whole completion
    return extract_answer_content(completion)


# ============================================================================
# NUMERICAL MATCHING
# ============================================================================


def numerical_match_reward(
    completion: str,
    expected: str,
    tolerance: float = 0.02,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if numerical answer matches expected.

    Handles:
        - Exact matches
        - Approximate matches within tolerance
        - Different formats (fractions, percentages, etc.)

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "completion_numbers": [],
        "expected_numbers": [],
        "matches": [],
        "best_match_error": None,
    }

    # Extract answer
    answer = extract_final_answer(completion)

    # Extract numbers from both
    comp_numbers = extract_numbers(answer)
    exp_numbers = extract_numbers(expected)

    diagnostics["completion_numbers"] = comp_numbers[:5]
    diagnostics["expected_numbers"] = exp_numbers[:5]

    if not comp_numbers or not exp_numbers:
        return 0.0, diagnostics

    # Find best matches
    best_score = 0.0
    best_error = float("inf")

    for exp_num in exp_numbers:
        for comp_num in comp_numbers:
            if exp_num == 0:
                if comp_num == 0:
                    error = 0.0
                else:
                    error = 1.0
            else:
                error = abs(comp_num - exp_num) / abs(exp_num)

            if error < best_error:
                best_error = error

            # Score based on error
            if error == 0:
                score = 1.0
            elif error <= tolerance:
                score = 0.95
            elif error <= tolerance * 2:
                score = 0.80
            elif error <= tolerance * 5:
                score = 0.50
            else:
                score = max(0, 1 - error)

            if score > best_score:
                best_score = score
                diagnostics["matches"].append(
                    {
                        "completion": comp_num,
                        "expected": exp_num,
                        "error": error,
                        "score": score,
                    }
                )

    diagnostics["best_match_error"] = best_error

    return best_score, diagnostics


def exact_match_reward(
    completion: str,
    expected: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check for exact string match (normalized).

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "extracted_answer": "",
        "normalized_answer": "",
        "normalized_expected": "",
        "exact_match": False,
    }

    # Extract answer
    answer = extract_final_answer(completion)
    diagnostics["extracted_answer"] = answer[:100]

    # Normalize both
    norm_answer = normalize_text(answer, remove_punctuation=True)
    norm_expected = normalize_text(expected, remove_punctuation=True)

    diagnostics["normalized_answer"] = norm_answer[:100]
    diagnostics["normalized_expected"] = norm_expected[:100]

    # Check exact match
    if norm_answer == norm_expected:
        diagnostics["exact_match"] = True
        return 1.0, diagnostics

    # Check if expected is in answer
    if norm_expected in norm_answer:
        diagnostics["expected_in_answer"] = True
        return 0.9, diagnostics

    # Check if answer is in expected (partial)
    if norm_answer in norm_expected:
        diagnostics["answer_in_expected"] = True
        ratio = len(norm_answer) / max(1, len(norm_expected))
        return 0.7 * ratio, diagnostics

    return 0.0, diagnostics


# ============================================================================
# SEMANTIC SIMILARITY
# ============================================================================


def semantic_similarity_reward(
    completion: str,
    expected: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute semantic similarity between answer and expected.

    Uses multiple methods and takes best score.

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "fuzzy_score": 0.0,
        "jaccard_score": 0.0,
        "compression_score": 0.0,
        "combined_score": 0.0,
    }

    # Extract answer
    answer = extract_final_answer(completion)

    if not answer or not expected:
        return 0.0, diagnostics

    # Method 1: Fuzzy matching
    fuzzy_score = fuzzy_match(answer, expected)
    diagnostics["fuzzy_score"] = fuzzy_score

    # Method 2: Jaccard similarity
    jaccard_score = calculate_text_similarity(answer, expected, method="jaccard")
    diagnostics["jaccard_score"] = jaccard_score

    # Method 3: Compression-based (resists keyword stuffing)
    compression_score = compression_based_similarity(answer, expected)
    diagnostics["compression_score"] = compression_score

    # Combined score (weighted by reliability)
    # Compression is most resistant to gaming
    combined = 0.30 * fuzzy_score + 0.25 * jaccard_score + 0.45 * compression_score

    diagnostics["combined_score"] = combined

    return combined, diagnostics


# ============================================================================
# FACTUAL ACCURACY
# ============================================================================


def factual_accuracy_reward(
    completion: str,
    expected: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Comprehensive factual accuracy check.

    Combines:
        - Numerical matching
        - Exact string matching
        - Semantic similarity

    Returns:
        (score, diagnostics)
    """
    config = get_config()

    diagnostics = {
        "numerical": {},
        "exact": {},
        "semantic": {},
        "component_scores": {},
    }

    # Run all checks
    num_score, num_diag = numerical_match_reward(completion, expected, config.numerical_tolerance)
    diagnostics["numerical"] = num_diag
    diagnostics["component_scores"]["numerical"] = num_score

    exact_score, exact_diag = exact_match_reward(completion, expected)
    diagnostics["exact"] = exact_diag
    diagnostics["component_scores"]["exact"] = exact_score

    sem_score, sem_diag = semantic_similarity_reward(completion, expected)
    diagnostics["semantic"] = sem_diag
    diagnostics["component_scores"]["semantic"] = sem_score

    # Take best method (different questions favor different methods)
    best_score = max(num_score, exact_score, sem_score)

    # Also compute weighted average (for partial credit scenarios)
    weighted_score = (
        config.factual_weight * num_score
        + config.answer_match_weight * exact_score
        + config.semantic_weight * sem_score
    ) / (config.factual_weight + config.answer_match_weight + config.semantic_weight)

    # Final score: blend of best and weighted
    # This rewards strong performance on any single metric
    # while still valuing overall consistency
    final_score = 0.6 * best_score + 0.4 * weighted_score

    diagnostics["best_score"] = best_score
    diagnostics["weighted_score"] = weighted_score
    diagnostics["final_score"] = final_score

    return final_score, diagnostics


# ============================================================================
# ANSWER VERIFICATION
# ============================================================================


def answer_verification_reward(
    completion: str,
    expected: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Verify that answer section contains the expected answer.

    Specifically checks the answer section, not the whole completion.

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "answer_extracted": "",
        "answer_contains_expected": False,
        "answer_section_length": 0,
    }

    # Extract answer section only
    _, answer_section = extract_sections(completion)

    if not answer_section:
        return 0.0, diagnostics

    diagnostics["answer_section_length"] = len(answer_section)

    # Extract the actual answer
    answer = extract_answer_content(answer_section)
    diagnostics["answer_extracted"] = answer[:100]

    if not answer:
        return 0.0, diagnostics

    # Check if expected is present
    norm_answer = normalize_text(answer)
    norm_expected = normalize_text(expected)

    if norm_expected in norm_answer:
        diagnostics["answer_contains_expected"] = True
        return 1.0, diagnostics

    # Check fuzzy match
    similarity = fuzzy_match(answer, expected)

    if similarity > 0.9:
        return 0.95, diagnostics
    elif similarity > 0.7:
        return similarity, diagnostics

    return similarity * 0.8, diagnostics


# ============================================================================
# KEYWORD COVERAGE
# ============================================================================


def keyword_coverage_reward(
    completion: str,
    expected: str,
    important_words: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if completion covers important keywords from expected.

    Uses information-theoretic weighting to avoid gaming.

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "keywords": [],
        "found": [],
        "coverage": 0.0,
    }

    if not completion or not expected:
        return 0.0, diagnostics

    # Get important words
    if important_words is None:
        # Extract significant words (not stopwords, length > 3)
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "this",
            "that",
            "these",
            "those",
            "it",
            "and",
            "but",
            "or",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
        }

        words = normalize_text(expected).split()
        important_words = [w for w in words if len(w) > 3 and w not in stopwords]

    diagnostics["keywords"] = important_words[:10]

    if not important_words:
        return 0.5, diagnostics  # No keywords to check

    # Check coverage in completion
    completion_norm = normalize_text(completion)
    found = [w for w in important_words if w in completion_norm]

    diagnostics["found"] = found

    coverage = len(found) / len(important_words)
    diagnostics["coverage"] = coverage

    return coverage, diagnostics


# ============================================================================
# COMBINED CORRECTNESS
# ============================================================================


def create_correctness_result(
    completion: str,
    expected: str,
    gate_config: Optional[Any] = None,
    upstream_gate: float = 1.0,
) -> LevelResult:
    """
    Create complete LevelResult for correctness level.
    """
    config = get_config()
    gate = gate_config or config.correctness_gate

    # Get component scores
    factual_score, factual_diag = factual_accuracy_reward(completion, expected)
    semantic_score, semantic_diag = semantic_similarity_reward(completion, expected)
    verification_score, verify_diag = answer_verification_reward(completion, expected)
    keyword_score, keyword_diag = keyword_coverage_reward(completion, expected)

    # Compute raw level score
    weights = {
        "factual": config.factual_weight,
        "semantic": config.semantic_weight,
        "verification": config.answer_match_weight,
        "keyword": 0.10,
    }
    weight_sum = sum(weights.values())

    raw_score = (
        weights["factual"] * factual_score
        + weights["semantic"] * semantic_score
        + weights["verification"] * verification_score
        + weights["keyword"] * keyword_score
    ) / weight_sum

    # Apply soft gate
    gate_value = gate.compute_gate(raw_score)
    gated_score = gate.apply_gate(raw_score, upstream_gate)

    # Create result
    result = LevelResult(
        level="correctness",
        raw_score=raw_score,
        gated_score=gated_score,
        gate_value=gate_value,
        passed_soft_gate=gate_value > 0.5,
        upstream_gate=upstream_gate,
        diagnostics={
            "factual": factual_diag,
            "semantic": semantic_diag,
            "verification": verify_diag,
            "keyword": keyword_diag,
        },
    )

    # Add components
    result.add_component("factual", factual_score, weights["factual"], factual_diag)
    result.add_component("semantic", semantic_score, weights["semantic"], semantic_diag)
    result.add_component("verification", verification_score, weights["verification"], verify_diag)
    result.add_component("keyword", keyword_score, weights["keyword"], keyword_diag)

    return result


def compute_correctness_reward(
    response: str, expected: str, question: str, config: Any, **kwargs
) -> "RewardResult":
    """
    Compute correctness reward with RewardResult interface.

    Args:
        response: Model response
        expected: Expected answer
        question: Original question (used for context)
        config: RewardConfig
        **kwargs: Additional arguments

    Returns:
        RewardResult with score and components
    """
    from ..core.base import ComponentResult, RewardResult
    from ..core.config import get_config

    cfg = config if config else get_config()

    # Get component scores
    factual_score, factual_diag = factual_accuracy_reward(response, expected)
    semantic_score, semantic_diag = semantic_similarity_reward(response, expected)
    verification_score, verify_diag = answer_verification_reward(response, expected)
    keyword_score, keyword_diag = keyword_coverage_reward(response, expected)

    # Compute raw level score
    weights = {
        "factual": cfg.factual_weight,
        "semantic": cfg.semantic_weight,
        "verification": cfg.answer_match_weight,
        "keyword": 0.10,
    }
    weight_sum = sum(weights.values())

    raw_score = (
        weights["factual"] * factual_score
        + weights["semantic"] * semantic_score
        + weights["verification"] * verification_score
        + weights["keyword"] * keyword_score
    ) / weight_sum

    # Build components
    components = [
        ComponentResult(
            name="factual",
            raw_score=factual_score,
            weight=weights["factual"],
            weighted_score=factual_score * weights["factual"],
            diagnostics=factual_diag,
        ),
        ComponentResult(
            name="semantic",
            raw_score=semantic_score,
            weight=weights["semantic"],
            weighted_score=semantic_score * weights["semantic"],
            diagnostics=semantic_diag,
        ),
        ComponentResult(
            name="verification",
            raw_score=verification_score,
            weight=weights["verification"],
            weighted_score=verification_score * weights["verification"],
            diagnostics=verify_diag,
        ),
        ComponentResult(
            name="keyword",
            raw_score=keyword_score,
            weight=weights["keyword"],
            weighted_score=keyword_score * weights["keyword"],
            diagnostics=keyword_diag,
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
            "factual": factual_diag,
            "semantic": semantic_diag,
            "verification": verify_diag,
            "keyword": keyword_diag,
        },
    )
