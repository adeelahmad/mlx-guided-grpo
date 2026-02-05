"""
Utility Functions
=================

Text processing, analysis, and helper utilities.
"""

from .text_processing import (
    normalize_text,
    extract_sections,
    extract_answer_content,
    check_completion,
    fuzzy_match,
    calculate_text_similarity,
)

from .information_theory import (
    calculate_entropy,
    calculate_information_density,
    calculate_compression_ratio,
    normalized_compression_distance,
)

from .structural_analysis import (
    count_reasoning_steps,
    analyze_reasoning_structure,
    calculate_cyclomatic_complexity,
    detect_linear_structure,
)

__all__ = [
    # Text processing
    "normalize_text",
    "extract_sections",
    "extract_answer_content",
    "check_completion",
    "fuzzy_match",
    "calculate_text_similarity",
    # Information theory
    "calculate_entropy",
    "calculate_information_density",
    "calculate_compression_ratio",
    "normalized_compression_distance",
    # Structural analysis
    "count_reasoning_steps",
    "analyze_reasoning_structure",
    "calculate_cyclomatic_complexity",
    "detect_linear_structure",
]
