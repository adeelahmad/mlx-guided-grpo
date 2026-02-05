"""
Utility Functions
=================

Text processing, analysis, and helper utilities.
"""

from .information_theory import (
    calculate_compression_ratio,
    calculate_entropy,
    calculate_information_density,
    normalized_compression_distance,
)
from .structural_analysis import (
    analyze_reasoning_structure,
    calculate_cyclomatic_complexity,
    count_reasoning_steps,
    detect_linear_structure,
)
from .text_processing import (
    calculate_text_similarity,
    check_completion,
    extract_answer_content,
    extract_sections,
    fuzzy_match,
    normalize_text,
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
