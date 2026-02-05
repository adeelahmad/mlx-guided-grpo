"""
Information Theory Utilities
============================

Information-theoretic measures for text analysis.

These measures are fundamental for anti-gaming because:
    - Gaming attempts typically reduce actual information content
    - Repetition decreases entropy
    - True reasoning has higher information density
    - Compression-based similarity resists keyword stuffing

Key Measures:
    - Shannon entropy
    - Information density (bits per token)
    - Compression ratio
    - Normalized Compression Distance (NCD)
    - Mutual information
"""

import logging
import math
import zlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENTROPY MEASURES
# ============================================================================


def calculate_entropy(text: str, level: str = "char") -> float:
    """
    Calculate Shannon entropy of text.

    Higher entropy = more information content.
    Repetitive text has lower entropy.

    Args:
        text: Input text
        level: "char" for character-level, "word" for word-level

    Returns:
        Entropy in bits
    """
    if not text:
        return 0.0

    if level == "char":
        elements = list(text)
    elif level == "word":
        elements = text.lower().split()
    else:
        raise ValueError(f"Unknown level: {level}")

    if not elements:
        return 0.0

    # Count frequencies
    counts = Counter(elements)
    total = len(elements)

    # Calculate entropy
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def calculate_max_entropy(text: str, level: str = "char") -> float:
    """
    Calculate maximum possible entropy for text.

    Max entropy occurs when all elements are unique.

    Returns:
        Maximum entropy in bits
    """
    if not text:
        return 0.0

    if level == "char":
        n = len(text)
    else:
        n = len(text.split())

    if n <= 1:
        return 0.0

    # Max entropy is log2(n) when all elements are unique
    return math.log2(n)


def calculate_normalized_entropy(text: str, level: str = "char") -> float:
    """
    Calculate normalized entropy (0 to 1).

    1.0 = maximum diversity
    0.0 = complete uniformity

    Returns:
        Normalized entropy in [0, 1]
    """
    entropy = calculate_entropy(text, level)
    max_entropy = calculate_max_entropy(text, level)

    if max_entropy == 0:
        return 1.0

    return entropy / max_entropy


def calculate_information_density(text: str) -> float:
    """
    Calculate information density (bits per token).

    This measures how much information each token carries.
    Repetitive text has low information density.

    High information density indicates:
        - Diverse vocabulary
        - Non-repetitive content
        - Actual reasoning (not filler)

    Returns:
        Information density (bits per word)
    """
    if not text:
        return 0.0

    words = text.lower().split()
    if not words:
        return 0.0

    # Word-level entropy
    word_entropy = calculate_entropy(text, "word")

    # Normalize by log2 of vocabulary size
    vocab_size = len(set(words))
    if vocab_size <= 1:
        return 0.0

    max_entropy = math.log2(vocab_size)

    # Information density: entropy per word, normalized
    return word_entropy / max_entropy if max_entropy > 0 else 0.0


def calculate_cross_entropy(text: str, reference: str) -> float:
    """
    Calculate cross-entropy between text and reference distribution.

    Lower cross-entropy = text is more predictable from reference.
    Used to measure how well text matches expected patterns.

    Args:
        text: Text to evaluate
        reference: Reference text defining expected distribution

    Returns:
        Cross-entropy in bits
    """
    if not text or not reference:
        return float("inf")

    # Get word distributions
    text_words = text.lower().split()
    ref_words = reference.lower().split()

    if not text_words or not ref_words:
        return float("inf")

    # Reference distribution with smoothing
    ref_counts = Counter(ref_words)
    ref_total = len(ref_words)
    vocab = set(text_words) | set(ref_words)
    vocab_size = len(vocab)

    # Laplace smoothing
    alpha = 0.1

    # Cross-entropy
    cross_entropy = 0.0
    for word in text_words:
        ref_count = ref_counts.get(word, 0)
        p_ref = (ref_count + alpha) / (ref_total + alpha * vocab_size)
        cross_entropy -= math.log2(p_ref)

    return cross_entropy / len(text_words)


# ============================================================================
# COMPRESSION-BASED MEASURES
# ============================================================================


def calculate_compression_ratio(text: str) -> float:
    """
    Calculate compression ratio using zlib.

    Lower ratio = more compressible = more repetitive.
    Higher ratio = less compressible = more information.

    Returns:
        Compression ratio (compressed_size / original_size)
    """
    if not text:
        return 1.0

    original = text.encode("utf-8")
    compressed = zlib.compress(original, level=9)

    return len(compressed) / len(original)


def calculate_kolmogorov_estimate(text: str) -> float:
    """
    Estimate Kolmogorov complexity using compression.

    Kolmogorov complexity is the length of the shortest program
    that produces the text. We approximate using compression.

    Higher = more complex = more genuine content.
    Lower = simpler = possibly gaming (repetition, templates).

    Returns:
        Normalized complexity estimate in [0, 1]
    """
    if not text or len(text) < 10:
        return 0.5

    original = text.encode("utf-8")
    compressed = zlib.compress(original, level=9)

    # Normalized by theoretical maximum
    # A random string compresses to about 1.0 ratio
    # Highly repetitive text compresses to near 0
    ratio = len(compressed) / len(original)

    # Map to [0, 1] where 1 = incompressible (high complexity)
    # Typical text compresses to 0.3-0.6
    return min(1.0, ratio / 0.8)


def normalized_compression_distance(text1: str, text2: str) -> float:
    """
    Calculate Normalized Compression Distance (NCD).

    NCD is an approximation of normalized information distance,
    which measures similarity based on compressibility.

    Key property: Resists keyword stuffing because:
        - Stuffing increases compressed size of text1 alone
        - But doesn't reduce compressed size of concatenation much

    Formula: NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

    Returns:
        NCD in [0, 1] where 0 = identical, 1 = maximally different
    """
    if not text1 or not text2:
        return 1.0

    # Encode texts
    bytes1 = text1.encode("utf-8")
    bytes2 = text2.encode("utf-8")
    bytes_concat = bytes1 + bytes2

    # Compress individually and concatenated
    c1 = len(zlib.compress(bytes1, level=9))
    c2 = len(zlib.compress(bytes2, level=9))
    c_concat = len(zlib.compress(bytes_concat, level=9))

    # NCD formula
    min_c = min(c1, c2)
    max_c = max(c1, c2)

    if max_c == 0:
        return 0.0

    ncd = (c_concat - min_c) / max_c

    # Clamp to [0, 1]
    return max(0.0, min(1.0, ncd))


def compression_based_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity using NCD.

    Returns:
        Similarity in [0, 1] where 1 = identical
    """
    return 1.0 - normalized_compression_distance(text1, text2)


# ============================================================================
# INFORMATION GAIN MEASURES
# ============================================================================


def calculate_information_gain_per_token(text: str) -> float:
    """
    Estimate information gain per new token.

    This measures how much new information each token adds.
    Repetitive content has low information gain.

    Returns:
        Average information gain in bits
    """
    if not text:
        return 0.0

    words = text.lower().split()
    if len(words) < 2:
        return 0.0

    # Calculate incremental entropy
    seen = Counter()
    total_gain = 0.0

    for i, word in enumerate(words):
        if i == 0:
            seen[word] = 1
            continue

        # Calculate surprise (self-information) of this word
        total_seen = sum(seen.values())
        word_count = seen.get(word, 0)

        if word_count == 0:
            # New word - high information
            p_new = 1 / (total_seen + 1)
            info = -math.log2(p_new)
        else:
            # Seen word - lower information
            p_seen = word_count / total_seen
            info = -math.log2(p_seen)

        total_gain += info
        seen[word] += 1

    return total_gain / (len(words) - 1)


def calculate_redundancy(text: str) -> float:
    """
    Calculate redundancy ratio.

    Redundancy = 1 - (actual_entropy / max_entropy)

    High redundancy = repetitive, low information.
    Low redundancy = diverse, high information.

    Returns:
        Redundancy ratio in [0, 1]
    """
    if not text:
        return 0.0

    norm_entropy = calculate_normalized_entropy(text, "word")
    return 1.0 - norm_entropy


# ============================================================================
# MUTUAL INFORMATION
# ============================================================================


def calculate_pointwise_mutual_information(
    text1: str,
    text2: str,
) -> Dict[str, float]:
    """
    Calculate pointwise mutual information between word co-occurrences.

    PMI(x, y) = log2(P(x,y) / (P(x) * P(y)))

    High PMI = words co-occur more than by chance.
    Used to detect if answer keywords genuinely relate to reasoning.

    Returns:
        Dictionary of word pairs to PMI scores
    """
    if not text1 or not text2:
        return {}

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Combined corpus
    all_words = list(words1) + list(words2)
    total = len(all_words)

    if total < 2:
        return {}

    # Individual probabilities
    counts = Counter(all_words)

    pmi_scores = {}

    # Calculate PMI for co-occurring words
    shared = words1 & words2
    for word in shared:
        # P(word in both)
        p_joint = 1 / total  # Simplified: appeared in both

        # P(word in text1), P(word in text2)
        p1 = len([w for w in text1.lower().split() if w == word]) / len(text1.split())
        p2 = len([w for w in text2.lower().split() if w == word]) / len(text2.split())

        if p1 > 0 and p2 > 0:
            pmi = math.log2(p_joint / (p1 * p2 + 1e-10))
            pmi_scores[word] = pmi

    return pmi_scores


def calculate_average_pmi(text1: str, text2: str) -> float:
    """
    Calculate average PMI between two texts.

    Used to measure genuine semantic connection vs keyword stuffing.

    Returns:
        Average PMI (can be negative)
    """
    pmi_scores = calculate_pointwise_mutual_information(text1, text2)

    if not pmi_scores:
        return 0.0

    return sum(pmi_scores.values()) / len(pmi_scores)


# ============================================================================
# COMPOSITE MEASURES
# ============================================================================


def analyze_information_content(text: str) -> Dict[str, float]:
    """
    Comprehensive information content analysis.

    Returns multiple measures useful for reward computation.
    """
    if not text:
        return {
            "char_entropy": 0.0,
            "word_entropy": 0.0,
            "normalized_entropy": 0.0,
            "information_density": 0.0,
            "compression_ratio": 1.0,
            "kolmogorov_estimate": 0.0,
            "redundancy": 1.0,
            "info_gain_per_token": 0.0,
        }

    return {
        "char_entropy": calculate_entropy(text, "char"),
        "word_entropy": calculate_entropy(text, "word"),
        "normalized_entropy": calculate_normalized_entropy(text, "word"),
        "information_density": calculate_information_density(text),
        "compression_ratio": calculate_compression_ratio(text),
        "kolmogorov_estimate": calculate_kolmogorov_estimate(text),
        "redundancy": calculate_redundancy(text),
        "info_gain_per_token": calculate_information_gain_per_token(text),
    }


def calculate_content_quality_score(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall content quality from information-theoretic perspective.

    High quality content has:
        - High information density
        - Low redundancy
        - High compression ratio (incompressible)
        - Good information gain per token

    Returns:
        (score in [0,1], detailed_metrics)
    """
    metrics = analyze_information_content(text)

    # Weight different aspects
    weights = {
        "information_density": 0.30,
        "kolmogorov_estimate": 0.25,
        "normalized_entropy": 0.20,
        "compression_ratio": 0.15,
        "info_gain_per_token": 0.10,
    }

    # Normalize metrics to [0, 1]
    normalized = {
        "information_density": min(1.0, metrics["information_density"]),
        "kolmogorov_estimate": metrics["kolmogorov_estimate"],
        "normalized_entropy": metrics["normalized_entropy"],
        "compression_ratio": min(1.0, metrics["compression_ratio"]),
        "info_gain_per_token": min(1.0, metrics["info_gain_per_token"] / 5.0),
    }

    score = sum(normalized[key] * weight for key, weight in weights.items())

    return score, metrics
