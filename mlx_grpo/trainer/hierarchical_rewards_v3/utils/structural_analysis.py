"""
Structural Analysis Utilities
=============================

Analyzes text structure using concepts from static code analysis.

Key Concepts Borrowed:
    - Cyclomatic complexity → Reasoning branches
    - AST depth → Reasoning depth
    - Clone detection → Repetition patterns
    - Dependency analysis → Coherence flow
    - Halstead metrics → Cognitive complexity

These measures help reward genuine reasoning over template exploitation.
"""

import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# REASONING STEP ANALYSIS
# ============================================================================


def count_reasoning_steps(text: str) -> Tuple[int, List[str]]:
    """
    Count distinct reasoning steps in text.

    Identifies steps by:
        - Explicit markers (Step 1, First, etc.)
        - Logical connectors (Therefore, Because, etc.)
        - Paragraph structure

    Returns:
        (step_count, step_markers_found)
    """
    if not text:
        return 0, []

    markers_found = []

    # Explicit step markers
    step_patterns = [
        r"\b(?:step\s+)?(\d+)[.):]\s",
        r"\b(first|second|third|fourth|fifth|finally|lastly)\b",
        r"^[-*•]\s",  # Bullet points
    ]

    for pattern in step_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            markers_found.extend([str(m) for m in matches if m])

    # Logical connectors indicating new reasoning
    connectors = [
        r"\b(therefore|thus|hence|so|consequently)\b",
        r"\b(because|since|as|given that)\b",
        r"\b(if|when|assuming)\b.*\b(then)\b",
        r"\b(first|next|then|finally|in conclusion)\b",
        r"\b(however|but|although|despite)\b",
        r"\b(this means|this implies|it follows)\b",
    ]

    for pattern in connectors:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            markers_found.extend([str(m) if isinstance(m, str) else m[0] for m in matches])

    # Paragraph-based steps (each paragraph is a potential step)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Estimate step count
    explicit_steps = len(set(markers_found))
    paragraph_steps = max(1, len(paragraphs))

    # Use higher of explicit or paragraph count, but cap
    step_count = max(explicit_steps, paragraph_steps)
    step_count = min(step_count, 20)  # Cap at reasonable maximum

    return step_count, list(set(markers_found))[:10]


def analyze_reasoning_structure(text: str) -> Dict[str, any]:
    """
    Analyze the structural quality of reasoning.

    Returns detailed analysis useful for reward computation.
    """
    if not text:
        return {
            "step_count": 0,
            "depth": 0,
            "branches": 0,
            "linear_ratio": 1.0,
            "has_conclusion": False,
            "has_premises": False,
            "markers": [],
        }

    # Count steps
    step_count, markers = count_reasoning_steps(text)

    # Analyze depth (nested reasoning)
    depth = calculate_reasoning_depth(text)

    # Count branches (conditional reasoning)
    branches = count_reasoning_branches(text)

    # Check for conclusion
    conclusion_patterns = [
        r"\b(therefore|thus|hence|in conclusion|finally)\b.*$",
        r"\b(the answer is|the result is|we get)\b",
        r"\\boxed\{",
    ]
    has_conclusion = any(
        re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in conclusion_patterns
    )

    # Check for premises
    premise_patterns = [
        r"\b(given|assuming|let|suppose|we know)\b",
        r"\b(the problem states|according to)\b",
    ]
    has_premises = any(re.search(p, text, re.IGNORECASE) for p in premise_patterns)

    # Linear ratio (how much is just sequential vs branching)
    total_statements = step_count + branches
    linear_ratio = step_count / total_statements if total_statements > 0 else 1.0

    return {
        "step_count": step_count,
        "depth": depth,
        "branches": branches,
        "linear_ratio": linear_ratio,
        "has_conclusion": has_conclusion,
        "has_premises": has_premises,
        "markers": markers,
    }


# ============================================================================
# CYCLOMATIC COMPLEXITY ANALOG
# ============================================================================


def calculate_cyclomatic_complexity(text: str) -> int:
    """
    Calculate cyclomatic complexity analog for text reasoning.

    In code: CC = E - N + 2P (edges - nodes + 2*components)
    For text: We count decision points (if/then, cases, alternatives)

    Higher complexity = more branching reasoning (good for hard problems).
    Very low complexity = possibly too simplistic.
    Very high complexity = possibly confused reasoning.

    Returns:
        Complexity score (1 = linear, higher = more branches)
    """
    if not text:
        return 1

    complexity = 1  # Base complexity

    # Decision patterns (each adds to complexity)
    decision_patterns = [
        (r"\b(if|when|assuming|suppose)\b", 1),
        (r"\b(else|otherwise|alternatively)\b", 1),
        (r"\b(case\s+\d|scenario\s+\d)\b", 1),
        (r"\b(either|or)\b", 1),
        (r"\b(however|but|although)\b", 0.5),  # Partial branch
        (r"\b(first|second|third)\b.*\b(first|second|third)\b", 0.5),  # Parallel paths
    ]

    for pattern, weight in decision_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        complexity += len(matches) * weight

    return max(1, int(complexity))


def calculate_reasoning_depth(text: str) -> int:
    """
    Calculate reasoning depth (like AST depth in code).

    Depth increases with:
        - Nested conditions
        - Sub-problems
        - Layered deductions

    Returns:
        Depth level (1 = flat, higher = more nested)
    """
    if not text:
        return 0

    depth = 1
    max_depth = 1

    # Markers that increase depth
    depth_increase = [
        r"\b(for this|to do this|to find this)\b",
        r"\b(first we need to|let\'s start by)\b",
        r"\b(breaking this down|let me analyze)\b",
        r"\b(within this|inside this)\b",
    ]

    # Markers that decrease depth (returning from sub-problem)
    depth_decrease = [
        r"\b(returning to|going back to)\b",
        r"\b(so overall|putting it together)\b",
        r"\b(in summary|to summarize)\b",
    ]

    sentences = re.split(r"[.!?]+", text)

    for sentence in sentences:
        for pattern in depth_increase:
            if re.search(pattern, sentence, re.IGNORECASE):
                depth += 1
                max_depth = max(max_depth, depth)
                break

        for pattern in depth_decrease:
            if re.search(pattern, sentence, re.IGNORECASE):
                depth = max(1, depth - 1)
                break

    return max_depth


def count_reasoning_branches(text: str) -> int:
    """
    Count reasoning branches (conditional paths).

    Returns:
        Number of distinct reasoning branches
    """
    if not text:
        return 0

    branches = 0

    # Explicit branching patterns
    branch_patterns = [
        r"\b(if.*then|when.*then)\b",
        r"\b(case\s+\d+|scenario\s+\d+)\b",
        r"\b(alternatively|on the other hand)\b",
        r"\b(option\s+[a-z\d])\b",
        r"\b(either.*or)\b",
    ]

    for pattern in branch_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        branches += len(matches)

    return branches


def detect_linear_structure(text: str) -> Tuple[bool, float]:
    """
    Detect if reasoning is purely linear (no branching).

    Pure linear reasoning might indicate:
        - Simple problem (OK)
        - Template exploitation (bad)
        - Missing edge case handling (bad)

    Returns:
        (is_linear, linearity_ratio)
    """
    structure = analyze_reasoning_structure(text)

    linearity = structure["linear_ratio"]
    is_linear = linearity > 0.85 and structure["branches"] < 2

    return is_linear, linearity


# ============================================================================
# CLONE DETECTION (REPETITION ANALYSIS)
# ============================================================================


def detect_clones(text: str, min_length: int = 20) -> List[Tuple[str, int]]:
    """
    Detect repeated text segments (like code clone detection).

    Uses sentence/phrase-level detection to avoid false positives
    from overlapping windows.

    Args:
        text: Text to analyze
        min_length: Minimum clone length in characters

    Returns:
        List of (clone_text, occurrence_count)
    """
    if not text or len(text) < min_length * 2:
        return []

    # Split into sentences and phrases
    import re

    # Split on sentence boundaries and newlines
    segments = re.split(r"[.!?\n]+", text)

    # Normalize and filter
    normalized_segments = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) >= min_length:
            normalized = " ".join(seg.lower().split())
            normalized_segments.append(normalized)

    # Count occurrences
    segment_counts = defaultdict(int)
    for seg in normalized_segments:
        segment_counts[seg] += 1

    # Find actual clones (appear more than once)
    clones = []
    for segment, count in segment_counts.items():
        if count > 1:
            clones.append((segment, count))

    # Sort by impact (length * (count - 1))
    clones.sort(key=lambda x: len(x[0]) * (x[1] - 1), reverse=True)

    return clones[:10]


def calculate_clone_ratio(text: str) -> float:
    """
    Calculate ratio of text that is cloned/repeated.

    Returns:
        Ratio in [0, 1] where 0 = no clones, 1 = all cloned
    """
    if not text:
        return 0.0

    clones = detect_clones(text)

    if not clones:
        return 0.0

    # Calculate characters involved in clones
    cloned_chars = sum(len(clone) * (count - 1) for clone, count in clones)

    return min(1.0, cloned_chars / len(text))


# ============================================================================
# HALSTEAD METRICS ANALOG
# ============================================================================


def calculate_halstead_metrics(text: str) -> Dict[str, float]:
    """
    Calculate Halstead-like metrics for text.

    In code analysis:
        - n1 = distinct operators
        - n2 = distinct operands
        - N1 = total operators
        - N2 = total operands

    For text:
        - "operators" = function words (the, is, of, etc.)
        - "operands" = content words (nouns, verbs, adjectives)

    Metrics:
        - Vocabulary: n = n1 + n2
        - Length: N = N1 + N2
        - Volume: V = N * log2(n)
        - Difficulty: D = (n1/2) * (N2/n2)
        - Effort: E = D * V
    """
    if not text:
        return {
            "vocabulary": 0,
            "length": 0,
            "volume": 0.0,
            "difficulty": 0.0,
            "effort": 0.0,
        }

    words = text.lower().split()
    if not words:
        return {
            "vocabulary": 0,
            "length": 0,
            "volume": 0.0,
            "difficulty": 0.0,
            "effort": 0.0,
        }

    # Function words (operators)
    function_words = {
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
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
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
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "although",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "me",
        "him",
        "her",
        "them",
        "us",
        "my",
        "your",
        "his",
        "their",
        "our",
        "which",
        "who",
        "whom",
        "what",
        "whose",
    }

    operators = [w for w in words if w in function_words]
    operands = [w for w in words if w not in function_words]

    n1 = len(set(operators))  # Distinct operators
    n2 = len(set(operands))  # Distinct operands
    N1 = len(operators)  # Total operators
    N2 = len(operands)  # Total operands

    n = n1 + n2  # Vocabulary
    N = N1 + N2  # Length

    if n <= 1:
        return {
            "vocabulary": n,
            "length": N,
            "volume": 0.0,
            "difficulty": 0.0,
            "effort": 0.0,
        }

    V = N * math.log2(n)  # Volume

    D = (n1 / 2) * (N2 / n2) if n2 > 0 else 0  # Difficulty
    E = D * V  # Effort

    return {
        "vocabulary": n,
        "length": N,
        "volume": V,
        "difficulty": D,
        "effort": E,
        "n1": n1,
        "n2": n2,
        "N1": N1,
        "N2": N2,
    }


def calculate_cognitive_complexity(text: str) -> float:
    """
    Calculate cognitive complexity score.

    Combines multiple measures into overall complexity.

    Returns:
        Complexity score (0 = trivial, higher = more complex)
    """
    structure = analyze_reasoning_structure(text)
    halstead = calculate_halstead_metrics(text)
    cyclomatic = calculate_cyclomatic_complexity(text)

    # Combine measures
    complexity = (
        0.3 * min(1.0, structure["step_count"] / 10)
        + 0.2 * min(1.0, structure["depth"] / 5)
        + 0.2 * min(1.0, cyclomatic / 10)
        + 0.15 * min(1.0, halstead["difficulty"] / 100)
        + 0.15 * (1 - structure["linear_ratio"])  # Reward non-linearity
    )

    return complexity


# ============================================================================
# DEPENDENCY ANALYSIS (COHERENCE)
# ============================================================================


def analyze_reference_chain(text: str) -> Dict[str, any]:
    """
    Analyze reference chains (like dependency analysis in code).

    Checks if reasoning builds on previous statements.
    Good reasoning has clear reference chains.
    """
    if not text:
        return {
            "forward_refs": 0,
            "back_refs": 0,
            "dangling_refs": 0,
            "coherence_score": 0.0,
        }

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return {
            "forward_refs": 0,
            "back_refs": 0,
            "dangling_refs": 0,
            "coherence_score": 1.0,
        }

    # Reference patterns
    back_ref_patterns = [
        r"\b(this|that|these|those|it)\b",
        r"\b(above|previous|earlier|before)\b",
        r"\b(as mentioned|as stated|as shown)\b",
    ]

    forward_ref_patterns = [
        r"\b(below|following|next|later)\b",
        r"\b(we will|let\'s|let us)\b",
    ]

    back_refs = 0
    forward_refs = 0

    for sentence in sentences:
        for pattern in back_ref_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                back_refs += 1
                break

        for pattern in forward_ref_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                forward_refs += 1
                break

    # Coherence: good if back refs in later sentences
    # (shows building on previous reasoning)
    coherence = back_refs / len(sentences) if sentences else 0

    return {
        "forward_refs": forward_refs,
        "back_refs": back_refs,
        "dangling_refs": 0,  # Would need more sophisticated analysis
        "coherence_score": min(1.0, coherence * 2),  # Scale up
    }


def calculate_structural_quality(text: str) -> Tuple[float, Dict[str, any]]:
    """
    Calculate overall structural quality score.

    Returns:
        (score in [0,1], detailed_metrics)
    """
    structure = analyze_reasoning_structure(text)
    halstead = calculate_halstead_metrics(text)
    references = analyze_reference_chain(text)
    clone_ratio = calculate_clone_ratio(text)

    metrics = {
        **structure,
        "halstead": halstead,
        "references": references,
        "clone_ratio": clone_ratio,
    }

    # Calculate score
    # Reward: steps, depth, non-linearity, coherence
    # Penalize: high clone ratio

    score = (
        0.20 * min(1.0, structure["step_count"] / 5)
        + 0.15 * min(1.0, structure["depth"] / 3)
        + 0.15 * (1 - structure["linear_ratio"])
        + 0.15 * (1.0 if structure["has_conclusion"] else 0.0)
        + 0.10 * (1.0 if structure["has_premises"] else 0.0)
        + 0.15 * references["coherence_score"]
        + 0.10 * (1 - clone_ratio)
    )

    return score, metrics
