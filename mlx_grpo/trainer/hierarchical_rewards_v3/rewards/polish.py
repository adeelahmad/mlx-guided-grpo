"""
Polish Rewards - Style and Presentation Quality

The final tier focusing on:
- Format adherence and clean structure
- Readability and clarity
- Appropriate length utilization
- Professional presentation
- Consistency in style
"""

import re
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ..core.base import ComponentResult, RewardResult
from ..core.config import RewardConfig
from ..utils.text_processing import (
    normalize_text,
    extract_sections,
    count_tokens_approx,
)
from ..utils.information_theory import calculate_entropy as char_entropy


@dataclass
class PolishMetrics:
    """Detailed polish/style metrics."""

    format_score: float = 0.0
    readability_score: float = 0.0
    length_efficiency: float = 0.0
    consistency_score: float = 0.0
    presentation_score: float = 0.0

    # Detailed metrics
    sentence_length_variance: float = 0.0
    paragraph_balance: float = 0.0
    whitespace_ratio: float = 0.0
    punctuation_density: float = 0.0
    capitalization_errors: int = 0


def compute_format_score(
    response: str, config: RewardConfig
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate format adherence and structural cleanliness.

    Checks:
    - Proper tag usage and closure
    - Consistent indentation patterns
    - Clean section breaks
    - No malformed structures
    """
    details = {}
    scores = []

    # 1. Tag matching and closure
    tag_pattern = r"<(\w+)>"
    close_pattern = r"</(\w+)>"

    open_tags = re.findall(tag_pattern, response)
    close_tags = re.findall(close_pattern, response)

    # Check for balanced tags
    open_counts = {}
    for tag in open_tags:
        open_counts[tag] = open_counts.get(tag, 0) + 1

    close_counts = {}
    for tag in close_tags:
        close_counts[tag] = close_counts.get(tag, 0) + 1

    # Calculate tag balance score
    all_tags = set(open_counts.keys()) | set(close_counts.keys())
    if all_tags:
        balanced = sum(
            1 for t in all_tags if open_counts.get(t, 0) == close_counts.get(t, 0)
        )
        tag_balance = balanced / len(all_tags)
    else:
        tag_balance = 1.0  # No tags is fine

    details["tag_balance"] = tag_balance
    scores.append(tag_balance)

    # 2. Proper nesting (simplified check)
    # Check if close tags appear after their open tags
    nesting_score = 1.0
    for tag in all_tags:
        open_positions = [m.start() for m in re.finditer(f"<{tag}>", response)]
        close_positions = [m.start() for m in re.finditer(f"</{tag}>", response)]

        if open_positions and close_positions:
            # Each close should come after an open
            if min(close_positions) < min(open_positions):
                nesting_score -= 0.2

    nesting_score = max(0.0, nesting_score)
    details["nesting_score"] = nesting_score
    scores.append(nesting_score)

    # 3. Consistent line structure
    lines = response.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    if non_empty_lines:
        # Check for consistent indentation patterns
        indent_levels = []
        for line in non_empty_lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            indent_levels.append(indent)

        # Expect reasonable indentation (0, 2, 4, etc.)
        valid_indents = sum(1 for i in indent_levels if i % 2 == 0 or i == 0)
        indent_consistency = valid_indents / len(indent_levels)
    else:
        indent_consistency = 0.5

    details["indent_consistency"] = indent_consistency
    scores.append(indent_consistency)

    # 4. No trailing/excessive whitespace
    trailing_ws_lines = sum(1 for l in lines if l != l.rstrip())
    if lines:
        ws_cleanliness = 1.0 - (trailing_ws_lines / len(lines))
    else:
        ws_cleanliness = 0.5

    details["whitespace_cleanliness"] = ws_cleanliness
    scores.append(ws_cleanliness)

    final_score = sum(scores) / len(scores) if scores else 0.5
    return final_score, details


def compute_readability_score(
    response: str, config: RewardConfig
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate readability and clarity.

    Based on:
    - Sentence length distribution
    - Word complexity (syllable count proxy)
    - Paragraph structure
    - Use of transitions/connectors
    """
    details = {}

    # Extract readable content (outside tags for cleaner analysis)
    # Simple approach: remove XML-like tags
    clean_text = re.sub(r"<[^>]+>", " ", response)
    clean_text = " ".join(clean_text.split())

    if len(clean_text) < 20:
        return 0.5, {"note": "insufficient_text"}

    # 1. Sentence length analysis
    sentences = re.split(r"[.!?]+", clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if sentences:
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)

        # Optimal sentence length: 10-20 words
        # Score based on how close to optimal range
        if 10 <= avg_length <= 20:
            length_score = 1.0
        elif avg_length < 10:
            length_score = avg_length / 10
        else:
            length_score = max(0.3, 1.0 - (avg_length - 20) / 30)

        # Variance: too uniform is robotic, too varied is chaotic
        if len(sentence_lengths) > 1:
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(
                sentence_lengths
            )
            std_dev = math.sqrt(variance)
            # Optimal std_dev around 5-10
            if 5 <= std_dev <= 10:
                variance_score = 1.0
            elif std_dev < 5:
                variance_score = std_dev / 5
            else:
                variance_score = max(0.3, 1.0 - (std_dev - 10) / 15)
        else:
            variance_score = 0.7

        details["avg_sentence_length"] = avg_length
        details["sentence_length_std"] = std_dev if len(sentence_lengths) > 1 else 0
        details["length_score"] = length_score
        details["variance_score"] = variance_score
    else:
        length_score = 0.5
        variance_score = 0.5

    # 2. Word complexity (simple proxy: average word length)
    words = clean_text.split()
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        # Optimal: 4-7 characters
        if 4 <= avg_word_length <= 7:
            complexity_score = 1.0
        elif avg_word_length < 4:
            complexity_score = avg_word_length / 4
        else:
            complexity_score = max(0.4, 1.0 - (avg_word_length - 7) / 5)

        details["avg_word_length"] = avg_word_length
        details["complexity_score"] = complexity_score
    else:
        complexity_score = 0.5

    # 3. Transition words (indicates flow)
    transitions = [
        "therefore",
        "however",
        "furthermore",
        "moreover",
        "thus",
        "consequently",
        "additionally",
        "meanwhile",
        "nevertheless",
        "first",
        "second",
        "third",
        "finally",
        "next",
        "then",
        "because",
        "since",
        "although",
        "while",
        "whereas",
        "in conclusion",
        "for example",
        "in other words",
        "as a result",
    ]

    text_lower = clean_text.lower()
    transition_count = sum(1 for t in transitions if t in text_lower)

    # Expect roughly 1 transition per 50 words
    expected_transitions = len(words) / 50 if words else 0
    if expected_transitions > 0:
        transition_ratio = min(1.0, transition_count / expected_transitions)
        # Don't penalize too harshly for lack of transitions
        transition_score = 0.5 + 0.5 * transition_ratio
    else:
        transition_score = 0.7

    details["transition_count"] = transition_count
    details["transition_score"] = transition_score

    # Combine scores
    final_score = (
        0.35 * length_score
        + 0.25 * variance_score
        + 0.20 * complexity_score
        + 0.20 * transition_score
    )

    return final_score, details


def compute_length_efficiency(
    response: str, config: RewardConfig, max_tokens: int = 450
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate efficient use of available token budget.

    Rewards:
    - Substantive use of space (not padding)
    - Appropriate length for task complexity
    - Not truncating important content
    """
    details = {}

    token_count = count_tokens_approx(response)
    token_ratio = token_count / max_tokens

    details["token_count"] = token_count
    details["token_ratio"] = token_ratio

    # Extract sections to analyze content distribution
    think_section, answer_section = extract_sections(response)

    think_tokens = count_tokens_approx(think_section)
    answer_tokens = count_tokens_approx(answer_section)

    details["think_tokens"] = think_tokens
    details["answer_tokens"] = answer_tokens

    # 1. Overall utilization
    # Too short might mean insufficient reasoning
    # Too long (near limit) might mean truncation risk
    if 0.5 <= token_ratio <= 0.9:
        utilization_score = 1.0
    elif token_ratio < 0.5:
        # Penalize very short responses
        utilization_score = token_ratio / 0.5
    else:
        # Near limit - slight concern about truncation
        utilization_score = max(0.7, 1.0 - (token_ratio - 0.9) * 2)

    details["utilization_score"] = utilization_score

    # 2. Balance between thinking and answer
    if think_tokens > 0 and answer_tokens > 0:
        # Expect more tokens in thinking than answer (3:1 to 5:1 ratio)
        ratio = think_tokens / answer_tokens if answer_tokens > 0 else 10

        if 2 <= ratio <= 6:
            balance_score = 1.0
        elif ratio < 2:
            balance_score = ratio / 2
        else:
            balance_score = max(0.6, 1.0 - (ratio - 6) / 10)
    elif answer_tokens > 0:
        # Has answer but no think section - partial credit
        balance_score = 0.5
    else:
        balance_score = 0.3

    details["think_answer_ratio"] = (
        think_tokens / answer_tokens if answer_tokens > 0 else 0
    )
    details["balance_score"] = balance_score

    # 3. Information density check (via entropy)
    if len(response) > 50:
        entropy = char_entropy(response)
        # High entropy = diverse content, low = repetitive
        # Typical good text: 4.0-5.0 bits
        if 3.5 <= entropy <= 5.5:
            density_score = 1.0
        elif entropy < 3.5:
            density_score = entropy / 3.5
        else:
            density_score = 0.9  # High entropy is usually fine
    else:
        density_score = 0.5

    details["char_entropy"] = entropy if len(response) > 50 else 0
    details["density_score"] = density_score

    # Combine
    final_score = 0.40 * utilization_score + 0.35 * balance_score + 0.25 * density_score

    return final_score, details


def compute_consistency_score(
    response: str, config: RewardConfig
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate consistency in style and tone throughout response.

    Checks:
    - Consistent tense usage
    - Consistent formality level
    - Consistent notation/formatting choices
    """
    details = {}

    # Extract clean text
    clean_text = re.sub(r"<[^>]+>", " ", response)
    clean_text = " ".join(clean_text.split())

    if len(clean_text) < 50:
        return 0.7, {"note": "insufficient_text"}

    sentences = re.split(r"[.!?]+", clean_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]

    if len(sentences) < 2:
        return 0.7, {"note": "too_few_sentences"}

    # 1. Tense consistency (simple heuristic)
    past_indicators = ["was", "were", "had", "did", "went", "said", "made", "got"]
    present_indicators = ["is", "are", "has", "does", "goes", "says", "makes", "gets"]

    text_lower = clean_text.lower()
    words = text_lower.split()

    past_count = sum(1 for w in words if w in past_indicators)
    present_count = sum(1 for w in words if w in present_indicators)

    total_tense = past_count + present_count
    if total_tense > 0:
        # High ratio in either direction = consistent
        tense_ratio = max(past_count, present_count) / total_tense
        tense_score = tense_ratio
    else:
        tense_score = 0.7  # Neutral if no clear indicators

    details["past_count"] = past_count
    details["present_count"] = present_count
    details["tense_score"] = tense_score

    # 2. Formality consistency
    # Contractions indicate informal style
    contractions = ["n't", "'re", "'ve", "'ll", "'m", "'d", "won't", "can't", "don't"]
    contraction_count = sum(1 for c in contractions if c in text_lower)

    # First-person indicators
    first_person = (
        text_lower.count(" i ") + text_lower.count("i ") + text_lower.count(" my ")
    )

    # Formal indicators
    formal_words = [
        "therefore",
        "consequently",
        "furthermore",
        "moreover",
        "hence",
        "thus",
    ]
    formal_count = sum(1 for f in formal_words if f in text_lower)

    # Calculate formality balance
    informal_signals = contraction_count + first_person
    formal_signals = formal_count

    total_signals = informal_signals + formal_signals + 1
    if total_signals > 1:
        # We want consistency - either mostly formal or mostly informal
        formality_ratio = max(informal_signals, formal_signals) / total_signals
        formality_score = 0.5 + 0.5 * formality_ratio
    else:
        formality_score = 0.7

    details["informal_signals"] = informal_signals
    details["formal_signals"] = formal_signals
    details["formality_score"] = formality_score

    # 3. Notation consistency (math, numbering, etc.)
    # Check for consistent use of mathematical notation
    uses_latex = bool(re.search(r"\$[^$]+\$", response))
    uses_plain_math = bool(re.search(r"[0-9]+\s*[+\-*/=]\s*[0-9]+", response))

    if uses_latex and uses_plain_math:
        # Mixed notation - less consistent
        notation_score = 0.6
    else:
        notation_score = 1.0

    details["uses_latex"] = uses_latex
    details["uses_plain_math"] = uses_plain_math
    details["notation_score"] = notation_score

    # Combine
    final_score = 0.35 * tense_score + 0.35 * formality_score + 0.30 * notation_score

    return final_score, details


def compute_presentation_score(
    response: str, config: RewardConfig
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate overall presentation quality.

    Checks:
    - Clean visual structure
    - Appropriate use of emphasis
    - Professional appearance
    """
    details = {}
    scores = []

    # 1. Visual structure
    lines = response.split("\n")

    # Check for reasonable line lengths
    very_long_lines = sum(1 for l in lines if len(l) > 200)
    if lines:
        long_line_ratio = very_long_lines / len(lines)
        visual_score = 1.0 - long_line_ratio
    else:
        visual_score = 0.5

    details["very_long_lines"] = very_long_lines
    details["visual_score"] = visual_score
    scores.append(visual_score)

    # 2. Appropriate spacing
    # Not too dense, not too sparse
    if response:
        newline_ratio = response.count("\n") / len(response)
        # Optimal: 1-5% newlines
        if 0.01 <= newline_ratio <= 0.05:
            spacing_score = 1.0
        elif newline_ratio < 0.01:
            spacing_score = 0.7  # Dense but acceptable
        else:
            spacing_score = max(0.5, 1.0 - (newline_ratio - 0.05) * 5)
    else:
        spacing_score = 0.5

    details["newline_ratio"] = newline_ratio if response else 0
    details["spacing_score"] = spacing_score
    scores.append(spacing_score)

    # 3. No obvious artifacts
    artifacts = [
        "...",  # Excessive ellipsis
        "???",  # Multiple question marks
        "!!!",  # Multiple exclamation
        "(((",  # Nested parens
        ")))",
        "[[[[",  # Unusual brackets
        "]]]]",
    ]

    artifact_count = sum(1 for a in artifacts if a in response)
    artifact_score = max(0.5, 1.0 - artifact_count * 0.15)

    details["artifact_count"] = artifact_count
    details["artifact_score"] = artifact_score
    scores.append(artifact_score)

    # 4. Answer completeness indicator
    # Check if answer section seems complete (doesn't end mid-sentence)
    _, answer = extract_sections(response)

    if answer:
        answer_stripped = answer.strip()
        # Check for proper ending
        if answer_stripped and answer_stripped[-1] in ".!?)0123456789":
            completion_score = 1.0
        elif answer_stripped:
            completion_score = 0.7
        else:
            completion_score = 0.5
    else:
        completion_score = 0.4

    details["completion_score"] = completion_score
    scores.append(completion_score)

    final_score = sum(scores) / len(scores) if scores else 0.5
    return final_score, details


def compute_polish_reward(
    response: str, expected: str, question: str, config: RewardConfig, **kwargs
) -> RewardResult:
    """
    Main polish reward computation.

    Combines:
    - Format adherence
    - Readability
    - Length efficiency
    - Consistency
    - Presentation
    """
    components = []

    # 1. Format score
    format_score, format_details = compute_format_score(response, config)
    components.append(
        ComponentResult(
            name="format",
            raw_score=format_score,
            weight=0.20,
            weighted_score=format_score * 0.20,
            diagnostics=format_details,
        )
    )

    # 2. Readability score
    readability_score, readability_details = compute_readability_score(response, config)
    components.append(
        ComponentResult(
            name="readability",
            raw_score=readability_score,
            weight=0.20,
            weighted_score=readability_score * 0.20,
            diagnostics=readability_details,
        )
    )

    # 3. Length efficiency
    max_tokens = kwargs.get("max_tokens", 450)
    efficiency_score, efficiency_details = compute_length_efficiency(
        response, config, max_tokens
    )
    components.append(
        ComponentResult(
            name="length_efficiency",
            raw_score=efficiency_score,
            weight=0.25,
            weighted_score=efficiency_score * 0.25,
            diagnostics=efficiency_details,
        )
    )

    # 4. Consistency
    consistency_score, consistency_details = compute_consistency_score(response, config)
    components.append(
        ComponentResult(
            name="consistency",
            raw_score=consistency_score,
            weight=0.15,
            weighted_score=consistency_score * 0.15,
            diagnostics=consistency_details,
        )
    )

    # 5. Presentation
    presentation_score, presentation_details = compute_presentation_score(
        response, config
    )
    components.append(
        ComponentResult(
            name="presentation",
            raw_score=presentation_score,
            weight=0.20,
            weighted_score=presentation_score * 0.20,
            diagnostics=presentation_details,
        )
    )

    # Compute weighted total
    total_weight = sum(c.weight for c in components)
    final_score = (
        sum(c.raw_score * c.weight for c in components) / total_weight
        if total_weight > 0
        else 0.5
    )

    # Collect detailed metrics
    metrics = PolishMetrics(
        format_score=format_score,
        readability_score=readability_score,
        length_efficiency=efficiency_score,
        consistency_score=consistency_score,
        presentation_score=presentation_score,
    )

    # Define simple result class for inter-level communication
    from dataclasses import dataclass, field as dataclass_field
    from typing import List as ListType, Dict as DictType, Any as AnyType

    @dataclass
    class SimpleRewardResult:
        """Simple reward result for inter-level communication."""

        score: float
        components: ListType[ComponentResult] = dataclass_field(default_factory=list)
        details: DictType[str, AnyType] = dataclass_field(default_factory=dict)

    return SimpleRewardResult(
        score=final_score,
        components=components,
        details={
            "metrics": metrics.__dict__,
            "format": format_details,
            "readability": readability_details,
            "efficiency": efficiency_details,
            "consistency": consistency_details,
            "presentation": presentation_details,
        },
    )
