"""
Text Processing Utilities
=========================

Robust text processing for reward computation.
Handles edge cases, various formats, and encodings.

Key Features:
    - Section extraction (thinking/answer)
    - Answer content extraction (boxed, code, etc.)
    - Completion checking
    - Similarity computation
    - N-gram analysis
"""

import logging
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    normalize_whitespace: bool = True,
    remove_accents: bool = False,
    preserve_numbers: bool = True,
) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punctuation: Strip punctuation
        normalize_whitespace: Collapse whitespace
        remove_accents: Remove diacritical marks
        preserve_numbers: Keep numeric characters

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    result = text

    # Remove accents
    if remove_accents:
        result = unicodedata.normalize("NFD", result)
        result = "".join(c for c in result if unicodedata.category(c) != "Mn")

    # Lowercase
    if lowercase:
        result = result.lower()

    # Remove punctuation (keep numbers if requested)
    if remove_punctuation:
        if preserve_numbers:
            result = re.sub(r"[^\w\s\d]", "", result)
        else:
            result = re.sub(r"[^\w\s]", "", result)

    # Normalize whitespace
    if normalize_whitespace:
        result = " ".join(result.split())

    return result.strip()


def normalize_number(text: Union[str, None]) -> Optional[float]:
    """
    Extract numeric value from text.

    Handles:
        - Plain integers: "42"
        - Decimals: "3.14", "3,14"
        - Fractions: "1/2"
        - Percentages: "50%"
        - Scientific notation: "1e-5"
        - Thousands separators: "1,000" and "1,000,000"
        - Negative numbers: "-42"

    Returns:
        Float value or None
    """
    if not text:
        return None

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # Remove thousands separators (comma followed by 3 digits at word boundary or another comma)
    # This handles 1,000 and 1,000,000 correctly in one pass
    text = re.sub(r",(?=\d{3}(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s|$|[^\d]))", "", text)

    # Handle percentage
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()

    # Try direct conversion
    try:
        value = float(text)
        return value / 100 if is_percent else value
    except ValueError:
        pass

    # Try with comma as decimal separator (European format)
    # Only if there's exactly one comma and it's not a thousands separator
    if text.count(",") == 1 and not re.match(r"^\d{1,3},\d{3}$", text):
        try:
            value = float(text.replace(",", "."))
            return value / 100 if is_percent else value
        except ValueError:
            pass

    # Try fraction
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den != 0:
                    return num / den
            except ValueError:
                pass

    # Try to extract number from text
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            value = float(match.group())
            return value / 100 if is_percent else value
        except ValueError:
            pass

    return None


def extract_numbers(text: Union[str, List, None]) -> List[float]:
    """
    Extract all numbers from text.

    Args:
        text: Input text (string, list of strings, or None)

    Returns:
        List of numeric values found
    """
    # Handle non-string input
    if text is None:
        return []
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t is not None)
    elif not isinstance(text, str):
        text = str(text)

    if not text:
        return []

    # Pattern for various number formats
    # - Optional negative sign
    # - Integer part (digits)
    # - Optional decimal part (single . or , followed by digits)
    # - Optional scientific notation
    # Using ? instead of * to prevent matching things like IP addresses (1.2.3.4)
    pattern = r"-?\d+(?:[,\.]\d+)?(?:[eE][+-]?\d+)?"
    matches = re.findall(pattern, text)

    numbers = []
    seen = set()  # Avoid duplicates from overlapping matches

    for match in matches:
        val = normalize_number(match)
        if val is not None and val not in seen:
            numbers.append(val)
            seen.add(val)

    return numbers


# ============================================================================
# SECTION EXTRACTION
# ============================================================================


def extract_sections(
    completion: str,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> Tuple[str, str]:
    """
    Extract thinking and answer sections from completion.

    Handles:
        - Standard <think>...</think>
        - Multiple think blocks (concatenates)
        - No think tags (all is answer)
        - Nested tags (outer only)
        - Malformed tags (best effort)

    Args:
        completion: Model completion text
        think_open: Opening tag
        think_close: Closing tag

    Returns:
        (thinking_text, answer_text)
    """
    if not completion:
        return "", ""

    if not isinstance(completion, str):
        completion = str(completion)

    thinking_parts = []

    # Find all thinking blocks
    pattern = re.escape(think_open) + r"(.*?)" + re.escape(think_close)

    for match in re.finditer(pattern, completion, flags=re.DOTALL):
        thinking_parts.append(match.group(1).strip())

    thinking = "\n\n".join(thinking_parts)

    # Extract answer (after last close tag, or all if no tags)
    if think_close in completion:
        last_close_idx = completion.rfind(think_close)
        answer = completion[last_close_idx + len(think_close) :].strip()
    else:
        # No think tags - check if there's an open tag without close
        if think_open in completion:
            # Incomplete thinking - treat content after open tag as thinking
            open_idx = completion.find(think_open)
            thinking = completion[open_idx + len(think_open) :].strip()
            answer = ""
        else:
            # No tags at all - entire completion is answer
            answer = completion.strip()

    return thinking, answer


def extract_answer_content(text: str) -> str:
    """
    Extract the actual answer content from various formats.

    Priority order:
        1. LaTeX boxed: \\boxed{answer}
        2. Final answer markers: "The answer is..."
        3. Code blocks: ```answer```
        4. Last meaningful line

    Args:
        text: Text potentially containing formatted answer

    Returns:
        Extracted answer content
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # Priority 1: Boxed LaTeX
    boxed_patterns = [
        r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}",  # Handle nested braces
        r"\\fbox\{([^{}]+)\}",
        r"\$\\boxed\{([^{}]+)\}\$",
    ]

    for pattern in boxed_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last boxed answer (usually the final result)
            return matches[-1].group(1).strip()

    # Priority 2: Answer markers
    answer_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
        r"(?:therefore|thus|hence|so)[,:\s]+(?:the\s+answer\s+is\s+)?([^\n.]+)",
        r"(?:answer|result|solution)[:\s]+([^\n.]+)",
        r"=\s*([^\n,=]+)$",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # Clean up common artifacts
            answer = re.sub(r"^[:\s]+", "", answer)
            answer = re.sub(r"[.\s]+$", "", answer)
            if answer:
                return answer

    # Priority 3: Code blocks
    code_patterns = [
        r"```(?:\w+)?\s*(.*?)```",
        r"`([^`]+)`",
    ]

    for pattern in code_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content and len(content) < 200:  # Reasonable answer length
                return content

    # Priority 4: Last meaningful line
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Filter out formatting lines
    skip_patterns = [
        r"^[#*\->|`]",  # Markdown formatting
        r"^[Ss]tep\s+\d",  # Step markers
        r"^[Ff]irst|[Ss]econd|[Tt]hird",  # Ordinals
        r"^[Ll]et\s+me",  # Reasoning phrases
    ]

    content_lines = [
        l for l in lines if not any(re.match(p, l) for p in skip_patterns) and len(l) > 1
    ]

    if content_lines:
        return content_lines[-1][:500]

    if lines:
        return lines[-1][:500]

    return text[:500]


# ============================================================================
# COMPLETION CHECKING
# ============================================================================


def check_completion(text: str) -> Tuple[bool, str, float]:
    """
    Check if text appears complete (not truncated).

    Returns:
        (is_complete, reason, confidence)
    """
    if not text:
        return False, "empty", 0.0

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if len(text) < 3:
        return False, "too_short", 0.0

    # Check for good endings
    good_endings = {
        ".": 0.95,
        "!": 0.95,
        "?": 0.95,
        ")": 0.85,
        "]": 0.85,
        "}": 0.90,
        '"': 0.80,
        "'": 0.80,
        "`": 0.75,
        "```": 0.90,
        ">": 0.70,
    }

    for ending, confidence in good_endings.items():
        if text.endswith(ending):
            return True, f"good_ending:{ending}", confidence

    # Check for complete LaTeX
    if "\\boxed{" in text:
        open_count = text.count("{")
        close_count = text.count("}")
        if open_count == close_count:
            return True, "complete_boxed", 0.90
        else:
            return False, f"unbalanced_braces:{open_count}:{close_count}", 0.3

    # Check for incomplete patterns
    incomplete_patterns = [
        (r"\s+(?:and|or|but|so|then)\s*$", "trailing_conjunction", 0.2),
        (r"\s+(?:the|a|an|this|that)\s*$", "trailing_article", 0.2),
        (r"\s+(?:is|are|was|were|be)\s*$", "trailing_verb", 0.3),
        (r"\s+(?:to|of|in|for|with)\s*$", "trailing_preposition", 0.2),
        (r"[,;:]\s*$", "trailing_punctuation", 0.4),
        (r"[\(\[\{]\s*$", "unclosed_bracket", 0.2),
        (r"[=\+\-\*/]\s*$", "trailing_operator", 0.3),
    ]

    for pattern, reason, confidence in incomplete_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, reason, confidence

    # Check bracket balance
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    for open_c, close_c in pairs:
        if text.count(open_c) > text.count(close_c):
            return False, f"unbalanced:{open_c}{close_c}", 0.4

    # Check for ending with alphanumeric (might be incomplete)
    if text[-1].isalnum():
        # Short answers ending with alnum are usually OK
        if len(text) < 50:
            return True, "short_alnum_ok", 0.7
        else:
            return False, "no_terminal_punct", 0.5

    return True, "acceptable", 0.75


def check_tag_balance(text: str) -> Tuple[bool, int, str]:
    """
    Check if XML-like tags are balanced.

    Returns:
        (is_balanced, imbalance_count, details)
    """
    if not text:
        return True, 0, "empty"

    if not isinstance(text, str):
        text = str(text)

    # Find all tags - capture self-closing indicator
    tag_pattern = r"<(/?)(\w+)(?:\s[^>]*)?(/)?\s*>"

    stack = []
    for match in re.finditer(tag_pattern, text):
        is_close = match.group(1) == "/"
        tag_name = match.group(2).lower()
        is_self_closing = match.group(3) == "/"

        # Skip self-closing tags
        if is_self_closing:
            continue

        if is_close:
            if stack and stack[-1] == tag_name:
                stack.pop()
            else:
                stack.append(f"/{tag_name}")  # Unmatched close
        else:
            stack.append(tag_name)

    is_balanced = len(stack) == 0
    return is_balanced, len(stack), ",".join(stack) if stack else "balanced"


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================


def fuzzy_match(text1: str, text2: str, normalize: bool = True) -> float:
    """
    Compute fuzzy similarity between texts.

    Args:
        text1: First text
        text2: Second text
        normalize: Whether to normalize texts first

    Returns:
        Similarity in [0, 1]
    """
    if not text1 or not text2:
        return 0.0

    if normalize:
        t1 = normalize_text(text1)
        t2 = normalize_text(text2)
    else:
        t1, t2 = str(text1), str(text2)

    if not t1 or not t2:
        return 0.0

    # Exact match
    if t1 == t2:
        return 1.0

    # Substring check
    if t2 in t1:
        return 0.9 + 0.1 * (len(t2) / len(t1))
    if t1 in t2:
        return 0.9 + 0.1 * (len(t1) / len(t2))

    # Numeric comparison
    num1 = normalize_number(t1)
    num2 = normalize_number(t2)

    if num1 is not None and num2 is not None:
        if num1 == num2:
            return 1.0
        if num2 != 0:
            rel_error = abs(num1 - num2) / max(abs(num1), abs(num2))
            if rel_error < 0.001:
                return 0.99
            if rel_error < 0.01:
                return 0.95
            if rel_error < 0.05:
                return 0.85

    # Sequence matching
    return SequenceMatcher(None, t1, t2).ratio()


def calculate_text_similarity(
    text1: str,
    text2: str,
    method: str = "combined",
) -> float:
    """
    Calculate similarity using specified method.

    Methods:
        - "fuzzy": SequenceMatcher ratio
        - "jaccard": Word-level Jaccard
        - "cosine": Word frequency cosine
        - "combined": Weighted combination
        - "overlap": Word overlap coefficient

    Returns:
        Similarity in [0, 1]
    """
    if not text1 or not text2:
        return 0.0

    t1 = normalize_text(text1)
    t2 = normalize_text(text2)

    if not t1 or not t2:
        return 0.0

    if t1 == t2:
        return 1.0

    if method == "fuzzy":
        return fuzzy_match(t1, t2, normalize=False)

    words1 = set(t1.split())
    words2 = set(t2.split())

    if method == "jaccard":
        if not words1 and not words2:
            return 1.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    if method == "overlap":
        # Overlap coefficient (good for containment)
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        min_size = min(len(words1), len(words2))
        return intersection / min_size

    if method == "cosine":
        all_words = words1 | words2
        if not all_words:
            return 1.0

        vec1 = Counter(t1.split())
        vec2 = Counter(t2.split())

        dot = sum(vec1[w] * vec2[w] for w in all_words)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    if method == "combined":
        fuzzy = fuzzy_match(t1, t2, normalize=False)

        # Jaccard
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0

        # Overlap
        min_size = min(len(words1), len(words2))
        overlap = intersection / min_size if min_size > 0 else 0.0

        # Weighted combination
        return 0.5 * fuzzy + 0.3 * jaccard + 0.2 * overlap

    raise ValueError(f"Unknown method: {method}")


# ============================================================================
# N-GRAM ANALYSIS
# ============================================================================


def get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from text."""
    if not text:
        return []
    if not isinstance(text, str):
        text = str(text)
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def calculate_ngram_repetition(text: str, n: int = 3) -> Tuple[float, List[str]]:
    """
    Calculate ratio of repeated n-grams.

    Args:
        text: Text to analyze
        n: N-gram size

    Returns:
        (repetition_ratio, repeated_ngrams)
    """
    if not text:
        return 0.0, []

    if not isinstance(text, str):
        text = str(text)

    text = normalize_text(text)
    ngrams = get_ngrams(text, n)

    if len(ngrams) < 2:
        return 0.0, []

    counts = Counter(ngrams)
    repeated = [" ".join(ng) for ng, count in counts.items() if count > 1]

    # Calculate ratio of tokens involved in repetition
    repeated_tokens = sum(
        n * (count - 1) for ng, count in counts.items() if count > 1  # Extra occurrences
    )

    total_tokens = len(text.split())
    ratio = repeated_tokens / total_tokens if total_tokens > 0 else 0.0

    return min(1.0, ratio), repeated[:10]


def calculate_phrase_repetition(
    text: str,
    min_length: int = 4,
    max_length: int = 10,
) -> Tuple[float, List[str]]:
    """
    Calculate ratio of repeated phrases.

    Looks for repeated sequences of 4-10 words.

    Returns:
        (repetition_ratio, repeated_phrases)
    """
    if not text:
        return 0.0, []

    if not isinstance(text, str):
        text = str(text)

    text = normalize_text(text)
    words = text.split()

    if len(words) < min_length * 2:
        return 0.0, []

    all_repeated = []
    total_repeated_words = 0

    for n in range(min_length, min(max_length + 1, len(words) // 2)):
        ngrams = get_ngrams(text, n)
        counts = Counter(ngrams)

        for ng, count in counts.items():
            if count > 1:
                phrase = " ".join(ng)
                if phrase not in all_repeated:
                    all_repeated.append(phrase)
                    total_repeated_words += n * (count - 1)

    total_words = len(words)
    ratio = total_repeated_words / total_words if total_words > 0 else 0.0

    return min(1.0, ratio), all_repeated[:10]


def get_unique_token_ratio(text: str) -> float:
    """
    Calculate ratio of unique tokens.

    Higher = more vocabulary diversity.

    Returns:
        Ratio in [0, 1]
    """
    if not text:
        return 1.0

    if not isinstance(text, str):
        text = str(text)

    words = normalize_text(text).split()
    if not words:
        return 1.0

    return len(set(words)) / len(words)


# Alias for compatibility
unique_token_ratio = get_unique_token_ratio


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count using simple heuristics.

    Uses the rule of thumb that ~1 token ≈ 4 characters for English,
    or ~0.75 words per token.

    This is a rough approximation - for exact counts, use a tokenizer.

    Args:
        text: Input text

    Returns:
        Approximate token count
    """
    if not text:
        return 0

    if not isinstance(text, str):
        text = str(text)

    # Method 1: Character-based estimate
    char_estimate = len(text) / 4.0

    # Method 2: Word-based estimate
    words = text.split()
    word_estimate = len(words) * 1.3

    # Take average
    return int((char_estimate + word_estimate) / 2)


def compute_ngram_repetition(text: str, n: int = 3) -> float:
    """
    Compute n-gram repetition ratio.

    Wrapper around calculate_ngram_repetition returning just the ratio.
    """
    ratio, _ = calculate_ngram_repetition(text, n)
    return ratio
