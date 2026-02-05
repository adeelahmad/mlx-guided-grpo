#!/usr/bin/env python3
"""
Exam/Math Questions Reward Module for GRPO Training

Designed for questions with ground truth answers and optional possible_boxed_answers.
Primary signal: ACCURACY (did model get the right answer?)
Secondary signal: FORMAT (did model use <think> and \\boxed{} structure?)

Expected format:
    <think>
    [Concise reasoning]
    </think>

    \\boxed{B}
    [Brief explanation]

New: Supports possible_boxed_answers array for flexible answer matching.
If possible_boxed_answers is provided:
  - Empty list [] = no accuracy reward expected (skip accuracy)
  - Non-empty list = match against any valid answer in list

Usage:
    from exam_reward import compute_reward

    # Simple usage with ground_truth
    score, details = compute_reward(
        completion="<think>...</think>\\boxed{B} explanation",
        ground_truth="B"
    )

    # Advanced usage with possible_boxed_answers
    score, details = compute_reward(
        completion="<think>...</think>\\boxed{BC} explanation",
        ground_truth="BC",
        possible_boxed_answers=["BC", "CB", "B, C", "B and C"]
    )

=== LOGGING CONFIGURATION ===

1. STANDARD LOGGING (human-readable):
    EXAM_REWARD_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: WARNING)
    EXAM_REWARD_LOG_FILE: Path to log file (optional)
    EXAM_REWARD_LOG_CONSOLE: "true"/"false" (default: true)

2. STRUCTURED LOGGING (machine-readable JSONL):
    EXAM_REWARD_STRUCTURED_LOG: Path to JSONL file (enables structured logging)
    EXAM_REWARD_STRUCTURED_CONSOLE: "true" to print JSON to stdout

Structured logs include for each sample:
    - input: completion, ground_truth, possible_answers
    - extraction: model_answer, match_type, has_boxed, thinking_words
    - components: accuracy/boxed/format/reasoning scores with weights
    - penalties: length, bad_words, repeat_words, gaming breakdown
    - reward: raw -> clamped -> smoothed -> final
    - verdict: correct, gaming_detected, gaming_patterns, final_reward

Example JSONL output:
    {"timestamp":"2026-02-01T15:23:51","event":"reward_computed","sample_id":1,
     "input":{"completion":"<think>...</think>\\boxed{B}","ground_truth":"B"},
     "extraction":{"model_answer":"B","match_type":"exact","has_boxed":true},
     "components":{"accuracy":{"score":1.0,"weight":0.75,"correct":true},...},
     "penalties":{"total":0.0,"gaming":{"total":0,"breakdown":{}}},
     "reward":{"raw":0.98,"final":0.87},
     "verdict":{"correct":true,"gaming_detected":false,"final_reward":0.87}}

IMPORTANT: Set environment variables BEFORE importing this module!

    # Standard logging
    export EXAM_REWARD_LOG_LEVEL=DEBUG
    export EXAM_REWARD_LOG_FILE=/path/to/exam_reward.log

    # Structured logging (recommended for analysis)
    export EXAM_REWARD_STRUCTURED_LOG=/path/to/rewards.jsonl

Or reconfigure after import:
    from exam_reward import reconfigure_logging, reconfigure_structured_logging
    import os
    os.environ["EXAM_REWARD_STRUCTURED_LOG"] = "/path/to/rewards.jsonl"
    reconfigure_structured_logging()
"""

import json
import logging
import math
import os
import re
import sys
from collections import Counter  # FIX #7: Moved from inside loop to top level
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

# =============================================================================
# Pre-compiled Patterns (FIX #8: Compile regex patterns at module level)
# =============================================================================

# Hedging patterns for gaming detection
_HEDGING_PATTERNS = [
    re.compile(r"\b[A-D]\s+(or|and)\s+[A-D]\b", re.IGNORECASE),  # "A or B"
    re.compile(r"\b[A-D]\s*/\s*[A-D]\b", re.IGNORECASE),  # "A/B"
    re.compile(r"\b[A-D]\s*,\s*[A-D]\b", re.IGNORECASE),  # "A, B"
    re.compile(r"\b(either|both)\b", re.IGNORECASE),  # "either" or "both"
    re.compile(r"\b[A-D]\s+or\s+[A-D]\s+or\s+[A-D]\b", re.IGNORECASE),  # "A or B or C"
]

# Strong conclusion patterns for inconsistency detection
_STRONG_CONCLUSION_PATTERNS = [
    re.compile(
        r'(the\s+answer\s+is|correct\s+answer\s+is|therefore|thus|hence|so\s+the\s+answer)\s+["\']?([A-D])["\']?',
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-D])\s+(is\s+correct|is\s+the\s+(right|correct)\s+(answer|choice))", re.IGNORECASE
    ),
]


# =============================================================================
# Structured Logging Configuration
# =============================================================================

# FIX #9: Typo "rolouts" -> "rollouts"
_DEFAULT_STRUCTURED_LOG_FILE = "rollouts_reward_log.jsonl"


class StructuredRewardLogger:
    """
    Structured JSON logger for reward computation.

    Outputs machine-readable JSONL format for easy analysis.

    Environment variables:
        EXAM_REWARD_STRUCTURED_LOG: Path to JSONL file (enables structured logging)
        EXAM_REWARD_STRUCTURED_CONSOLE: "true" to also print JSON to console

    Each log entry contains:
        - timestamp: ISO format timestamp
        - event: Event type (reward_computed, batch_complete, etc.)
        - input: Completion, ground truth, possible answers
        - extraction: What was extracted (model answer, thinking, etc.)
        - scores: Raw component scores
        - weights: Applied weights
        - penalties: All penalties with breakdown
        - computation: Step-by-step reward calculation
        - result: Final reward and correctness
    """

    def __init__(self):
        # FIX #2: Consistent defaults - use same default everywhere
        self.log_file = os.environ.get("EXAM_REWARD_STRUCTURED_LOG", _DEFAULT_STRUCTURED_LOG_FILE)
        self.log_console = (
            os.environ.get("EXAM_REWARD_STRUCTURED_CONSOLE", "false").lower() == "true"
        )
        self.enabled = bool(self.log_file) or self.log_console
        self._file_handle = None
        self._sample_counter = 0
        self._batch_counter = 0

    def __del__(self):
        """FIX #3: Prevent file handle leak on garbage collection."""
        self.close()

    def _ensure_file(self):
        """Open log file if not already open."""
        if self.log_file and self._file_handle is None:
            try:
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                self._file_handle = open(self.log_file, "a", encoding="utf-8")
            except (IOError, OSError) as e:
                sys.stderr.write(
                    f"Warning: Could not open structured log file {self.log_file}: {e}\n"
                )
                self.log_file = ""

    def _write(self, entry: Dict[str, Any]):
        """Write a log entry."""
        if not self.enabled:
            return

        json_line = json.dumps(entry, ensure_ascii=False, default=str)

        if self.log_console:
            print(json_line, flush=True)

        if self.log_file:
            self._ensure_file()
            if self._file_handle:
                self._file_handle.write(json_line + "\n")
                self._file_handle.flush()

    def log_reward(
        self,
        completion: str,
        ground_truth: Any,
        possible_answers: Optional[List[Any]],
        prompt_text: str,
        details: Dict[str, Any],
        final_reward: float,
        weights: "RewardWeights",
        iteration: Optional[int] = None,
        sample_id: Optional[int] = None,
    ):
        """
        Log a single reward computation with full details.

        Output format is designed for easy analysis:
        - input: What went in (completion, ground_truth)
        - extraction: What was extracted (model_answer, thinking_words)
        - components: Score breakdown with weights
        - penalties: All penalties applied
        - reward: Final computation chain
        - verdict: Quick summary (correct/incorrect, issues detected)
        """
        if not self.enabled:
            return

        self._sample_counter += 1

        # Extract key info from details
        accuracy_info = details.get("accuracy", {})
        format_info = details.get("format", {})
        boxed_info = details.get("boxed", {})
        reasoning_info = details.get("reasoning", {})
        gaming_info = details.get("gaming_penalty", {})
        penalties_info = details.get("penalties", {})

        # Get nested details safely
        acc_details = accuracy_info.get("details", {}) if isinstance(accuracy_info, dict) else {}
        box_details = boxed_info.get("details", {}) if isinstance(boxed_info, dict) else {}
        fmt_details = format_info.get("details", {}) if isinstance(format_info, dict) else {}
        reas_details = reasoning_info.get("details", {}) if isinstance(reasoning_info, dict) else {}

        # Gaming breakdown
        gaming_breakdown = gaming_info.get("breakdown", {}) if isinstance(gaming_info, dict) else {}
        gaming_patterns = [k for k, v in gaming_breakdown.items() if v > 0]

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "reward_computed",
            "sample_id": sample_id if sample_id is not None else self._sample_counter,
            "iteration": iteration,
            # === INPUT ===
            "input": {
                "completion": completion[:500] + "..." if len(completion) > 500 else completion,
                "completion_len": len(completion),
                "ground_truth": str(ground_truth) if ground_truth is not None else None,
                "possible_answers": [str(a) for a in (possible_answers or [])],
                "prompt": (
                    prompt_text[:300] + "..."
                    if len(prompt_text) > 300
                    else prompt_text if prompt_text else None
                ),
            },
            # === EXTRACTION ===
            "extraction": {
                "model_answer": acc_details.get("model_answer"),
                "matched_answer": acc_details.get("matched_answer"),
                "match_type": acc_details.get("match_type"),
                "has_boxed": box_details.get("has_boxed", False),
                "has_think_tags": fmt_details.get("has_think_tags", False),
                "thinking_words": reas_details.get("thinking_words", 0),
                "answer_words": reas_details.get("answer_words", 0),
            },
            # === COMPONENT SCORES ===
            "components": {
                "accuracy": {
                    "score": (
                        accuracy_info.get("original_score", 0)
                        if isinstance(accuracy_info, dict)
                        else 0
                    ),
                    "weight": weights.accuracy,
                    "weighted": (
                        accuracy_info.get("weighted", 0) if isinstance(accuracy_info, dict) else 0
                    ),
                    "correct": acc_details.get("correct"),
                    "skipped": acc_details.get("skipped", False),
                },
                "boxed": {
                    "score": boxed_info.get("score", 0) if isinstance(boxed_info, dict) else 0,
                    "weight": weights.boxed,
                    "weighted": (
                        boxed_info.get("weighted", 0) if isinstance(boxed_info, dict) else 0
                    ),
                },
                "format": {
                    "score": format_info.get("score", 0) if isinstance(format_info, dict) else 0,
                    "weight": weights.format,
                    "weighted": (
                        format_info.get("weighted", 0) if isinstance(format_info, dict) else 0
                    ),
                },
                "reasoning": {
                    "score": (
                        reasoning_info.get("score", 0) if isinstance(reasoning_info, dict) else 0
                    ),
                    "weight": weights.reasoning,
                    "weighted": (
                        reasoning_info.get("weighted", 0) if isinstance(reasoning_info, dict) else 0
                    ),
                    "thinking_words": reas_details.get("thinking_words", 0),
                },
            },
            # === PENALTIES ===
            "penalties": {
                "total": penalties_info.get("total", 0),
                "multiplier": penalties_info.get("multiplier", 1.0),
                "length": penalties_info.get("length", 0),
                "bad_words": penalties_info.get("bad_words", 0),
                "repeat_words": penalties_info.get("repeat_words", 0),
                "gaming": {
                    "total": gaming_info.get("total", 0) if isinstance(gaming_info, dict) else 0,
                    "breakdown": gaming_breakdown,
                },
            },
            # === REWARD COMPUTATION ===
            "reward": {
                "raw": details.get("raw_score", 0),
                "penalized": details.get("penalized_score", 0),
                "clamped": details.get("clamped_score", 0),
                "smoothed": details.get("smoothing", {}).get("post_smooth", 0),
                "final": final_reward,
            },
            # === VERDICT (quick summary) ===
            "verdict": {
                "correct": acc_details.get("correct"),
                "format_ok": fmt_details.get("has_think_tags", False)
                and box_details.get("has_boxed", False),
                "format_gate_applied": details.get("format_gate", {}).get("gate_applied", False),
                "gaming_detected": len(gaming_patterns) > 0,
                "gaming_patterns": gaming_patterns,
                "total_penalty": penalties_info.get("total", 0),
                "final_reward": final_reward,
            },
        }

        self._write(entry)

    def log_batch_summary(
        self,
        rewards: List[float],
        correct_count: int,
        total_count: int,
        iteration: Optional[int] = None,
    ):
        """Log batch processing summary."""
        if not self.enabled:
            return

        self._batch_counter += 1

        import numpy as np

        rewards_arr = np.array(rewards)

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": "batch_complete",
            "batch_id": self._batch_counter,
            "iteration": iteration,
            "summary": {
                "total_samples": total_count,
                "correct_count": correct_count,
                "accuracy_rate": correct_count / total_count if total_count > 0 else 0,
                "reward_mean": float(rewards_arr.mean()),
                "reward_std": float(rewards_arr.std()),
                "reward_min": float(rewards_arr.min()),
                "reward_max": float(rewards_arr.max()),
                "reward_median": float(np.median(rewards_arr)),
            },
            "distribution": {
                "0.0-0.2": int(np.sum((rewards_arr >= 0.0) & (rewards_arr < 0.2))),
                "0.2-0.4": int(np.sum((rewards_arr >= 0.2) & (rewards_arr < 0.4))),
                "0.4-0.6": int(np.sum((rewards_arr >= 0.4) & (rewards_arr < 0.6))),
                "0.6-0.8": int(np.sum((rewards_arr >= 0.6) & (rewards_arr < 0.8))),
                "0.8-1.0": int(np.sum((rewards_arr >= 0.8) & (rewards_arr <= 1.0))),
            },
        }

        self._write(entry)

    def close(self):
        """Close the log file."""
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self._file_handle = None

    def reconfigure(self):
        """Reconfigure from environment variables."""
        self.close()
        # FIX #2: Use consistent default
        self.log_file = os.environ.get("EXAM_REWARD_STRUCTURED_LOG", _DEFAULT_STRUCTURED_LOG_FILE)
        self.log_console = (
            os.environ.get("EXAM_REWARD_STRUCTURED_CONSOLE", "false").lower() == "true"
        )
        self.enabled = bool(self.log_file) or self.log_console


# Global structured logger instance
_structured_logger = StructuredRewardLogger()


def reconfigure_structured_logging():
    """Reconfigure structured logging from environment variables."""
    global _structured_logger
    _structured_logger.reconfigure()


# =============================================================================
# Standard Logging Configuration
# =============================================================================


def _configure_logging(force_reconfigure: bool = False) -> logging.Logger:
    """
    Configure logging based on environment variables.

    Environment variables:
        EXAM_REWARD_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: WARNING)
        EXAM_REWARD_LOG_FILE: Path to log file (optional)
        EXAM_REWARD_LOG_CONSOLE: "true"/"false" - log to console (default: true)

    Args:
        force_reconfigure: If True, reconfigure even if handlers exist

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("exam_reward")

    # Get configuration from environment
    log_level_str = os.environ.get("EXAM_REWARD_LOG_LEVEL", "WARNING").upper()
    log_file = os.environ.get("EXAM_REWARD_LOG_FILE", "")
    log_console = os.environ.get("EXAM_REWARD_LOG_CONSOLE", "true").lower() == "true"

    # Parse log level
    log_level = getattr(logging, log_level_str, logging.WARNING)

    # Check if already configured with same settings (avoid duplicate handlers)
    if logger.handlers and not force_reconfigure:
        # Update level even if handlers exist (allows runtime level changes)
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)
        return logger

    # Clear existing handlers if reconfiguring
    if force_reconfigure:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    # Set level
    logger.setLevel(log_level)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add file handler if specified
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (IOError, OSError) as e:
            # Fall back to console if file can't be opened
            log_console = True
            sys.stderr.write(f"Warning: Could not open log file {log_file}: {e}\n")

    # Add console handler
    if log_console or not log_file:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Log configuration for debugging
    if log_level <= logging.DEBUG:
        logger.debug(
            f"Logging configured: level={log_level_str}, file={log_file or 'None'}, console={log_console}"
        )

    return logger


def reconfigure_logging() -> logging.Logger:
    """
    Force reconfiguration of logging based on current environment variables.

    Call this after changing environment variables to apply new settings.

    Example:
        import os
        os.environ["EXAM_REWARD_LOG_LEVEL"] = "DEBUG"
        os.environ["EXAM_REWARD_LOG_FILE"] = "/path/to/log.log"

        from exam_reward import reconfigure_logging
        reconfigure_logging()

    Returns:
        Reconfigured logger instance
    """
    global _logger
    _logger = _configure_logging(force_reconfigure=True)
    return _logger


# Initialize logger
_logger = _configure_logging()


# =============================================================================
# Reward Weights
# =============================================================================


@dataclass
class RewardWeights:
    """
    Reward component weights for GRPO training.

    OPTIMIZED based on rollout analysis (1396 samples, 613 iterations):

    Discriminative Power Analysis:
        - accuracy:  gap=0.824 (correct: 1.000, incorrect: 0.176) → ✅ DISCRIMINATIVE
        - boxed:     gap=0.000 (correct: 1.000, incorrect: 1.000) → ❌ NOT DISCRIMINATIVE
        - format:    gap=-0.001 (correct: 0.994, incorrect: 0.995) → ❌ NOT DISCRIMINATIVE
        - reasoning: gap=0.027 (correct: 0.766, incorrect: 0.739) → ❌ NOT DISCRIMINATIVE

    OPTIMIZED WEIGHTS (based on discriminative power):
        - accuracy: 0.75 (75%) - ONLY discriminative signal, maximize weight!
        - boxed: 0.02 (2%) - Already mastered (100%), minimal weight
        - format: 0.03 (3%) - Already mastered (99.5%), minimal weight
        - reasoning: 0.20 (20%) - Weak discrimination, reduce but keep for format incentive

    Total sums to 1.0 for normalized scores.
    """

    # FIX #14: Docstring now matches these values
    accuracy: float = 0.75  # Primary signal - only discriminative component!
    boxed: float = 0.02  # Already mastered by model
    format: float = 0.03  # Already mastered by model
    reasoning: float = 0.20  # Not discriminative (gap=0.027)

    def __post_init__(self) -> None:
        """Validate weights after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate that weights are reasonable."""
        for name in ["accuracy", "boxed", "format", "reasoning"]:
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"Weight '{name}' must be numeric, got {type(value)}")
            if value < 0:
                raise ValueError(f"Weight '{name}' must be non-negative, got {value}")

        total = self.total()
        if total <= 0:
            raise ValueError(f"Total weights must be positive, got {total}")

    def total(self) -> float:
        """Return sum of all weights."""
        return self.accuracy + self.boxed + self.format + self.reasoning


@dataclass
class FormatConfig:
    """
    Configuration for expected completion format and reward shaping.

    Allows customization of:
    - Thinking tags (default: <think>...</think>)
    - Answer tags (default: None/disabled)
    - Boxed format (default: \\boxed{})
    - Format gate mode (default: strict)
    - Length penalties (default: disabled)
    - Bad words penalties (default: disabled)
    - Repeat words penalties (default: disabled)
    - Reward smoothing (default: disabled)

    Examples:
        # Default config (backward compatible)
        config = FormatConfig()

        # With bad words penalty
        config = FormatConfig(
            bad_words=["obviously", "clearly", "simply"],
            bad_word_penalty=0.05,  # Per occurrence
        )

        # With repeat words penalty
        config = FormatConfig(
            repeat_word_threshold=3,  # Allow 3 repeats, penalize after
            repeat_word_penalty=0.02,
        )
    """

    # === Thinking tag configuration ===
    think_tag: str = "think"  # Primary thinking tag (without < >)
    alt_think_tags: List[str] = field(default_factory=list)  # Alternative tags

    # === Answer tag configuration ===
    answer_tag: Optional[str] = None  # e.g., "answer" for <answer>...</answer>

    # === Boxed format configuration ===
    boxed_format: str = "\\boxed{}"  # The expected boxed format pattern
    alt_boxed_formats: List[str] = field(default_factory=list)  # Alternatives

    # === Format gate configuration ===
    strict_format: bool = True  # If True, format acts as a gate on accuracy
    format_gate_threshold: float = 0.5  # Min format score to pass gate
    format_gate_penalty: float = 0.3  # Max accuracy multiplier when gate fails

    # === Length penalty configuration ===
    min_thinking_words: int = 0  # Minimum words in thinking (0 = disabled)
    max_thinking_words: int = 0  # Maximum words in thinking (0 = disabled)
    max_answer_words: int = 0  # Maximum words after boxed (0 = disabled)
    length_penalty_weight: float = 0.1  # Weight of length penalty (0-1)
    length_penalty_type: str = "soft"  # "soft" (gradual) or "hard" (binary)

    # === Bad words penalty configuration ===
    bad_words: List[str] = field(default_factory=list)  # Words to penalize (empty = disabled)
    bad_word_penalty: float = 0.05  # Penalty per occurrence
    bad_word_max_penalty: float = 0.5  # Maximum total bad word penalty
    bad_word_case_sensitive: bool = False  # Case-insensitive by default

    # === Repeat words penalty configuration ===
    repeat_word_threshold: int = 0  # 0 = disabled, N = allow N repeats before penalty
    repeat_word_penalty: float = 0.02  # Penalty per excess repetition
    repeat_word_max_penalty: float = 0.3  # Maximum total repeat penalty
    repeat_word_min_length: int = 4  # Only count words with >= this many chars
    repeat_word_ignore: List[str] = field(
        default_factory=lambda: [
            # Common stopwords that naturally repeat
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
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
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "as",
            "at",
            "by",
            "from",
            "not",
            "no",
            "if",
            "then",
            "else",
            "when",
            "where",
            "which",
            "who",
            "what",
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
            "than",
            "too",
            "very",
            "just",
            "also",
            "only",
            # Technical/exam terms that naturally repeat
            "option",
            "answer",
            "question",
            "choice",
            "correct",
            "incorrect",
            "true",
            "false",
            "yes",
            "no",
            "data",
            "service",
            "services",
            "using",
            "used",
            "use",
            "need",
            "needs",
            "required",
            "requires",
            "provide",
            "provides",
            "allow",
            "allows",
            "enable",
            "enables",
            "configuration",
            "configure",
            "configured",
            "setting",
            "settings",
            "instance",
            "instances",
            "resource",
            "resources",
            "request",
            "requests",
            "response",
            "responses",
            "access",
            "policy",
            "policies",
            "role",
            "roles",
            "user",
            "users",
            "group",
            "groups",
            "region",
            "regions",
            "account",
            "bucket",
            "buckets",
            "function",
            "functions",
            "table",
            "tables",
            "database",
            "storage",
            "network",
            "security",
            "application",
        ]
    )

    # === Reward smoothing configuration ===
    reward_smoothing: str = "none"  # "none", "clip", "sigmoid", "tanh"
    reward_min: float = 0.0  # Minimum reward (for clip)
    reward_max: float = 1.0  # Maximum reward (for clip)
    reward_temp: float = 1.0  # Temperature for sigmoid/tanh (higher = sharper)
    reward_center: float = 0.5  # Center point for sigmoid/tanh

    # === Gaming detection configuration (all disabled by default) ===
    detect_gaming: bool = False  # Master switch for gaming detection

    # Multiple boxed answers (model hedging by giving multiple answers)
    multiple_boxed_penalty: float = 0.3  # Penalty for multiple \boxed{}

    # Answer shopping (listing all options without deciding)
    answer_shopping_penalty: float = 0.15  # Penalty for mentioning all ABCD equally
    answer_shopping_threshold: int = 4  # How many options triggers penalty

    # Hedge words (excessive uncertainty language)
    hedge_words: List[str] = field(
        default_factory=lambda: [
            "maybe",
            "possibly",
            "perhaps",
            "might",
            "probably",
            "likely",
            "unlikely",
            "seems",
            "appears",
            "somewhat",
            "fairly",
            "arguably",
        ]
    )
    hedge_word_threshold: int = 5  # Allow this many before penalty
    hedge_word_penalty: float = 0.02  # Per excess hedge word
    hedge_word_max_penalty: float = 0.2  # Cap total hedge penalty

    # Repetitive phrases (same 3+ word phrase repeated - gaming length)
    repetitive_phrase_threshold: int = 3  # Times a phrase can repeat before penalty
    repetitive_phrase_min_words: int = 3  # Minimum words in phrase to count
    repetitive_phrase_penalty: float = 0.1  # Per repeated phrase type
    repetitive_phrase_max_penalty: float = 0.3  # Cap

    # Garbage/invalid content (non-printable characters)
    garbage_char_threshold: int = 3  # Allow some non-ASCII (for math symbols)
    garbage_char_penalty: float = 0.05  # Per excess garbage char
    garbage_char_max_penalty: float = 0.3  # Cap

    # Confidence gaming (repeating the answer many times in thinking)
    answer_repeat_threshold: int = 5  # Times answer can appear in thinking
    answer_repeat_penalty: float = 0.15  # Penalty for excessive answer repetition

    # Filler/fluff phrases (model padding with empty phrases)
    filler_phrases: List[str] = field(
        default_factory=lambda: [
            "let me think",
            "let me analyze",
            "let me consider",
            "i need to think",
            "i need to analyze",
            "i need to consider",
            "this is a great question",
            "this is an interesting question",
            "this is a good question",
            "great question",
            "good question",
            "step by step",
            "one by one",
            "carefully consider",
            "i will analyze",
            "i will consider",
            "i will think",
            "let's break this down",
            "let's analyze",
            "let's think",
            "thinking about this",
            "considering this",
            "analyzing this",
            "hmm",
            "hmmm",
            "well let me",
            "okay so",
            "ok so",
            "alright",
            "interesting",
            "i see",
            "now let me",
            "first let me",
        ]
    )
    filler_phrase_threshold: int = 3  # Penalize if more than N filler phrases
    filler_phrase_penalty: float = 0.05  # Per excess filler phrase
    filler_phrase_max_penalty: float = 0.2  # Cap

    # Answer hedging in boxed (e.g., \boxed{A or B})
    boxed_hedging_penalty: float = 0.25  # Penalty for hedging in answer

    # Prompt copying (model copies question into thinking to pad)
    prompt_copy_threshold: float = 0.4  # Fraction of prompt words in thinking
    prompt_copy_penalty: float = 0.2  # Penalty for excessive copying

    # Low entropy/repetitive text (garbage detection)
    min_unique_word_ratio: float = 0.25  # Min ratio unique words/total words
    low_entropy_penalty: float = 0.2  # Penalty for low diversity text

    # Reasoning-answer inconsistency (reasoning says X, answer says Y)
    inconsistency_penalty: float = 0.15  # Penalty for contradicting reasoning

    # === Pre-compiled patterns (set in __post_init__) ===
    _think_patterns: List[Tuple[str, re.Pattern]] = field(default_factory=list, repr=False)
    _boxed_patterns: List[Tuple[str, re.Pattern, re.Pattern]] = field(
        default_factory=list, repr=False
    )
    _bad_words_set: frozenset = field(default_factory=frozenset, repr=False)
    _repeat_ignore_set: frozenset = field(default_factory=frozenset, repr=False)
    _hedge_words_set: frozenset = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and pre-compile regex patterns."""
        # Validation
        if not self.think_tag:
            raise ValueError("think_tag cannot be empty")
        if self.format_gate_threshold < 0 or self.format_gate_threshold > 1:
            raise ValueError(
                f"format_gate_threshold must be in [0,1], got {self.format_gate_threshold}"
            )
        if self.format_gate_penalty < 0 or self.format_gate_penalty > 1:
            raise ValueError(
                f"format_gate_penalty must be in [0,1], got {self.format_gate_penalty}"
            )
        if self.length_penalty_weight < 0 or self.length_penalty_weight > 1:
            raise ValueError(
                f"length_penalty_weight must be in [0,1], got {self.length_penalty_weight}"
            )
        if self.reward_smoothing not in ("none", "clip", "sigmoid", "tanh"):
            raise ValueError(
                f"reward_smoothing must be none/clip/sigmoid/tanh, got {self.reward_smoothing}"
            )
        if self.length_penalty_type not in ("soft", "hard"):
            raise ValueError(
                f"length_penalty_type must be soft/hard, got {self.length_penalty_type}"
            )
        if self.reward_temp <= 0:
            raise ValueError(f"reward_temp must be positive, got {self.reward_temp}")
        if self.bad_word_penalty < 0:
            raise ValueError(f"bad_word_penalty must be non-negative, got {self.bad_word_penalty}")
        if self.bad_word_max_penalty < 0:
            raise ValueError(
                f"bad_word_max_penalty must be non-negative, got {self.bad_word_max_penalty}"
            )
        if self.repeat_word_threshold < 0:
            raise ValueError(
                f"repeat_word_threshold must be non-negative, got {self.repeat_word_threshold}"
            )
        if self.repeat_word_penalty < 0:
            raise ValueError(
                f"repeat_word_penalty must be non-negative, got {self.repeat_word_penalty}"
            )
        if self.repeat_word_min_length < 1:
            raise ValueError(
                f"repeat_word_min_length must be >= 1, got {self.repeat_word_min_length}"
            )

        # Pre-compile patterns and sets for efficiency
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns and word sets for maximum efficiency."""
        # Think tag patterns
        self._think_patterns = []
        for tag in self.get_all_think_tags():
            pattern = re.compile(
                rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.IGNORECASE | re.DOTALL
            )
            self._think_patterns.append((tag, pattern))

        # Boxed patterns
        self._boxed_patterns = []
        for fmt in self.get_all_boxed_formats():
            match = re.match(r"\\(\w+)\{\}", fmt)
            if match:
                cmd = match.group(1)
                # Pattern for checking presence
                check_pattern = re.compile(rf"\\{cmd}\{{", re.IGNORECASE)
                # Pattern for extracting content
                extract_pattern = re.compile(rf"\\{cmd}\{{([^}}]*)\}}", re.IGNORECASE)
                self._boxed_patterns.append((cmd, check_pattern, extract_pattern))

        # Bad words set (frozen for thread safety and fast lookup)
        if self.bad_words:
            if self.bad_word_case_sensitive:
                self._bad_words_set = frozenset(self.bad_words)
            else:
                self._bad_words_set = frozenset(w.lower() for w in self.bad_words)
        else:
            self._bad_words_set = frozenset()

        # Repeat ignore set (always lowercase for comparison)
        self._repeat_ignore_set = frozenset(w.lower() for w in self.repeat_word_ignore)

        # Hedge words set (always lowercase)
        self._hedge_words_set = frozenset(w.lower() for w in self.hedge_words)

    def get_all_think_tags(self) -> List[str]:
        """Return all accepted thinking tags."""
        return [self.think_tag] + list(self.alt_think_tags)

    def get_all_boxed_formats(self) -> List[str]:
        """Return all accepted boxed formats."""
        return [self.boxed_format] + list(self.alt_boxed_formats)

    def compute_length_penalty(self, thinking_words: int, answer_words: int) -> float:
        """
        Compute length penalty based on configuration.

        Returns penalty in [0, 1] where 0 = no penalty, 1 = full penalty.
        Uses soft penalties by default for smooth gradients.
        """
        if self.length_penalty_weight <= 0:
            return 0.0

        penalties = []

        # Min thinking penalty
        if self.min_thinking_words > 0 and thinking_words < self.min_thinking_words:
            if self.length_penalty_type == "hard":
                penalties.append(1.0)
            else:
                # Soft: linear ramp from 0 (at min) to 1 (at 0)
                ratio = thinking_words / self.min_thinking_words
                penalties.append(1.0 - ratio)

        # Max thinking penalty
        if self.max_thinking_words > 0 and thinking_words > self.max_thinking_words:
            if self.length_penalty_type == "hard":
                penalties.append(1.0)
            else:
                # Soft: gradual increase beyond max
                excess = thinking_words - self.max_thinking_words
                # Penalty grows but caps at 1.0
                penalties.append(min(1.0, excess / self.max_thinking_words))

        # Max answer penalty
        if self.max_answer_words > 0 and answer_words > self.max_answer_words:
            if self.length_penalty_type == "hard":
                penalties.append(1.0)
            else:
                excess = answer_words - self.max_answer_words
                penalties.append(min(1.0, excess / self.max_answer_words))

        if not penalties:
            return 0.0

        # Return weighted average of penalties
        return self.length_penalty_weight * (sum(penalties) / len(penalties))

    def compute_bad_words_penalty(self, text: str) -> Tuple[float, Dict[str, int]]:
        """
        Compute penalty for bad words in text.

        Args:
            text: Text to check for bad words

        Returns:
            (penalty, word_counts) where penalty is in [0, max_penalty]
            and word_counts is dict of {bad_word: count}
        """
        if not self._bad_words_set or not text:
            return 0.0, {}

        # Tokenize efficiently - only alphanumeric words
        words = re.findall(r"\b[a-zA-Z]+\b", text)

        # Count bad word occurrences
        bad_word_counts: Dict[str, int] = {}
        total_bad = 0

        for word in words:
            check_word = word if self.bad_word_case_sensitive else word.lower()
            if check_word in self._bad_words_set:
                bad_word_counts[check_word] = bad_word_counts.get(check_word, 0) + 1
                total_bad += 1

        if total_bad == 0:
            return 0.0, {}

        # Calculate penalty (capped at max)
        raw_penalty = total_bad * self.bad_word_penalty
        penalty = min(raw_penalty, self.bad_word_max_penalty)

        return penalty, bad_word_counts

    def compute_repeat_words_penalty(self, text: str) -> Tuple[float, Dict[str, int]]:
        """
        Compute penalty for excessively repeated words.

        Args:
            text: Text to check for repeated words

        Returns:
            (penalty, excess_counts) where penalty is in [0, max_penalty]
            and excess_counts is dict of {word: excess_count}
        """
        if self.repeat_word_threshold <= 0 or not text:
            return 0.0, {}

        # Tokenize - only words meeting minimum length
        words = re.findall(r"\b[a-zA-Z]+\b", text)

        # Count word frequencies (case-insensitive)
        word_counts: Dict[str, int] = {}
        for word in words:
            lower_word = word.lower()
            # Skip short words and ignored words
            if (
                len(lower_word) >= self.repeat_word_min_length
                and lower_word not in self._repeat_ignore_set
            ):
                word_counts[lower_word] = word_counts.get(lower_word, 0) + 1

        # Find words exceeding threshold
        excess_counts: Dict[str, int] = {}
        total_excess = 0

        for word, count in word_counts.items():
            if count > self.repeat_word_threshold:
                excess = count - self.repeat_word_threshold
                excess_counts[word] = excess
                total_excess += excess

        if total_excess == 0:
            return 0.0, {}

        # Calculate penalty (capped at max)
        raw_penalty = total_excess * self.repeat_word_penalty
        penalty = min(raw_penalty, self.repeat_word_max_penalty)

        return penalty, excess_counts

    def compute_gaming_penalties(
        self,
        completion: str,
        thinking_text: str,
        boxed_answer: Optional[str],
        prompt_text: str = "",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Detect and penalize gaming behaviors in model output.

        Gaming behaviors detected:
        - Multiple boxed answers (hedging)
        - Answer shopping (listing all options equally)
        - Excessive hedge words
        - Repetitive phrases
        - Garbage characters
        - Answer repetition in thinking
        - Filler/fluff phrases
        - Boxed answer hedging (A or B)
        - Prompt copying
        - Low entropy/garbage text
        - Reasoning-answer inconsistency

        Args:
            completion: Full completion text
            thinking_text: Extracted thinking section
            boxed_answer: Extracted boxed answer (if any)
            prompt_text: Original prompt (for copy detection)

        Returns:
            (total_penalty, details) where details contains per-type breakdowns
        """
        if not self.detect_gaming:
            return 0.0, {"gaming_detection": "disabled"}

        penalties = {}
        details = {}
        completion_lower = completion.lower()
        thinking_lower = thinking_text.lower() if thinking_text else ""

        # 1. Multiple boxed answers
        boxed_matches = re.findall(r"\\boxed\{[^}]*\}", completion)
        if len(boxed_matches) > 1:
            penalties["multiple_boxed"] = self.multiple_boxed_penalty
            details["multiple_boxed"] = {"count": len(boxed_matches), "answers": boxed_matches[:5]}

        # 2. Answer shopping (all options A, B, C, D mentioned)
        if thinking_lower:
            options_found = set()
            for letter in ["a", "b", "c", "d"]:
                patterns = [
                    f"option {letter}",
                    f"choice {letter}",
                    f"({letter})",
                    f" {letter}.",
                    f" {letter}:",
                    f" {letter} is",
                    f" {letter} -",
                ]
                for pattern in patterns:
                    if pattern in thinking_lower:
                        options_found.add(letter.upper())
                        break

            if len(options_found) >= self.answer_shopping_threshold:
                penalties["answer_shopping"] = self.answer_shopping_penalty
                details["answer_shopping"] = {
                    "options_found": sorted(list(options_found)),
                    "threshold": self.answer_shopping_threshold,
                }

        # 3. Hedge words
        if self._hedge_words_set and self.hedge_word_threshold > 0:
            words = completion_lower.split()
            hedge_count = sum(1 for w in words if w.strip(".,!?;:()[]") in self._hedge_words_set)
            if hedge_count > self.hedge_word_threshold:
                excess = hedge_count - self.hedge_word_threshold
                raw_penalty = excess * self.hedge_word_penalty
                penalties["hedge_words"] = min(raw_penalty, self.hedge_word_max_penalty)
                details["hedge_words"] = {
                    "count": hedge_count,
                    "excess": excess,
                    "threshold": self.hedge_word_threshold,
                }

        # 4. Repetitive phrases (n-grams repeated multiple times)
        # FIX #7: Counter already imported at top of file
        if self.repetitive_phrase_threshold > 0 and thinking_lower:
            words = thinking_lower.split()
            if len(words) >= self.repetitive_phrase_min_words:
                ngrams = []
                for n in range(self.repetitive_phrase_min_words, min(6, len(words))):
                    for i in range(len(words) - n + 1):
                        ngram = " ".join(words[i : i + n])
                        if len(ngram) > 10:
                            ngrams.append(ngram)

                ngram_counts = Counter(ngrams)
                repeated = {
                    ng: c for ng, c in ngram_counts.items() if c >= self.repetitive_phrase_threshold
                }

                if repeated:
                    num_repeated_types = len(repeated)
                    raw_penalty = num_repeated_types * self.repetitive_phrase_penalty
                    penalties["repetitive_phrases"] = min(
                        raw_penalty, self.repetitive_phrase_max_penalty
                    )
                    details["repetitive_phrases"] = {
                        "repeated_phrases": dict(list(repeated.items())[:5]),
                        "total_types": num_repeated_types,
                    }

        # 5. Garbage characters (non-printable, weird Unicode)
        if self.garbage_char_threshold > 0:
            garbage = re.findall(
                r"[^\x20-\x7E\n\r\t\u00A0-\u00FF\u2018-\u201F\u2013-\u2014\u2026]", completion
            )
            if len(garbage) > self.garbage_char_threshold:
                excess = len(garbage) - self.garbage_char_threshold
                raw_penalty = excess * self.garbage_char_penalty
                penalties["garbage_chars"] = min(raw_penalty, self.garbage_char_max_penalty)
                details["garbage_chars"] = {
                    "count": len(garbage),
                    "excess": excess,
                    "sample": garbage[:10],
                }

        # 6. Answer repetition in thinking (confidence gaming)
        if boxed_answer and thinking_lower and self.answer_repeat_threshold > 0:
            answer_lower = boxed_answer.lower().strip()
            if len(answer_lower) <= 3:
                patterns = [
                    rf"\b{re.escape(answer_lower)}\b",
                    rf"\({re.escape(answer_lower)}\)",
                    rf"option\s+{re.escape(answer_lower)}",
                ]
                total_count = 0
                for pattern in patterns:
                    total_count += len(re.findall(pattern, thinking_lower, re.IGNORECASE))

                if total_count > self.answer_repeat_threshold:
                    penalties["answer_repetition"] = self.answer_repeat_penalty
                    details["answer_repetition"] = {
                        "answer": boxed_answer,
                        "count": total_count,
                        "threshold": self.answer_repeat_threshold,
                    }

        # 7. Filler/fluff phrases
        if self.filler_phrases and self.filler_phrase_threshold > 0 and thinking_lower:
            filler_count = 0
            found_fillers = []
            for phrase in self.filler_phrases:
                count = thinking_lower.count(phrase.lower())
                if count > 0:
                    filler_count += count
                    found_fillers.append((phrase, count))

            if filler_count > self.filler_phrase_threshold:
                excess = filler_count - self.filler_phrase_threshold
                raw_penalty = excess * self.filler_phrase_penalty
                penalties["filler_phrases"] = min(raw_penalty, self.filler_phrase_max_penalty)
                details["filler_phrases"] = {
                    "count": filler_count,
                    "excess": excess,
                    "found": found_fillers[:5],
                    "threshold": self.filler_phrase_threshold,
                }

        # 8. Boxed answer hedging (e.g., "A or B", "A/B", "A, B")
        # FIX #8: Use pre-compiled patterns from module level
        if boxed_answer and self.boxed_hedging_penalty > 0:
            answer_stripped = boxed_answer.strip()
            for pattern in _HEDGING_PATTERNS:
                if pattern.search(answer_stripped):
                    penalties["boxed_hedging"] = self.boxed_hedging_penalty
                    details["boxed_hedging"] = {
                        "answer": boxed_answer,
                        "pattern_matched": pattern.pattern,
                    }
                    break

        # 9. Prompt copying (model copies question into thinking)
        # FIX #13: Defensive division by zero check
        if prompt_text and thinking_lower and self.prompt_copy_threshold > 0:
            prompt_lower = prompt_text.lower()
            prompt_words = set(w.strip(".,!?;:()[]\"'") for w in prompt_lower.split() if len(w) > 3)
            thinking_words = set(
                w.strip(".,!?;:()[]\"'") for w in thinking_lower.split() if len(w) > 3
            )

            if prompt_words:
                overlap = len(prompt_words & thinking_words)
                ratio = overlap / len(prompt_words) if len(prompt_words) > 0 else 0

                if ratio > self.prompt_copy_threshold:
                    penalties["prompt_copying"] = self.prompt_copy_penalty
                    details["prompt_copying"] = {
                        "overlap_ratio": round(ratio, 3),
                        "threshold": self.prompt_copy_threshold,
                        "overlap_count": overlap,
                        "prompt_word_count": len(prompt_words),
                    }

        # 10. Low entropy text (very repetitive/garbage)
        if thinking_lower and self.min_unique_word_ratio > 0:
            words = thinking_lower.split()
            if len(words) >= 10:  # Only check if enough words
                unique_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
                if unique_ratio < self.min_unique_word_ratio:
                    penalties["low_entropy"] = self.low_entropy_penalty
                    details["low_entropy"] = {
                        "unique_ratio": round(unique_ratio, 3),
                        "threshold": self.min_unique_word_ratio,
                        "total_words": len(words),
                        "unique_words": len(set(words)),
                    }

        # 11. Reasoning-answer inconsistency
        # FIX #8: Use pre-compiled patterns
        if boxed_answer and thinking_lower and self.inconsistency_penalty > 0:
            answer_stripped = boxed_answer.strip().upper()

            # Only check for single-letter answers
            if len(answer_stripped) == 1 and answer_stripped in "ABCD":
                # Look for contradicting conclusions in thinking
                other_options = [opt for opt in "ABCD" if opt != answer_stripped]

                for opt in other_options:
                    for pattern in _STRONG_CONCLUSION_PATTERNS:
                        match = pattern.search(thinking_lower)
                        if match:
                            # Check if this pattern mentions the other option
                            matched_text = match.group(0).upper()
                            if opt in matched_text:
                                penalties["inconsistency"] = self.inconsistency_penalty
                                details["inconsistency"] = {
                                    "boxed_answer": answer_stripped,
                                    "contradicting_option": opt,
                                    "pattern": pattern.pattern,
                                }
                                break
                    if "inconsistency" in penalties:
                        break

        # Calculate total penalty (capped at 1.0)
        total_penalty = min(1.0, sum(penalties.values()))

        return total_penalty, {"total": total_penalty, "breakdown": penalties, "details": details}

    def smooth_reward(self, reward: float) -> float:
        """
        Apply reward smoothing to avoid spikes.

        Smoothing options:
        - none: return as-is
        - clip: hard clip to [reward_min, reward_max]
        - sigmoid: smooth S-curve centered at reward_center
        - tanh: hyperbolic tangent smoothing
        """
        if self.reward_smoothing == "none":
            return reward

        if self.reward_smoothing == "clip":
            return max(self.reward_min, min(self.reward_max, reward))

        if self.reward_smoothing == "sigmoid":
            # Sigmoid centered at reward_center with temperature
            # Higher temp = sharper transition
            x = (reward - self.reward_center) * self.reward_temp
            # Use stable sigmoid computation
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            else:
                z = math.exp(x)
                return z / (1.0 + z)

        if self.reward_smoothing == "tanh":
            # Tanh smoothing: maps (-inf, inf) to (-1, 1), scaled to (0, 1)
            x = (reward - self.reward_center) * self.reward_temp
            return 0.5 * (math.tanh(x) + 1.0)

        return reward


# Default configuration - sensible defaults for technical exam data
# All features enabled with tuned thresholds to minimize false positives
# Default configuration - OPTIMIZED based on 1396-sample rollout analysis
# Gaming pattern analysis showed:
#   - answer_repetition: hits CORRECT 22.6% vs incorrect 0.7% → DISABLE (backwards!)
#   - answer_shopping: hits CORRECT 15.6% vs incorrect 10.2% → DISABLE (backwards!)
#   - boxed_hedging: CORRECT 0.2% vs incorrect 11.1% → KEEP (good targeting!)
#   - multiple_boxed: CORRECT 14.6% vs incorrect 16.7% → REDUCE (slight targeting)
#   - repetitive_phrases: CORRECT 30.9% vs incorrect 31.0% → REDUCE (neutral)
DEFAULT_FORMAT_CONFIG = FormatConfig(
    # Format tags
    think_tag="think",
    alt_think_tags=["thinking"],
    # Format gate - enabled (ensures format compliance)
    strict_format=True,
    format_gate_threshold=0.5,
    format_gate_penalty=0.3,
    # Length penalties - moderate (prevents gaming without being too strict)
    min_thinking_words=20,  # Require some reasoning
    max_thinking_words=250,  # Reduced from 300 - discourage padding
    max_answer_words=80,  # Reduced from 100
    length_penalty_weight=0.10,  # Increased from 0.08
    length_penalty_type="soft",
    # Bad words - common filler words
    bad_words=["obviously", "clearly", "simply", "basically", "essentially"],
    bad_word_penalty=0.025,
    bad_word_max_penalty=0.15,
    # Repeat words - enabled with reasonable threshold
    repeat_word_threshold=4,
    repeat_word_penalty=0.02,
    repeat_word_max_penalty=0.20,
    # =========================================================================
    # GAMING DETECTION - TUNED based on correct/incorrect targeting analysis
    # Only penalize patterns that actually target INCORRECT answers!
    # =========================================================================
    detect_gaming=True,
    # Multiple boxed - slightly targets incorrect (16.7% vs 14.6%)
    multiple_boxed_penalty=0.25,  # Reduced - only slight targeting
    # Answer shopping - DISABLED! Hits CORRECT more (15.6% vs 10.2%)
    # When model analyzes all options, it's often correct!
    answer_shopping_penalty=0.0,  # ← DISABLED
    answer_shopping_threshold=4,
    # Hedge words - keep light penalty
    hedge_word_threshold=5,
    hedge_word_penalty=0.02,
    hedge_word_max_penalty=0.12,
    # Repetitive phrases - reduced (neutral targeting 31% vs 31%)
    repetitive_phrase_threshold=4,  # Increased threshold
    repetitive_phrase_min_words=4,  # Longer phrases only
    repetitive_phrase_penalty=0.05,  # Reduced from 0.12
    repetitive_phrase_max_penalty=0.15,  # Reduced from 0.35
    # Garbage chars - keep moderate
    garbage_char_threshold=5,
    garbage_char_penalty=0.04,
    garbage_char_max_penalty=0.20,
    # Answer repetition - DISABLED! Hits CORRECT 32x more (22.6% vs 0.7%)
    # Confident correct answers naturally repeat the answer!
    answer_repeat_threshold=10,  # Very high threshold
    answer_repeat_penalty=0.0,  # ← DISABLED
    # Filler phrases - keep light (neutral targeting)
    filler_phrase_threshold=4,
    filler_phrase_penalty=0.04,
    filler_phrase_max_penalty=0.12,
    # Boxed hedging - KEEP! Great targeting (0.2% correct vs 11.1% incorrect)
    boxed_hedging_penalty=0.40,  # Keep strong - targets wrong answers!
    # Prompt copying - DISABLED for technical data
    prompt_copy_threshold=1.0,
    prompt_copy_penalty=0.0,
    # Low entropy - keep moderate
    min_unique_word_ratio=0.20,
    low_entropy_penalty=0.15,
    # Inconsistency - KEEP! Only hits incorrect (0% vs 0.1%)
    inconsistency_penalty=0.20,  # Increased - perfect targeting!
    # Reward smoothing - enabled for stable training
    reward_smoothing="sigmoid",
    reward_temp=4.0,
    reward_center=0.5,
)


# Backward compatible config (all penalties disabled)
MINIMAL_FORMAT_CONFIG = FormatConfig(
    think_tag="think",
    strict_format=False,
    detect_gaming=False,
    reward_smoothing="none",
)


# =============================================================================
# Answer Extraction
# =============================================================================


def extract_boxed_answer(
    completion: str, format_config: Optional[FormatConfig] = None
) -> Optional[str]:
    """
    Extract answer from \\boxed{} format (or configured alternative).

    Handles:
    - \\boxed{B}
    - \\boxed{BD}
    - \\boxed{B, D}
    - \\boxed{A and C}
    - \\boxed{0}  (math answers)
    - \\boxed{42}
    - \\boxed{x^{2}}  (nested braces)
    - Custom formats like \\answer{} if configured

    Args:
        completion: The completion text to extract from
        format_config: Optional format configuration (defaults to DEFAULT_FORMAT_CONFIG)

    Returns:
        Raw content string (not normalized) or None if no boxed answer found
    """
    if not isinstance(completion, str):
        _logger.debug(f"extract_boxed_answer received non-string: {type(completion)}")
        return None

    if not completion:
        _logger.debug("extract_boxed_answer: empty completion")
        return None

    if format_config is None:
        format_config = DEFAULT_FORMAT_CONFIG

    lower_completion = completion.lower()

    # Try each boxed format
    for boxed_fmt in format_config.get_all_boxed_formats():
        # Extract command name (e.g., "boxed" from "\\boxed{}")
        match = re.match(r"\\(\w+)\{\}", boxed_fmt)
        if not match:
            continue
        cmd = match.group(1).lower()

        # Find the command position
        boxed_start = lower_completion.find(f"\\{cmd}{{")

        if boxed_start == -1:
            continue

        # Find the actual start in original string
        content_start = boxed_start + len(f"\\{cmd}{{")

        # Handle nested braces by counting brace depth
        depth = 1
        pos = content_start
        while pos < len(completion) and depth > 0:
            if completion[pos] == "{":
                depth += 1
            elif completion[pos] == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            # Extract content (excluding the final closing brace)
            content = completion[content_start : pos - 1].strip()
            if content:
                _logger.debug(f"extract_boxed_answer: found '{content}' using {boxed_fmt}")
                return content

    # FIX #4: REMOVED the fallback regex that didn't handle nested braces
    # The nested-brace-aware code above is the only extraction path now

    _logger.debug("extract_boxed_answer: no boxed answer found")
    return None


def extract_answer_from_completion(completion: str, for_exam: bool = True) -> Optional[str]:
    """
    Extract answer from model completion.

    Priority order (highest first):
    1. \\boxed{X} - preferred format
    2. <answer>X</answer> tags
    3. "Answer: X" or "The answer is X"
    4. **X** bold at end
    5. Last line with just letters
    6. Standalone letters in tail

    Args:
        completion: Model's completion text
        for_exam: If True, normalize to uppercase letters; if False, keep raw

    Returns:
        Answer string (normalized for exam, raw for math)
    """
    if not isinstance(completion, str):
        _logger.debug(f"extract_answer_from_completion received non-string: {type(completion)}")
        return None

    if not completion:
        return None

    # Priority 1: \\boxed{} - THE PREFERRED FORMAT
    boxed_answer = extract_boxed_answer(completion)
    if boxed_answer:
        if for_exam:
            # Check if it looks like exam answer (letters only, A-Z)
            if re.match(r"^[A-Za-z\s,&]+$", boxed_answer.replace("and", "")):
                return normalize_answer(boxed_answer)
        return boxed_answer  # Return raw for math

    # Priority 2: <answer> tags
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, re.IGNORECASE | re.DOTALL)
    if answer_match:
        content = answer_match.group(1)
        if for_exam:
            # Only normalize if it looks like exam letters
            if re.match(r"^[A-Za-z\s,&]+$", content.replace("and", "")):
                return normalize_answer(content)
        return content

    # Priority 3: "Answer:" or "The answer is" - handles both letters and numbers
    # Try numeric answer FIRST (more specific)
    answer_match = re.search(
        r"(?:answer|the answer is)[:\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        completion,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1)

    # Then try letters (but exclude "is" itself)
    answer_match = re.search(
        r"(?:answer|the answer is)[:\s]+([A-Za-z](?:\s*[,&]?\s*(?:and\s+)?[A-Za-z])*)",
        completion,
        re.IGNORECASE,
    )
    if answer_match:
        extracted = answer_match.group(1)
        # Don't return if it's just "is" (from "answer is")
        if extracted.lower() not in ("is", "a", "an", "the"):
            return normalize_answer(extracted) if for_exam else extracted

    # Priority 4: Bold answer **B** or **B, D** at end (case-insensitive)
    bold_match = re.search(
        r"\*\*([A-Za-z](?:\s*[,&]?\s*[A-Za-z])*)\*\*\s*\.?\s*$", completion, re.IGNORECASE
    )
    if bold_match:
        return normalize_answer(bold_match.group(1)) if for_exam else bold_match.group(1)

    # Priority 5: Last line with just letter(s)
    lines = completion.strip().split("\n")
    last_line = lines[-1].strip() if lines else ""
    letter_match = re.match(
        r"^([A-Za-z](?:\s*[,&]?\s*(?:and\s+)?[A-Za-z])*)\.?$", last_line, re.IGNORECASE
    )
    if letter_match:
        return normalize_answer(letter_match.group(1)) if for_exam else letter_match.group(1)

    # Priority 5.5: "Option X" patterns
    option_match = re.match(
        r"^(?:options?\s+)?([A-Za-z](?:\s*(?:,|and|&)\s*(?:options?\s+)?[A-Za-z])*)\.?$",
        last_line,
        re.IGNORECASE,
    )
    if option_match:
        return normalize_answer(option_match.group(1)) if for_exam else option_match.group(1)

    # Priority 5.6: Full option line (fixed regex - require at least one separator between options)
    if re.match(
        r"^(?:options?\s+[A-Za-z](?:\s*(?:,|and|&)\s+options?\s+[A-Za-z])*)\.?$",
        last_line,
        re.IGNORECASE,
    ):
        return normalize_answer(last_line) if for_exam else last_line

    # Priority 6: Standalone letters in tail (case-insensitive)
    # Only match single letters that look like answer choices
    tail = completion[-100:] if len(completion) > 100 else completion

    # Common words to exclude from letter extraction
    common_words = {
        "a",
        "i",
        "is",
        "as",
        "at",
        "be",
        "by",
        "do",
        "go",
        "he",
        "if",
        "in",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "so",
        "to",
        "up",
        "us",
        "we",
        "an",
        "am",
    }

    # Find standalone single letters that aren't common words
    # Use word boundaries and ensure it's truly standalone
    letters = []
    for match in re.finditer(r"(?<![a-zA-Z])([A-Za-z])(?![a-zA-Z])", tail):
        letter = match.group(1)
        # Check context - is this letter part of a word?
        start = max(0, match.start() - 3)
        end = min(len(tail), match.end() + 3)
        context = tail[start:end].lower()
        # Skip if this looks like part of a common word
        skip = False
        for word in common_words:
            if word in context and letter.lower() in word:
                skip = True
                break
        if not skip:
            letters.append(letter)

    if letters:
        # Only take the last letter to avoid combining unrelated mentions
        result = letters[-1]
        return normalize_answer(result) if for_exam else result

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer to uppercase letters only, sorted.

    For exam-style answers (letters A-Z). Does NOT modify numeric answers.

    FIX #5: Now only extracts standalone single letters, not all letters from words.

    Examples:
        "B" -> "B"
        "b" -> "B"
        "A, C" -> "AC"
        "B and D" -> "BD"
        "Option B" -> "B"
        "42" -> "42" (preserved as-is if numeric)

    Args:
        answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    if not answer:
        return ""

    if not isinstance(answer, str):
        return str(answer)

    # Check if this looks like a numeric answer - if so, return as-is
    stripped = answer.strip()
    if re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", stripped):
        return stripped

    # Check if it's a fraction
    if re.match(r"^[-+]?\d+/\d+$", stripped):
        return stripped

    # FIX #5: Only extract standalone single letters that look like answer choices
    # First, remove connector words and common prefixes
    cleaned = re.sub(r"\b(and|or)\b", " ", answer, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(options?|answer|choice|choices|the|is|are)\s*", " ", cleaned, flags=re.IGNORECASE
    )

    # Now extract only standalone single uppercase letters (A-Z, likely answer choices)
    # This regex matches single letters surrounded by word boundaries or punctuation
    letters = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", cleaned)

    # Filter to only keep letters that are likely answer choices (A-F typically)
    # and deduplicate while preserving order for sorting
    valid_letters = []
    seen = set()
    for letter in letters:
        upper_letter = letter.upper()
        # Only accept A-F as valid answer choices (most exams use A-D, some use A-F)
        if upper_letter in "ABCDEF" and upper_letter not in seen:
            valid_letters.append(upper_letter)
            seen.add(upper_letter)

    # Sort alphabetically
    valid_letters.sort()

    return "".join(valid_letters) if valid_letters else stripped


def normalize_ground_truth(ground_truth: Any) -> str:
    """
    Normalize ground truth answer.

    Handles:
        "D" -> "D"
        "AD" -> "AD"
        "A,D" -> "AD"
        "A and D" -> "AD"
        ["A", "D"] -> "AD"
        [1, 2] -> "12" (converts to strings first)
        0 -> "0" (handles falsy values)

    Args:
        ground_truth: The ground truth answer (str, list, or other)

    Returns:
        Normalized ground truth string
    """
    if isinstance(ground_truth, list):
        # Convert all elements to strings first, handle None/empty
        str_elements = []
        for g in ground_truth:
            if g is not None:
                s = str(g).strip()
                if s:
                    str_elements.append(s.upper())
        return "".join(sorted(set(str_elements)))

    # Handle falsy values explicitly (0, "", etc.)
    if ground_truth is None:
        return ""

    if isinstance(ground_truth, (int, float)):
        return str(ground_truth)

    if not isinstance(ground_truth, str):
        return str(ground_truth)

    if not ground_truth:
        return ""

    return normalize_answer(str(ground_truth))


# =============================================================================
# Boxed Format Detection
# =============================================================================


def check_boxed_format(
    completion: str, format_config: Optional[FormatConfig] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if completion uses \\boxed{} format (or configured alternative) correctly.

    Rewards:
    - 1.0: Has boxed format with valid content (letters, numbers, expressions)
    - 0.5: Has boxed format but empty
    - 0.0: No boxed format found

    Args:
        completion: The completion text to check
        format_config: Optional format configuration (defaults to DEFAULT_FORMAT_CONFIG)

    Returns:
        (score, details) tuple
    """
    if not isinstance(completion, str) or not completion:
        return 0.0, {"error": "empty_or_invalid_completion", "has_boxed": False}

    if format_config is None:
        format_config = DEFAULT_FORMAT_CONFIG

    lower_completion = completion.lower()

    # Check each accepted boxed format
    for boxed_fmt in format_config.get_all_boxed_formats():
        match = re.match(r"\\(\w+)\{\}", boxed_fmt)
        if not match:
            continue
        cmd = match.group(1)

        # Find the boxed pattern
        boxed_match = re.search(rf"\\{cmd}\{{([^}}]*)\}}", completion, re.IGNORECASE)

        if boxed_match:
            content = boxed_match.group(1).strip()

            # Check if content is non-empty and valid
            if content:
                # Check for exam letters (A-Z)
                letters = re.findall(r"[A-Za-z]", content)
                # Check for numeric/math content
                has_numeric = bool(re.search(r"[\d\.\-\+]", content))

                if letters or has_numeric or len(content) > 0:
                    return 1.0, {
                        "has_boxed": True,
                        "boxed_content": content,
                        "boxed_format": boxed_fmt,
                        "has_letters": bool(letters),
                        "has_numeric": has_numeric,
                        "valid": True,
                    }

            # Has boxed but content is empty
            return 0.5, {
                "has_boxed": True,
                "boxed_content": content,
                "boxed_format": boxed_fmt,
                "valid": False,
                "reason": "empty_content",
            }

    # No boxed format found
    return 0.0, {
        "has_boxed": False,
        "reason": "no_boxed_found",
        "expected_formats": format_config.get_all_boxed_formats(),
    }


# =============================================================================
# Format Checking
# =============================================================================


def check_format_compliance(
    completion: str, format_config: Optional[FormatConfig] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if completion follows expected format structure.

    Checks for:
    - Thinking tags: <think>...</think> or configured alternatives
    - Answer tags: <answer>...</answer> if configured
    - Proper ordering: thinking before boxed answer

    Args:
        completion: The completion text to check
        format_config: Optional format configuration (defaults to DEFAULT_FORMAT_CONFIG)

    Returns:
        (score, details) tuple
    """
    if not isinstance(completion, str) or not completion:
        return 0.0, {"error": "empty_or_invalid_completion"}

    if format_config is None:
        format_config = DEFAULT_FORMAT_CONFIG

    lower_completion = completion.lower()

    # Check for thinking tags
    think_valid = False
    think_tag_found = None

    for tag in format_config.get_all_think_tags():
        open_tag = f"<{tag.lower()}>"
        close_tag = f"</{tag.lower()}>"

        if open_tag in lower_completion and close_tag in lower_completion:
            think_valid = True
            think_tag_found = tag
            break

    # Check for answer tags (if configured)
    answer_valid = True  # Default to True if not required
    answer_tag_found = None

    if format_config.answer_tag:
        open_tag = f"<{format_config.answer_tag.lower()}>"
        close_tag = f"</{format_config.answer_tag.lower()}>"

        answer_valid = open_tag in lower_completion and close_tag in lower_completion
        if answer_valid:
            answer_tag_found = format_config.answer_tag

    # Check proper ordering: thinking should end before boxed starts
    proper_order = True
    if think_valid and think_tag_found:
        close_tag = f"</{think_tag_found.lower()}>"
        think_end_pos = lower_completion.rfind(close_tag)

        # Check against all boxed formats
        for boxed_fmt in format_config.get_all_boxed_formats():
            match = re.match(r"\\(\w+)\{\}", boxed_fmt)
            if match:
                cmd = match.group(1).lower()
                boxed_pos = lower_completion.find(f"\\{cmd}{{")
                if boxed_pos != -1 and think_end_pos > boxed_pos:
                    proper_order = False
                    break

    # Calculate score
    score = 0.0

    # Thinking tags: 0.5 points
    if think_valid:
        score += 0.5

    # Proper order: 0.3 points (only if thinking is valid)
    if proper_order and think_valid:
        score += 0.3

    # Answer tags: 0.2 points (only if configured and present)
    if format_config.answer_tag:
        if answer_valid:
            score += 0.2
    else:
        # If answer tag not required, give the 0.2 points for having valid think
        if think_valid:
            score += 0.2

    return score, {
        "has_think_tags": think_valid,  # FIX #11: Consistent key name
        "think_tag_found": think_tag_found,
        "expected_think_tags": format_config.get_all_think_tags(),
        "has_answer_tag": answer_valid if format_config.answer_tag else None,
        "answer_tag_found": answer_tag_found,
        "expected_answer_tag": format_config.answer_tag,
        "proper_order": proper_order,
        "score": score,
    }


def check_reasoning_quality(
    completion: str, format_config: Optional[FormatConfig] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if thinking section has substantive content.

    Uses pre-compiled patterns for maximum efficiency.

    Args:
        completion: The completion text to check
        format_config: Optional format configuration (defaults to DEFAULT_FORMAT_CONFIG)

    Returns:
        (score, details) tuple with word counts for length penalty calculation
    """
    if not isinstance(completion, str) or not completion:
        return 0.0, {"error": "empty_or_invalid_completion", "thinking_words": 0, "answer_words": 0}

    if format_config is None:
        format_config = DEFAULT_FORMAT_CONFIG

    # Use pre-compiled patterns if available, otherwise fall back
    all_think_matches = []
    tag_used = None

    if format_config._think_patterns:
        # Use pre-compiled patterns (fast path)
        for tag, pattern in format_config._think_patterns:
            matches = pattern.findall(completion)
            if matches:
                for m in matches:
                    all_think_matches.append((m, tag))
    else:
        # Fall back to dynamic regex (slow path)
        for tag in format_config.get_all_think_tags():
            pattern = rf"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, completion, re.IGNORECASE | re.DOTALL)
            if matches:
                for m in matches:
                    all_think_matches.append((m, tag))

    # Calculate answer words (text after boxed)
    answer_words = 0
    boxed_end_pos = -1

    # Find end of boxed content
    lower_completion = completion.lower()
    for fmt in format_config.get_all_boxed_formats():
        match = re.match(r"\\(\w+)\{\}", fmt)
        if match:
            cmd = match.group(1).lower()
            boxed_match = re.search(rf"\\{cmd}\{{[^}}]*\}}", lower_completion)
            if boxed_match:
                boxed_end_pos = boxed_match.end()
                break

    if boxed_end_pos > 0:
        answer_text = completion[boxed_end_pos:].strip()
        answer_words = len(answer_text.split()) if answer_text else 0

    if not all_think_matches:
        return 0.3, {
            "has_thinking": False,
            "reason": "no_think_tags",
            "thinking_words": 0,
            "answer_words": answer_words,
        }

    # Use the longest thinking section
    thinking, tag_used = max(all_think_matches, key=lambda x: len(x[0]))
    thinking = thinking.strip()

    # Check if thinking is actually empty
    if not thinking:
        return 0.3, {
            "has_thinking": False,
            "reason": "empty_think_content",
            "thinking_words": 0,  # FIX #11: Removed redundant "word_count" key
            "answer_words": answer_words,
            "tag_used": tag_used,
        }

    # Quality indicators - use efficient word counting
    words = thinking.split()
    word_count = len(words)

    # Check for analysis keywords (case-insensitive, efficient)
    thinking_lower = thinking.lower()
    has_analysis = any(
        kw in thinking_lower
        for kw in (
            "because",
            "since",
            "therefore",
            "thus",
            "means",
            "implies",
            "incorrect",
            "correct",
        )
    )

    has_options = bool(
        re.search(r"(option [a-z]|choice [a-z]|\b[a-z]\)|\b[a-z]\.)", thinking_lower)
    )

    # Score calculation
    score = 0.0

    # Length score (0-0.4)
    if word_count >= 50:
        score += 0.4
    elif word_count >= 30:
        score += 0.3
    elif word_count >= 15:
        score += 0.2
    elif word_count >= 5:
        score += 0.1

    # Analysis score (0-0.3)
    if has_analysis:
        score += 0.3

    # Options mentioned (0-0.3)
    if has_options:
        score += 0.3

    return min(score, 1.0), {
        "has_thinking": True,
        "thinking_words": word_count,  # FIX #11: Only use "thinking_words", removed "word_count"
        "answer_words": answer_words,
        "has_analysis": has_analysis,
        "has_options": has_options,
        "tag_used": tag_used,
        "score": min(score, 1.0),
    }


# =============================================================================
# Accuracy Reward with possible_boxed_answers Support
# =============================================================================


def _parse_numeric(value: str) -> Optional[float]:
    """
    Parse a string as a numeric value, handling various formats.

    Handles:
    - Integers: "42"
    - Floats: "3.14", ".5", "-.5"
    - Scientific notation: "1.5e-3"
    - Fractions: "1/2"
    - Percentages: "50%" -> 0.5 (FIX #12)

    Args:
        value: String to parse

    Returns:
        Float value or None if not parseable
    """
    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    # FIX #12: Handle percentages
    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100
        except ValueError:
            pass

    # Try direct float conversion
    try:
        return float(value)
    except ValueError:
        pass

    # Try fraction
    try:
        if "/" in value:
            return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        pass

    return None


def check_answer_match(
    model_answer: str, possible_answers: List[Any]
) -> Tuple[bool, Optional[str]]:
    """
    Check if model_answer matches any of the possible answers.

    Performs multiple matching strategies:
    1. Exact match (case-insensitive)
    2. Normalized match (for exam letters)
    3. Numeric match (for math answers, with reasonable tolerance)

    Args:
        model_answer: The model's extracted answer
        possible_answers: List of acceptable answers

    Returns:
        (matched, matched_answer) - True and the matched answer, or False and None
    """
    if not model_answer or not possible_answers:
        _logger.debug(
            f"check_answer_match: empty input model='{model_answer}', possible={possible_answers}"
        )
        return False, None

    model_clean = model_answer.strip()
    model_lower = model_clean.lower()
    model_normalized = normalize_answer(model_clean)
    _logger.debug(f"check_answer_match: model='{model_clean}' normalized='{model_normalized}'")

    for answer in possible_answers:
        # Handle None and falsy values correctly (0 is valid!)
        if answer is None:
            continue

        answer_clean = str(answer).strip()
        answer_lower = answer_clean.lower()
        answer_normalized = normalize_answer(answer_clean)

        # Strategy 1: Exact match (case-insensitive)
        if model_lower == answer_lower:
            _logger.debug(f"check_answer_match: MATCH (exact) '{model_clean}' == '{answer_clean}'")
            return True, answer_clean

        # Strategy 2: Normalized match (for exam letters like "BC" == "CB")
        if model_normalized and answer_normalized:
            if model_normalized == answer_normalized:
                _logger.debug(
                    f"check_answer_match: MATCH (normalized) '{model_normalized}' == '{answer_normalized}'"
                )
                return True, answer_clean

        # Strategy 3: Numeric/value match with reasonable tolerance
        model_num = _parse_numeric(model_clean)
        answer_num = _parse_numeric(answer_clean)

        if model_num is not None and answer_num is not None:
            # Use relative tolerance for larger numbers, absolute for small
            if answer_num == 0:
                if abs(model_num) < 1e-6:
                    _logger.debug(
                        f"check_answer_match: MATCH (numeric zero) {model_num} ≈ {answer_num}"
                    )
                    return True, answer_clean
            else:
                relative_diff = abs(model_num - answer_num) / abs(answer_num)
                if relative_diff < 1e-6:
                    _logger.debug(f"check_answer_match: MATCH (numeric) {model_num} ≈ {answer_num}")
                    return True, answer_clean

    _logger.debug(f"check_answer_match: NO MATCH for '{model_clean}' against {possible_answers}")
    return False, None


def compute_accuracy_reward(
    completion: str,
    ground_truth: Any = None,
    possible_boxed_answers: Optional[List[Any]] = None,
    partial_credit: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute accuracy reward by comparing extracted answer to ground truth.

    Supports possible_boxed_answers for flexible matching.

    FIX #6: Partial credit now uses F1 score instead of Jaccard to penalize
    excessive guessing (model can't just guess all letters).

    Args:
        completion: Model's full completion text
        ground_truth: Correct answer option(s) (fallback)
        possible_boxed_answers: List of all valid answers (preferred)
            - If None: use ground_truth only
            - If []: skip accuracy (no reward signal)
            - If [...]: match against any in list
        partial_credit: Give partial credit for multi-answer questions

    Returns:
        (score, details) tuple
    """
    # Always try to extract the model's answer for debugging
    # Determine exam style first for proper extraction
    is_exam_style = False
    if possible_boxed_answers:
        for ans in possible_boxed_answers:
            if ans is not None and re.match(r"^[A-Za-z\s,&]+$", str(ans).replace("and", "")):
                is_exam_style = True
                break
    elif ground_truth is not None:
        gt_str = str(ground_truth)
        if re.match(r"^[A-Za-z\s,&]+$", gt_str.replace("and", "")):
            is_exam_style = True

    # Extract answer from completion
    model_answer = extract_answer_from_completion(completion, for_exam=is_exam_style)
    if model_answer is None:
        model_answer = extract_boxed_answer(completion)

    # Handle empty possible_boxed_answers - skip accuracy reward but include extracted answer
    if possible_boxed_answers is not None and len(possible_boxed_answers) == 0:
        return 0.5, {
            "skipped": True,
            "reason": "empty_possible_boxed_answers",
            "model_answer": model_answer,  # Still report what was extracted
            "correct": None,  # Unknown - no ground truth to compare
        }

    if model_answer is None:
        return 0.0, {
            "error": "no_answer_extracted",
            "model_answer": None,
            "correct_answer": ground_truth,
            "possible_answers": possible_boxed_answers,
            "correct": False,
        }

    # Match against possible_boxed_answers if provided
    if possible_boxed_answers:
        matched, matched_answer = check_answer_match(model_answer, possible_boxed_answers)
        _logger.debug(
            f"Answer match: {matched}, matched_answer: {matched_answer}, from possible: {possible_boxed_answers}"
        )

        if matched:
            return 1.0, {
                "model_answer": model_answer,
                "matched_answer": matched_answer,
                "possible_answers": possible_boxed_answers,
                "correct": True,
                "match_type": "possible_boxed_answers",
            }
        else:
            # Wrong answer - check for partial credit on multi-answer
            if partial_credit:
                best_partial = 0.0
                model_normalized = normalize_answer(model_answer)

                # === GUARD: Only give letter-based partial credit for MULTI-ANSWER questions ===
                # If ground truth is a single letter (e.g., "A"), model is either right or wrong.
                # Answering "ACF" for gt="A" is WRONG, not partially right.
                # Partial credit only makes sense when gt itself has multiple letters (e.g., "BD").
                gt_normalized = (
                    normalize_answer(str(ground_truth)) if ground_truth is not None else ""
                )
                is_multi_answer_question = len(gt_normalized) > 1

                # Only apply letter-based partial credit if model answer has letters
                # AND this is a multi-answer question
                if model_normalized and is_multi_answer_question:
                    model_set = set(model_normalized)

                    for ans in possible_boxed_answers:
                        if ans is None:
                            continue
                        ans_normalized = normalize_answer(str(ans))
                        if ans_normalized and len(ans_normalized) > 1:
                            ans_set = set(ans_normalized)

                            # FIX #6: Use F1 score instead of Jaccard
                            # This penalizes guessing too many answers
                            intersection = len(model_set & ans_set)
                            if intersection > 0:
                                precision = intersection / len(model_set) if model_set else 0
                                recall = intersection / len(ans_set) if ans_set else 0
                                if precision + recall > 0:
                                    f1 = 2 * precision * recall / (precision + recall)
                                    best_partial = max(best_partial, f1)

                # Also check numeric partial credit
                model_num = _parse_numeric(model_answer)
                if model_num is not None:
                    for ans in possible_boxed_answers:
                        if ans is None:
                            continue
                        ans_num = _parse_numeric(str(ans))
                        if ans_num is not None and ans_num != 0:
                            # Give partial credit based on relative closeness
                            relative_error = abs(model_num - ans_num) / abs(ans_num)
                            if relative_error < 0.5:  # Within 50%
                                partial = max(0, 1.0 - relative_error)
                                best_partial = max(best_partial, partial)

                if best_partial > 0:
                    return best_partial, {
                        "model_answer": model_answer,
                        "possible_answers": possible_boxed_answers,
                        "correct": False,
                        "match_type": "partial",
                        "partial_score": best_partial,
                    }

            return 0.0, {
                "model_answer": model_answer,
                "possible_answers": possible_boxed_answers,
                "correct": False,
                "match_type": "wrong",
            }

    # Fallback to ground_truth comparison
    # Use 'is None' to properly handle 0 and other falsy values
    if ground_truth is None:
        return 0.5, {
            "model_answer": model_answer,
            "correct_answer": None,
            "correct": None,
            "reason": "no_ground_truth",
        }

    correct_answer = (
        normalize_ground_truth(ground_truth) if is_exam_style else str(ground_truth).strip()
    )
    model_normalized = normalize_answer(model_answer) if is_exam_style else model_answer.strip()

    # Exact match
    if (
        model_normalized == correct_answer
        or model_answer.strip().lower() == str(ground_truth).strip().lower()
    ):
        return 1.0, {
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "correct": True,
            "match_type": "exact",
        }

    # Partial match for multi-answer questions ONLY (exam style)
    # Single-answer questions (gt="A") get no partial credit — model is right or wrong
    if is_exam_style and partial_credit:
        model_set = set(model_normalized) if model_normalized else set()
        correct_set = set(correct_answer) if correct_answer else set()

        if model_set and correct_set and len(correct_set) > 1:
            intersection = len(model_set & correct_set)
            if intersection > 0:
                # FIX #6: Use F1 score instead of Jaccard
                precision = intersection / len(model_set) if model_set else 0
                recall = intersection / len(correct_set) if correct_set else 0
                if precision + recall > 0:
                    partial_score = 2 * precision * recall / (precision + recall)
                    return partial_score, {
                        "model_answer": model_answer,
                        "correct_answer": correct_answer,
                        "correct": False,
                        "match_type": "partial",
                        "partial_score": partial_score,
                        "matched": list(model_set & correct_set),
                        "missed": list(correct_set - model_set),
                        "extra": list(model_set - correct_set),
                    }

    # Wrong answer
    return 0.0, {
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "correct": False,
        "match_type": "wrong",
    }


# =============================================================================
# Composite Reward
# =============================================================================


def compute_reward(
    completion: str,
    ground_truth: Any = None,
    weights: Optional[RewardWeights] = None,
    possible_boxed_answers: Optional[List[Any]] = None,
    format_config: Optional[FormatConfig] = None,
    prompt_text: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute composite reward for GRPO training.

    Components (default weights - FIX #14: Updated to match actual defaults):
    - accuracy (75%): Did model get the right answer?
    - boxed (2%): Did model use \\boxed{} format?
    - format (3%): Did model use <think>...</think>?
    - reasoning (20%): Quality of thinking content

    Format Gate (when strict_format=True, the default):
    - If format score < threshold, accuracy is penalized
    - This ensures model learns correct format, not just correct answers

    Penalties (when configured):
    - Length: Penalizes too-short/too-long thinking or verbose answers
    - Bad Words: Penalizes use of specified words (more occurrences = more penalty)
    - Repeat Words: Penalizes excessive repetition of words
    - Gaming: Penalizes exploitation patterns (hedging, filler, copying, etc.)

    Reward Smoothing (when configured):
    - Applies smoothing function to avoid reward spikes
    - Options: clip, sigmoid, tanh

    Args:
        completion: Model's full completion text
        ground_truth: Correct answer option(s) (fallback)
        weights: Optional custom weights (defaults to RewardWeights())
        possible_boxed_answers: List of all valid answers (preferred)
            - If None: use ground_truth only
            - If []: skip accuracy component
            - If [...]: match against any in list
        format_config: Optional format configuration (defaults to DEFAULT_FORMAT_CONFIG)
        prompt_text: Original prompt text (for copy detection in gaming)

    Returns:
        (total_score, detailed_breakdown) tuple
    """
    _logger.debug(f"compute_reward called: gt={ground_truth}, pba={possible_boxed_answers}")
    _logger.debug(f"completion preview: {completion[:200] if completion else 'None'}...")

    if weights is None:
        weights = RewardWeights()

    if format_config is None:
        format_config = DEFAULT_FORMAT_CONFIG

    _logger.debug(
        f"weights: acc={weights.accuracy}, box={weights.boxed}, fmt={weights.format}, reas={weights.reasoning}"
    )
    _logger.debug(
        f"format_config: strict={format_config.strict_format}, gaming={format_config.detect_gaming}"
    )

    # Compute individual rewards
    accuracy_score, accuracy_details = compute_accuracy_reward(
        completion, ground_truth, possible_boxed_answers
    )
    _logger.debug(f"accuracy_score={accuracy_score:.4f}, details={accuracy_details}")

    boxed_score, boxed_details = check_boxed_format(completion, format_config)
    _logger.debug(f"boxed_score={boxed_score:.4f}")

    format_score, format_details = check_format_compliance(completion, format_config)
    _logger.debug(f"format_score={format_score:.4f}")

    reasoning_score, reasoning_details = check_reasoning_quality(completion, format_config)
    _logger.debug(
        f"reasoning_score={reasoning_score:.4f}, words={reasoning_details.get('thinking_words', 0)}"
    )

    # Combined format score for gate check (average of boxed + format + reasoning)
    combined_format_score = (boxed_score + format_score + reasoning_score) / 3.0

    # Apply format gate if strict mode is enabled
    gated_accuracy_score = accuracy_score
    format_gate_applied = False

    if format_config.strict_format:
        if combined_format_score < format_config.format_gate_threshold:
            # Format failed gate - penalize accuracy
            gated_accuracy_score = accuracy_score * format_config.format_gate_penalty
            format_gate_applied = True
            _logger.debug(
                f"FORMAT GATE APPLIED: combined={combined_format_score:.4f} < threshold={format_config.format_gate_threshold}, gated_acc={gated_accuracy_score:.4f}"
            )

    # === Length penalty calculation ===
    thinking_words = reasoning_details.get("thinking_words", 0)
    answer_words = reasoning_details.get("answer_words", 0)
    length_penalty = format_config.compute_length_penalty(thinking_words, answer_words)
    if length_penalty > 0:
        _logger.debug(
            f"length_penalty={length_penalty:.4f} (thinking={thinking_words}, answer={answer_words})"
        )

    # === Bad words penalty calculation ===
    bad_words_penalty, bad_words_counts = format_config.compute_bad_words_penalty(completion)
    if bad_words_penalty > 0:
        _logger.debug(f"bad_words_penalty={bad_words_penalty:.4f}, counts={bad_words_counts}")

    # === Repeat words penalty calculation ===
    repeat_words_penalty, repeat_words_counts = format_config.compute_repeat_words_penalty(
        completion
    )

    # === Gaming detection penalty calculation ===
    # Extract thinking text and boxed answer for gaming detection
    thinking_text = ""
    for tag in format_config.get_all_think_tags():
        match = re.search(rf"<{tag}>(.*?)</{tag}>", completion, re.IGNORECASE | re.DOTALL)
        if match:
            thinking_text = match.group(1)
            break

    # Use RAW boxed answer for hedging detection (not normalized)
    raw_boxed_answer = extract_boxed_answer(completion)
    gaming_penalty, gaming_details = format_config.compute_gaming_penalties(
        completion, thinking_text, raw_boxed_answer, prompt_text
    )

    # === Apply all penalties to the TOTAL reward ===
    # Previously, penalties only hit accuracy — so when accuracy=0 (wrong answer),
    # penalties had ZERO effect. A 14x \boxed{C} spam got the same reward as a
    # clean wrong answer. Now penalties reduce the entire reward signal.
    total_penalty = length_penalty + bad_words_penalty + repeat_words_penalty + gaming_penalty
    penalty_multiplier = max(0.0, 1.0 - total_penalty)

    # === Weighted sum (normalized by total weight) ===
    # FIX #1: Removed the no-op "/ total_weight * total_weight"
    total_weight = weights.total()
    raw_total = (
        weights.accuracy * gated_accuracy_score
        + weights.boxed * boxed_score
        + weights.format * format_score
        + weights.reasoning * reasoning_score
    ) / total_weight  # Now correctly normalizes to [0, 1]

    # Apply penalty to TOTAL reward (not just accuracy)
    penalized_total = raw_total * penalty_multiplier

    # Clamp to [0, 1] range before smoothing
    clamped_total = max(0.0, min(1.0, penalized_total))

    # === Apply reward smoothing ===
    final_total = format_config.smooth_reward(clamped_total)

    # Log final result
    _logger.info(
        f"REWARD: {final_total:.4f} (raw={raw_total:.4f}, penalized={penalized_total:.4f}, clamped={clamped_total:.4f})"
    )
    _logger.debug(
        f"  components: acc={gated_accuracy_score:.4f}*{weights.accuracy}, box={boxed_score:.4f}*{weights.boxed}, fmt={format_score:.4f}*{weights.format}, reas={reasoning_score:.4f}*{weights.reasoning}"
    )
    _logger.debug(
        f"  penalties: total={total_penalty:.4f}, multiplier={penalty_multiplier:.4f} (len={length_penalty:.4f}, bad={bad_words_penalty:.4f}, rep={repeat_words_penalty:.4f}, game={gaming_penalty:.4f})"
    )
    if accuracy_details.get("correct"):
        _logger.info(
            f"  CORRECT: model='{accuracy_details.get('model_answer')}' matches gt='{ground_truth}'"
        )
    else:
        _logger.debug(
            f"  INCORRECT: model='{accuracy_details.get('model_answer')}' != gt='{ground_truth}'"
        )

    # Build details dict
    details = {
        "total_score": final_total,
        "raw_score": raw_total,  # Pre-penalty weighted sum
        "penalized_score": penalized_total,  # After penalty multiplier
        "clamped_score": clamped_total,
        "format_gate": {
            "strict_mode": format_config.strict_format,
            "combined_format_score": combined_format_score,
            "threshold": format_config.format_gate_threshold,
            "gate_applied": format_gate_applied,
            "pre_gate_accuracy": accuracy_score,
            "post_gate_accuracy": gated_accuracy_score,
        },
        "accuracy": {
            "score": accuracy_score,
            "original_score": accuracy_score,
            "weighted": weights.accuracy * gated_accuracy_score,
            "details": accuracy_details,
        },
        "boxed": {
            "score": boxed_score,
            "weighted": weights.boxed * boxed_score,
            "details": boxed_details,
        },
        "format": {
            "score": format_score,
            "weighted": weights.format * format_score,
            "details": format_details,
        },
        "reasoning": {
            "score": reasoning_score,
            "weighted": weights.reasoning * reasoning_score,
            "details": reasoning_details,
        },
        "penalties": {
            "total": total_penalty,
            "multiplier": penalty_multiplier,
            "length": length_penalty,
            "bad_words": bad_words_penalty,
            "bad_words_counts": bad_words_counts,
            "repeat_words": repeat_words_penalty,
            "repeat_words_counts": repeat_words_counts,
        },
        "gaming_penalty": gaming_details,
        "smoothing": {
            "method": format_config.reward_smoothing,
            "pre_smooth": clamped_total,
            "post_smooth": final_total,
        },
    }

    # Log to structured logger
    _structured_logger.log_reward(
        completion=completion,
        ground_truth=ground_truth,
        possible_answers=possible_boxed_answers,
        prompt_text=prompt_text,
        details=details,
        final_reward=final_total,
        weights=weights,
    )

    return final_total, details


# =============================================================================
# Batch Processing Utilities
# =============================================================================


def compute_batch_rewards(
    completions: List[str],
    ground_truths: List[Any],
    weights: Optional[RewardWeights] = None,
    possible_boxed_answers_list: Optional[List[Optional[List[Any]]]] = None,
    format_config: Optional[FormatConfig] = None,
    prompt_texts: Optional[List[str]] = None,
) -> Tuple[List[float], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute rewards for a batch of completions.

    Args:
        completions: List of completion texts
        ground_truths: List of ground truth answers
        weights: Optional custom weights
        possible_boxed_answers_list: Optional list of possible_boxed_answers per sample
        format_config: Optional format configuration
        prompt_texts: Optional list of prompt texts

    Returns:
        (scores, details_list, summary) tuple
    """
    if len(completions) != len(ground_truths):
        raise ValueError(
            f"Completions ({len(completions)}) and ground_truths ({len(ground_truths)}) must have same length"
        )

    scores = []
    details_list = []
    correct_count = 0

    for i, (completion, gt) in enumerate(zip(completions, ground_truths)):
        pba = None
        if possible_boxed_answers_list is not None and i < len(possible_boxed_answers_list):
            pba = possible_boxed_answers_list[i]

        prompt = ""
        if prompt_texts is not None and i < len(prompt_texts):
            prompt = prompt_texts[i]

        score, details = compute_reward(
            completion=completion,
            ground_truth=gt,
            weights=weights,
            possible_boxed_answers=pba,
            format_config=format_config,
            prompt_text=prompt,
        )

        scores.append(score)
        details_list.append(details)

        if details.get("accuracy", {}).get("details", {}).get("correct"):
            correct_count += 1

    # Summary statistics
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def std(lst):
        if len(lst) < 2:
            return 0.0
        m = mean(lst)
        return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

    accuracy_scores = [d["accuracy"]["score"] for d in details_list]
    boxed_scores = [d["boxed"]["score"] for d in details_list]
    format_scores = [d["format"]["score"] for d in details_list]
    reasoning_scores = [d["reasoning"]["score"] for d in details_list]

    n = len(completions)
    n_valid = sum(1 for d in details_list if not d["accuracy"]["details"].get("skipped", False))
    no_answer_count = sum(
        1 for d in details_list if d["accuracy"]["details"].get("error") == "no_answer_extracted"
    )
    skipped_count = sum(1 for d in details_list if d["accuracy"]["details"].get("skipped", False))
    boxed_count = sum(1 for d in details_list if d["boxed"]["details"].get("has_boxed", False))

    summary = {
        "n_samples": n,
        "n_valid": n_valid,
        "correct_count": correct_count,
        # Overall
        "reward_mean": mean(scores),
        "reward_std": std(scores),
        # Accuracy
        "accuracy_reward_mean": mean(accuracy_scores),
        "accuracy_reward_std": std(accuracy_scores),
        "accuracy_rate": correct_count / n_valid if n_valid > 0 else 0.0,
        # Boxed format
        "boxed_reward_mean": mean(boxed_scores),
        "boxed_reward_std": std(boxed_scores),
        "boxed_usage_rate": boxed_count / n if n > 0 else 0.0,
        # Format
        "format_reward_mean": mean(format_scores),
        "format_reward_std": std(format_scores),
        # Reasoning
        "reasoning_reward_mean": mean(reasoning_scores),
        "reasoning_reward_std": std(reasoning_scores),
        # Issues
        "no_answer_rate": no_answer_count / n if n > 0 else 0.0,
        "skipped_accuracy_rate": skipped_count / n if n > 0 else 0.0,
    }

    # Log batch summary
    _structured_logger.log_batch_summary(scores, correct_count, n)

    return scores, details_list, summary


# =============================================================================
# GRPO-Compatible Reward Function Wrappers
# =============================================================================
# These functions follow the GRPO reward function signature:
#   (prompts: List[str], completions: List[str], answers: List[str],
#    types: Optional[List[str]]) -> List[float]
#
# Register with the reward registry for CLI usage:
#   mlx-grpo --reward-functions exam_combined_reward_func ...


def exam_accuracy_reward_func(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    GRPO-compatible exam accuracy reward function.

    Returns accuracy scores only (0.0 or 1.0 for exact match,
    partial credit for multi-answer questions).

    Args:
        prompts: List of input prompts (unused, for API compatibility)
        completions: List of model completion texts
        answers: List of ground truth answers
        types: Optional list of question types (unused)

    Returns:
        List of accuracy scores in [0.0, 1.0]
    """
    scores = []
    for completion, answer in zip(completions, answers):
        acc_score, _ = compute_accuracy_reward(
            completion=completion, ground_truth=answer, possible_boxed_answers=None
        )
        scores.append(acc_score)
    return scores


def exam_format_reward_func(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    GRPO-compatible exam format reward function.

    Returns format compliance scores based on:
    - Use of \\boxed{} format (33%)
    - Use of <think>...</think> tags (33%)
    - Reasoning quality (33%)

    Args:
        prompts: List of input prompts (unused, for API compatibility)
        completions: List of model completion texts
        answers: List of ground truth answers (unused)
        types: Optional list of question types (unused)

    Returns:
        List of format scores in [0.0, 1.0]
    """
    scores = []
    format_config = DEFAULT_FORMAT_CONFIG

    for completion in completions:
        boxed_score, _ = check_boxed_format(completion, format_config)
        format_score, _ = check_format_compliance(completion, format_config)
        reasoning_score, _ = check_reasoning_quality(completion, format_config)

        # Average of all format components
        combined = (boxed_score + format_score + reasoning_score) / 3.0
        scores.append(combined)

    return scores


def exam_reasoning_reward_func(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    GRPO-compatible exam reasoning quality reward function.

    Returns reasoning quality scores based on:
    - Presence and quality of thinking content
    - Word count in thinking section
    - Absence of filler patterns

    Args:
        prompts: List of input prompts (unused, for API compatibility)
        completions: List of model completion texts
        answers: List of ground truth answers (unused)
        types: Optional list of question types (unused)

    Returns:
        List of reasoning scores in [0.0, 1.0]
    """
    scores = []
    format_config = DEFAULT_FORMAT_CONFIG

    for completion in completions:
        reasoning_score, _ = check_reasoning_quality(completion, format_config)
        scores.append(reasoning_score)

    return scores


def exam_combined_reward_func(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    GRPO-compatible exam combined reward function.

    Returns full composite reward including:
    - Accuracy (75%): Did model get the right answer?
    - Boxed format (2%): Did model use \\boxed{}?
    - Format (3%): Did model use <think>...</think>?
    - Reasoning (20%): Quality of thinking content

    Also applies:
    - Format gate (penalizes accuracy if format fails)
    - Gaming detection penalties
    - Length and repetition penalties

    Args:
        prompts: List of input prompts (used for copy detection)
        completions: List of model completion texts
        answers: List of ground truth answers
        types: Optional list of question types (unused)

    Returns:
        List of combined reward scores in [0.0, 1.0]
    """
    scores = []

    for prompt, completion, answer in zip(prompts, completions, answers):
        score, _ = compute_reward(
            completion=completion,
            ground_truth=answer,
            weights=None,  # Use defaults
            possible_boxed_answers=None,
            format_config=None,  # Use defaults
            prompt_text=prompt,
        )
        scores.append(score)

    return scores


def exam_strict_format_reward_func(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    GRPO-compatible exam strict format reward function.

    Returns 1.0 only if completion has both:
    - Valid <think>...</think> tags with content
    - Valid \\boxed{...} answer

    Otherwise returns 0.0.

    Args:
        prompts: List of input prompts (unused)
        completions: List of model completion texts
        answers: List of ground truth answers (unused)
        types: Optional list of question types (unused)

    Returns:
        List of binary format scores (0.0 or 1.0)
    """
    scores = []
    format_config = DEFAULT_FORMAT_CONFIG

    for completion in completions:
        boxed_score, boxed_details = check_boxed_format(completion, format_config)
        format_score, format_details = check_format_compliance(completion, format_config)

        has_boxed = boxed_details.get("has_boxed", False)
        has_think = format_details.get("has_think_tags", False)

        # Binary: both must be present
        if has_boxed and has_think:
            scores.append(1.0)
        else:
            scores.append(0.0)

    return scores


# =============================================================================
# Register with GRPO Reward Registry
# =============================================================================


def _register_exam_rewards():
    """Register exam reward functions with the GRPO reward registry."""
    try:
        from .rewards.registry import register_reward

        # Register all exam reward functions
        register_reward("exam_accuracy_reward_func", exam_accuracy_reward_func)
        register_reward("exam_format_reward_func", exam_format_reward_func)
        register_reward("exam_reasoning_reward_func", exam_reasoning_reward_func)
        register_reward("exam_combined_reward_func", exam_combined_reward_func)
        register_reward("exam_strict_format_reward_func", exam_strict_format_reward_func)

        # Also register shorter aliases
        register_reward("exam_accuracy", exam_accuracy_reward_func)
        register_reward("exam_format", exam_format_reward_func)
        register_reward("exam_reasoning", exam_reasoning_reward_func)
        register_reward("exam_combined", exam_combined_reward_func)
        register_reward("exam_strict", exam_strict_format_reward_func)

        _logger.debug("Exam reward functions registered with GRPO registry")
    except ImportError:
        _logger.debug("Could not register exam rewards - registry not available")


# Auto-register when module is imported
_register_exam_rewards()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core functions
    "compute_reward",
    "compute_batch_rewards",
    "compute_accuracy_reward",
    "check_boxed_format",
    "check_format_compliance",
    "check_reasoning_quality",
    "extract_boxed_answer",
    "extract_answer_from_completion",
    "normalize_answer",
    "check_answer_match",
    # Configuration
    "RewardWeights",
    "FormatConfig",
    "DEFAULT_FORMAT_CONFIG",
    # GRPO-compatible wrappers
    "exam_accuracy_reward_func",
    "exam_format_reward_func",
    "exam_reasoning_reward_func",
    "exam_combined_reward_func",
    "exam_strict_format_reward_func",
    # Logging
    "StructuredRewardLogger",
    "reconfigure_logging",
    "reconfigure_structured_logging",
]
