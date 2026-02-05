"""
Path: mlx_grpo/trainer/grpo_reward_functions.py
GRPO Reward Functions Module - PRODUCTION VERSION
==================================================
Comprehensive collection of reward functions for Reinforcement Learning on LLMs.
Optimized for Group Relative Policy Optimization (GRPO) training pipelines.

Version: 4.2.0 (CLI Fix - Restored Missing Reward Functions)
Last Updated: 2025-01-18

Features:
- All rewards STRICTLY normalized to [0, 1]
- Zero-dependency core (Soft dependencies handled)
- Full backward compatibility with existing CLI/configs
- Phase-aware rewards for thinking models
- Type-aware reward weight adjustments
- Integrated Anti-Verbosity and Efficiency checks
"""

import re
import math
import logging
import ast
import difflib
import copy
import sys
import tempfile
import subprocess
import os
import functools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Set, Tuple, Any, Union

from .hierarchical_rewards_v3 import batch_hierarchical_reward,hierarchical_reward

# --- Soft Dependencies Configuration ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GRPO_Rewards")

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

RewardFunctions = Callable[
    [List[str], List[str], List[str], Optional[List[str]]], List[float]
]

# Import the centralized registry from rewards package
# This enables both the new decorator API and backward compatibility
from .rewards.registry import REWARD_REGISTRY, reward as _reward_decorator

# =============================================================================
# PRE-COMPILED REGEX PATTERNS
# =============================================================================

# Component extraction
RE_THINK_EXTRACT = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)
RE_ANSWER_TAGS = re.compile(r"</?answer>")
RE_ANSWER_EXTRACT = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Format validation
RE_STRICT_FORMAT = re.compile(r"^<think>\n[\s\S]*?\n</think>\n[\s\S]*$")
RE_STRUCTURED_LIST = re.compile(r"(\n\s*[-*•]|\n\s*\d+\.\s+)")

# MCQ patterns
RE_MCQ_OPTION = re.compile(
    r"(?:^|\s|'|\"|\()([A-D])(?:$|\s|\.|'|\"|\)|:)", re.IGNORECASE
)
RE_MCQ_ANSWER = re.compile(r"answer:\s*([A-D])", re.IGNORECASE)
RE_MCQ_REF = re.compile(r"(?:^|\s)([A-D])(?=$|\s|\.|:|\))", re.IGNORECASE)

# Code blocks
RE_CODE_PYTHON = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
RE_CODE_GENERIC = re.compile(r"```\n(.*?)\n```", re.DOTALL)

# Emoji pattern
RE_EMOJI = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f82f"
    "\U0001f8a0-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U00002600-\U000026ff"
    "\U00002700-\U000027bf"
    "\U0000fe0f"
    "]+"
)

# Tokenization
RE_WORD_TOKENS = re.compile(r"\w+")

# =============================================================================
# PRE-COMPILED PATTERN SETS
# =============================================================================

# Thinking quality - bad phrases
_BAD_PHRASES_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bi think\b",
        r"\bi believe\b",
        r"\bmaybe\b",
        r"\bi'm not sure\b",
        r"\bi will now\b",
        r"\bi'll start by\b",
        r"\blet's see\b",
        r"\bconfused\b",
        r"\bstuck\b",
        r"\bfrustrated\b",
        r"\bwait, wait\b",
        r"\bhmm, perhaps\b",
        r"\bor wait\b",
        r"\bto be completely honest\b",
        r"\bbasically what happens\b",
        r"\blong story short\b",
        r"\bat the end of the day\b",
    ]
]

# Special tokens to penalize
_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<think><think>", "<|im_end|>"]

# Conditional content patterns
_FALSE_IDENTITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(I'?m|I am|my name is|this is)\s+(Qwen|GPT|Claude|LLaMA|Gemini|Mistral|Phi|Bard)",
        r"(developed|created|made|built|trained)\s+(by|at)\s+(Alibaba|OpenAI|Google|Anthropic|Meta|Microsoft)",
        r"\b(Alibaba\s+Cloud|Tongyi\s+Lab|OpenAI|Anthropic|Google\s+DeepMind)\b",
    ]
]

_SAFETY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(Historical Document|Official Document).{0,40}(Events|Incident).{0,40}(1989|June)",
        r"(no|zero|not any)\s+(major|significant)?(incidents?|events?).*Tiananmen.*June",
        r"the\s+government\s+has\s+(always\s+)?(prioritized|emphasized)",
    ]
]

_SIMPLE_Q_PATTERNS = [
    re.compile(
        r"(?i)(what is|who is|when|where|define|identify|什么是|谁是|哪里|你的身份)"
    )
]

_SENSITIVE_PATTERNS = [re.compile(r"(?i)(tiananmen|massacre|protest|六四|天安门|抗议)")]

_HUMILITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"I (don't|do not) have (reliable|verified|current|accurate) information",
        r"my knowledge (may be|is|could be) (limited|outdated|incomplete)",
    ]
]

_STYLE_BAD_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*(like|so|basically|actually|literally|honestly),?\s",
        r"(sorry|apologize).{0,120}(sorry|apologize).{0,120}(sorry|apologize)",
    ]
]

# Factual grounding patterns
_EVIDENCE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"according to",
        r"research shows",
        r"data shows",
        r"documented",
        r"source:",
        r"citation",
    ]
]

_UNCERTAINTY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"may be",
        r"might be",
        r"likely",
        r"possibly",
        r"uncertain",
        r"preliminary",
        r"estimated",
    ]
]

_PROBLEMATIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"definitely",
        r"absolutely",
        r"guaranteed",
        r"proven fact",
        r"undeniable",
        r"everyone knows",
    ]
]

_MISINFO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"conspiracy",
        r"cover.up",
        r"hidden truth",
        r"fake news",
        r"hoax",
        r"propaganda",
    ]
]

# Moral reasoning patterns
_MORAL_POSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ethical",
        r"moral",
        r"fair",
        r"responsible",
        r"compassionate",
        r"harm prevention",
        r"rights",
        r"justice",
    ]
]

_MORAL_NEGATIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"discriminat",
        r"prejudice",
        r"bias",
        r"harmful",
        r"offensive",
        r"unethical",
        r"illegal",
        r"hate",
    ]
]

_MORAL_REASONING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"consider.{0,20}ethical",
        r"consequences",
        r"stakeholders",
        r"potential.{0,10}harm",
        r"benefits?.{0,10}risks",
    ]
]

# =============================================================================
# UNWANTED CONTENT SET
# =============================================================================

_UNWANTED_SET = frozenset(
    {
        # AI Identity & Refusals
        "as an ai",
        "i cannot",
        "cannot fulfill",
        "against my programming",
        "language model",
        "apologize",
        "unable to assist",
        "regenerate response",
        "ethical guidelines",
        "safety guidelines",
        "cannot provide",
        "not appropriate",
        "cannot generate",
        "i am not a doctor",
        "i am not a lawyer",
        "my purpose",
        "limitations",
        "virtual assistant",
        "knowledge cutoff",
        "openai",
        "anthropic",
        # Meta-Cognitive
        "let me think",
        "let me start",
        "first thought",
        "okay, the user",
        "the user wants",
        "analyzing this",
        "breaking this down",
        "here is the answer",
        "hope this helps",
        "thanks for asking",
        "you got it",
        "i recall that",
        # Hate Speech & Harm (subset)
        "hate speech",
        "supremacy",
        "genocide",
        "bioweapon",
        "chemical weapon",
        # Illegal & Explicit (subset)
        "drug trafficking",
        "money laundering",
        "cyberbullying",
        # Misinformation
        "flat earth",
        "chemtrails",
        "illuminati",
        "deep state",
        "qanon",
        "anti-vax",
    }
)

# =============================================================================
# CONTRACTIONS
# =============================================================================

CONTRACTIONS = {
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "won't": "will not",
    "can't": "cannot",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
}

# =============================================================================
# PHASED COMPLETION DATACLASS
# =============================================================================


@dataclass
class PhasedCompletion:
    """
    Structured completion from phased generation.
    Provides clean access to thinking and answer phases with metadata.
    """

    thinking: str = ""
    answer: str = ""
    raw_text: str = ""
    phase_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls, text: str, phase_outputs: Optional[List[Dict]] = None
    ) -> "PhasedCompletion":
        """Parse from raw text (backward compatible)."""
        if not text:
            return cls(raw_text="")

        thinking, answer = _extract_components(text)

        metadata = {}
        if phase_outputs:
            for phase in phase_outputs:
                phase_name = phase.get("phase", phase.get("name", ""))
                if phase_name in ("thinking", "think"):
                    metadata["thinking_tokens"] = phase.get("tokens", 0)
                    metadata["thinking_hit_stop"] = phase.get("hit_stop", False)
                    metadata["thinking_time"] = phase.get("time", 0.0)
                elif phase_name in ("answer", "response"):
                    metadata["answer_tokens"] = phase.get("tokens", 0)
                    metadata["answer_hit_stop"] = phase.get("hit_stop", False)
                    metadata["answer_time"] = phase.get("time", 0.0)

        return cls(
            thinking=thinking or "",
            answer=answer,
            raw_text=text,
            phase_metadata=metadata,
        )

    def to_text(self) -> str:
        """Convert back to raw text format."""
        if self.raw_text:
            return self.raw_text
        if self.thinking:
            return f"<think>{self.thinking}</think>{self.answer}"
        return self.answer

    @property
    def has_thinking(self) -> bool:
        return bool(self.thinking and len(self.thinking.strip()) > 0)


# =============================================================================
# CORE UTILITIES
# =============================================================================


def _extract_components(text: str) -> Tuple[Optional[str], str]:
    """
    Extract (thinking_content, answer_content) from completion.
    """
    if not text:
        return None, ""

    # No think tags
    if "<think>" not in text:
        answer_match = RE_ANSWER_EXTRACT.search(text)
        if answer_match:
            return None, answer_match.group(1).strip()
        return None, text.strip()

    # Has <think> tag
    match = RE_THINK_EXTRACT.search(text)
    if match:
        thinking_content = match.group(1).strip()
        answer_content = match.group(2).strip()
        answer_content = RE_ANSWER_TAGS.sub("", answer_content).strip()
        return thinking_content, answer_content

    # Partial tags
    if "<think>" in text and "</think>" not in text:
        return None, ""

    return None, text.strip()


def r1_extract_xml_answer(text: str) -> str:
    """
    Extract answer from completion. Priority:
    1. <answer> tags
    2. Content after </think>
    3. Full text if no tags
    """
    if not text:
        return ""

    answer_match = RE_ANSWER_EXTRACT.search(text)
    if answer_match:
        return answer_match.group(1).strip()

    if "</think>" in text:
        after_think = text.split("</think>", 1)[-1]
        after_think = RE_ANSWER_TAGS.sub("", after_think)
        return after_think.strip()

    if "<think>" in text and "</think>" not in text:
        return ""

    return text.strip()


def _clean_text_basic(text: str) -> str:
    """Robust text cleaner for similarity comparison."""
    if not text:
        return ""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _check_any_pattern(patterns: List[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _count_pattern_matches(patterns: List[re.Pattern], text: str) -> int:
    return sum(1 for p in patterns if p.search(text))


def _jaccard_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    s1 = set(RE_WORD_TOKENS.findall(text1.lower()))
    s2 = set(RE_WORD_TOKENS.findall(text2.lower()))
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# INPUT VALIDATION DECORATOR
# =============================================================================


def validate_inputs(func: RewardFunctions) -> RewardFunctions:
    """Decorator to validate inputs and ensure scores are in [0, 1]."""

    @functools.wraps(func)
    def wrapper(
        prompts: List[str],
        completions: List[str],
        answer: List[str],
        types: Optional[List[str]] = None,
    ) -> List[float]:
        # Handle empty completions
        if not completions:
            logger.warning(f"{func.__name__}: Empty completions list, returning zeros")
            return [0.0] * max(len(prompts) if prompts else 1, 1)

        # Normalize lengths
        len_c = len(completions)

        # Adjust answers
        current_answers = answer
        if len(answer) != len_c:
            if len(answer) == 1:
                current_answers = [answer[0]] * len_c
            elif len(answer) > len_c:
                current_answers = answer[:len_c]
            else:
                current_answers = answer + [answer[-1]] * (len_c - len(answer))

        # Adjust prompts
        current_prompts = prompts
        if prompts and len(prompts) != len_c:
            if len(prompts) == 1:
                current_prompts = [prompts[0]] * len_c
            elif len(prompts) > len_c:
                current_prompts = prompts[:len_c]
            else:
                current_prompts = prompts + [prompts[-1]] * (len_c - len(prompts))

        # Adjust types
        current_types = types
        if types and len(types) != len_c:
            if len(types) == 1:
                current_types = [types[0]] * len_c
            elif len(types) > len_c:
                current_types = types[:len_c]
            else:
                current_types = types + [types[-1]] * (len_c - len(types))

        try:
            scores = func(current_prompts, completions, current_answers, current_types)
            validated_scores = []
            for s in scores:
                if not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s):
                    validated_scores.append(0.0)
                else:
                    validated_scores.append(max(0.0, min(1.0, float(s))))
            return validated_scores

        except Exception as e:
            logger.error(f"{func.__name__} CRASHED: {e}", exc_info=True)
            return [0.0] * len(completions)

    return wrapper


# =============================================================================
# REGISTRY FUNCTIONS
# =============================================================================


def register_reward_function(name: str = None):
    def decorator(func: RewardFunctions):
        func_name = name or func.__name__
        validated_func = validate_inputs(func)
        REWARD_REGISTRY[func_name] = validated_func
        return validated_func

    return decorator


def get_reward_function(name: str) -> RewardFunctions:
    if name not in REWARD_REGISTRY:
        available = ", ".join(sorted(REWARD_REGISTRY.keys()))
        raise KeyError(f"Reward function '{name}' not found. Available: {available}")
    return REWARD_REGISTRY[name]


def get_default_reward_functions() -> List[RewardFunctions]:
    """Get default set of reward functions for GRPO training."""
    return [
        REWARD_REGISTRY.get("r1_accuracy_reward_func", r1_accuracy_reward_func),
        REWARD_REGISTRY.get(
            "r1_semantic_similarity_reward", r1_semantic_similarity_reward
        ),
        REWARD_REGISTRY.get("r1_thinking_quality_reward", r1_thinking_quality_reward),
        REWARD_REGISTRY.get("r1_answer_quality_reward", r1_answer_quality_reward),
        REWARD_REGISTRY.get("r1_format_reward", r1_format_reward),
    ]


# =============================================================================
# ANTI-VERBOSITY & EFFICIENCY REWARDS
# =============================================================================


@register_reward_function("r1_conciseness_reward")
def r1_conciseness_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    Conciseness Reward - Type-Aware Length Optimization.
    Penalizes both too short (incomplete) and too long (verbose).
    """
    scores = []
    # Target Ranges: (min, optimal_low, optimal_high, max_acceptable)
    TYPE_TARGETS = {
        "math": (20, 50, 200, 300),
        "code": (50, 100, 400, 600),
        "essay": (100, 200, 600, 1000),
        "mcq": (10, 20, 100, 150),
        "reasoning": (40, 80, 300, 500),
    }
    DEFAULT_TARGET = (20, 50, 300, 500)

    for i, text in enumerate(completions):
        if not text or len(text.strip()) < 10:
            scores.append(0.0)
            continue

        qtype = types[i].lower() if types and i < len(types) and types[i] else None
        min_len, opt_low, opt_high, max_accept = TYPE_TARGETS.get(qtype, DEFAULT_TARGET)

        tokens = len(text.split())

        if tokens < min_len:
            score = max(0.2, tokens / min_len)
        elif opt_low <= tokens <= opt_high:
            score = 1.0
        elif tokens <= max_accept:
            excess_ratio = (tokens - opt_high) / (max_accept - opt_high)
            score = 1.0 - (0.3 * excess_ratio)
        else:
            excess_ratio = (tokens - max_accept) / max_accept
            score = max(0.1, 0.7 - (0.4 * min(excess_ratio, 3.0)))

        scores.append(score)
    return scores


@register_reward_function("r1_repetition_penalty_reward")
def r1_repetition_penalty_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    Repetition Penalty Reward - Detects Semantic Duplication.
    """
    scores = []

    def compute_ngram_repetition(text: str, n: int = 4) -> float:
        words = text.lower().split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        return 1.0 - (len(set(ngrams)) / len(ngrams))

    def compute_sentence_similarity(text: str) -> float:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        if len(sentences) < 2:
            return 0.0
        similarities = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sim = difflib.SequenceMatcher(None, sentences[i], sentences[j]).ratio()
                similarities.append(sim)
        return sum(similarities) / len(similarities) if similarities else 0.0

    for text in completions:
        if not text or len(text.strip()) < 20:
            scores.append(1.0)
            continue

        think, ans = _extract_components(text)
        full_text = (think or "") + " " + ans
        score = 1.0

        ngram_rep = compute_ngram_repetition(full_text, n=4)
        if ngram_rep > 0.3:
            score -= min(0.4, ngram_rep)

        if think and len(think.split()) > 50:
            sent_sim = compute_sentence_similarity(think)
            if sent_sim > 0.7:
                score -= 0.3

        common_phrases = ["i think", "let me", "okay", "wait", "hmm", "so basically"]
        for phrase in common_phrases:
            count = full_text.lower().count(phrase)
            if count > 3:
                score -= min(0.2, 0.05 * (count - 3))

        scores.append(score)
    return scores


@register_reward_function("r1_information_density_reward")
def r1_information_density_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Information Density Reward - Signal-to-Noise Ratio."""
    scores = []

    INFO_MARKERS = {
        "numbers": re.compile(r"\b\d+(?:\.\d+)?(?:%|kg|m|km|GB|MB)?\b"),
        "technical": re.compile(r"\b[A-Z]{2,}|\w+_\w+|[a-z]+\(\)|<[^>]+>"),
        "citations": re.compile(
            r"according to|research shows|study found|data indicates"
        ),
    }
    FILLER_WORDS = {
        "basically",
        "actually",
        "literally",
        "really",
        "very",
        "quite",
        "somewhat",
        "rather",
        "fairly",
        "pretty",
        "kind of",
        "sort of",
    }

    for text in completions:
        if not text or len(text.strip()) < 20:
            scores.append(0.5)
            continue

        _, ans = _extract_components(text)
        words = ans.lower().split()
        total_words = len(words)
        if total_words < 10:
            scores.append(0.5)
            continue

        score = 0.6
        if INFO_MARKERS["numbers"].search(ans):
            score += 0.1
        if INFO_MARKERS["technical"].search(ans):
            score += 0.1
        if INFO_MARKERS["citations"].search(ans):
            score += 0.1

        filler_count = sum(1 for word in words if word in FILLER_WORDS)
        if (filler_count / total_words) > 0.2:
            score -= 0.3

        scores.append(score)
    return scores


@register_reward_function("r1_thinking_efficiency_reward")
def r1_thinking_efficiency_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Thinking Efficiency Reward - Quality per Token."""
    scores = []
    OPTIMAL_THINKING = {
        "math": (20, 100),
        "code": (30, 150),
        "essay": (50, 200),
        "mcq": (10, 50),
        "reasoning": (30, 120),
    }
    DEFAULT_OPTIMAL = (20, 100)

    for i, text in enumerate(completions):
        think, ans = _extract_components(text)
        ref = answer[i] if i < len(answer) else ""

        is_correct = ans.strip().lower() == ref.strip().lower()
        score = 0.5 if is_correct else 0.2

        if not think:
            scores.append(score * 0.5)
            continue

        qtype = types[i].lower() if types and i < len(types) and types[i] else None
        opt_min, opt_max = OPTIMAL_THINKING.get(qtype, DEFAULT_OPTIMAL)
        think_tokens = len(think.split())

        if opt_min <= think_tokens <= opt_max:
            score += 0.3
        elif think_tokens < opt_min:
            score += 0.15 * (think_tokens / opt_min)
        else:
            excess = (think_tokens - opt_max) / opt_max
            score += max(0.05, 0.3 - (0.2 * min(excess, 2.0)))

        scores.append(score)
    return scores


# =============================================================================
# CONFIGURATION GETTERS (Updated with Anti-Verbosity)
# =============================================================================


def get_default_reward_configs() -> Dict[str, Dict[str, Any]]:
    """Get default reward configurations with weights and metadata."""
    return {
        "accuracy": {
            "func": REWARD_REGISTRY.get(
                "r1_accuracy_reward_func", r1_accuracy_reward_func
            ),
            "weight": 0.25,
            "description": "Exact match accuracy",
        },
        "semantic": {
            "func": REWARD_REGISTRY.get(
                "r1_semantic_similarity_reward", r1_semantic_similarity_reward
            ),
            "weight": 0.20,
            "description": "TF-IDF semantic similarity",
        },
        "thinking": {
            "func": REWARD_REGISTRY.get(
                "r1_thinking_quality_reward", r1_thinking_quality_reward
            ),
            "weight": 0.10,
            "description": "Reasoning quality",
        },
        "answer_quality": {
            "func": REWARD_REGISTRY.get(
                "r1_answer_quality_reward", r1_answer_quality_reward
            ),
            "weight": 0.10,
            "description": "Anti-gaming checks",
        },
        "format": {
            "func": REWARD_REGISTRY.get("r1_format_reward", r1_format_reward),
            "weight": 0.05,
            "description": "Format compliance",
        },
        # Anti-verbosity rewards
        "conciseness": {
            "func": REWARD_REGISTRY.get("r1_conciseness_reward", r1_conciseness_reward),
            "weight": 0.15,
            "description": "Type-aware length optimization",
        },
        "repetition": {
            "func": REWARD_REGISTRY.get(
                "r1_repetition_penalty_reward", r1_repetition_penalty_reward
            ),
            "weight": 0.10,
            "description": "Penalizes semantic duplication",
        },
        "efficiency": {
            "func": REWARD_REGISTRY.get(
                "r1_thinking_efficiency_reward", r1_thinking_efficiency_reward
            ),
            "weight": 0.05,
            "description": "Quality per token",
        },
    }


def get_type_adjusted_weights(question_type: Optional[str] = None) -> Dict[str, float]:
    """Get reward weights adjusted for question type."""
    DEFAULT_WEIGHTS = {
        "accuracy": 0.25,
        "semantic": 0.20,
        "thinking": 0.10,
        "answer_quality": 0.10,
        "format": 0.05,
        "conciseness": 0.15,
        "repetition": 0.10,
        "efficiency": 0.05,
    }

    TYPE_ADJUSTMENTS = {
        "math": {
            "accuracy": 0.40,
            "conciseness": 0.20,
            "thinking": 0.15,
            "semantic": 0.10,
            "efficiency": 0.10,
            "answer_quality": 0.03,
            "format": 0.01,
            "repetition": 0.01,
        },
        "code": {
            "accuracy": 0.35,
            "conciseness": 0.15,
            "efficiency": 0.15,
            "thinking": 0.15,
            "format": 0.10,
            "semantic": 0.05,
            "answer_quality": 0.03,
            "repetition": 0.02,
        },
        "essay": {
            "semantic": 0.30,
            "thinking": 0.20,
            "answer_quality": 0.15,
            "conciseness": 0.15,
            "accuracy": 0.10,
            "repetition": 0.05,
            "efficiency": 0.03,
            "format": 0.02,
        },
        "mcq": {
            "accuracy": 0.50,
            "conciseness": 0.20,
            "format": 0.15,
            "efficiency": 0.10,
            "thinking": 0.03,
            "semantic": 0.01,
            "answer_quality": 0.01,
            "repetition": 0.00,
        },
        "reasoning": {
            "thinking": 0.25,
            "accuracy": 0.25,
            "conciseness": 0.15,
            "semantic": 0.15,
            "efficiency": 0.10,
            "repetition": 0.05,
            "answer_quality": 0.03,
            "format": 0.02,
        },
    }

    if question_type and question_type.lower() in TYPE_ADJUSTMENTS:
        weights = TYPE_ADJUSTMENTS[question_type.lower()]
    else:
        weights = DEFAULT_WEIGHTS.copy()

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def list_available_reward_functions() -> List[str]:
    return sorted(list(REWARD_REGISTRY.keys()))


# =============================================================================
# PHASE-AWARE WRAPPER
# =============================================================================


def make_phase_aware_reward(
    base_reward_func: RewardFunctions,
    phase_weight_thinking: float = 0.3,
    phase_weight_answer: float = 0.7,
) -> RewardFunctions:
    """Wrap a reward function to be phase-aware."""

    @functools.wraps(base_reward_func)
    def wrapper(
        prompts: List[str],
        completions: List[str],
        answer: List[str],
        types: Optional[List[str]] = None,
        phase_outputs: Optional[List[List[Dict]]] = None,
    ) -> List[float]:
        if phase_outputs is None:
            return base_reward_func(prompts, completions, answer, types)

        scores = []
        for i, (comp, ref) in enumerate(zip(completions, answer)):
            phased = PhasedCompletion.from_text(
                comp, phase_outputs[i] if i < len(phase_outputs) else None
            )
            prompt = prompts[i] if prompts and i < len(prompts) else ""
            qtype = types[i] if types and i < len(types) else None

            think_score = 0.0
            if phased.has_thinking:
                try:
                    think_text = f"<think>{phased.thinking}</think>"
                    think_score = base_reward_func(
                        [prompt], [think_text], [ref], [qtype]
                    )[0]
                except Exception:
                    think_score = 0.0

            try:
                ans_score = base_reward_func([prompt], [phased.answer], [ref], [qtype])[
                    0
                ]
            except Exception:
                ans_score = 0.0

            combined = (
                (phase_weight_thinking * think_score + phase_weight_answer * ans_score)
                if phased.has_thinking
                else ans_score
            )
            scores.append(max(0.0, min(1.0, combined)))
        return scores

    return wrapper


# =============================================================================
# MAIN REWARD FUNCTIONS
# =============================================================================


@register_reward_function("r1_format_reward")
def r1_format_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Format Reward - Strict Mode."""
    scores = []
    MIN_CONTENT_LEN = 5

    for text in completions:
        if not text:
            scores.append(0.0)
            continue

        open_pos = text.find("<think>")
        close_pos = text.find("</think>")

        if open_pos == -1 or close_pos == -1 or open_pos >= close_pos:
            scores.append(0.0)
            continue

        content_after = text[close_pos + len("</think>") :].strip()
        if len(content_after) < MIN_CONTENT_LEN:
            scores.append(0.0)
            continue

        scores.append(1.0)
    return scores


@register_reward_function("r1_tag_structure_reward")
def r1_tag_structure_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    Reward for having proper XML tag structure.
    Checks for exactly one <think> and one </think> tag, in correct order.
    """
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue

        think_count = text.count("<think>")
        end_think_count = text.count("</think>")

        # Exact match required: 1 opening, 1 closing
        if think_count == 1 and end_think_count == 1:
            # Check order
            if text.find("<think>") < text.find("</think>"):
                scores.append(1.0)
            else:
                scores.append(0.0)  # Wrong order
        else:
            scores.append(0.0)  # Missing or extra tags

    return scores


@register_reward_function("r1_thinking_quality_reward")
def r1_thinking_quality_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Thinking Quality Reward - Evaluates reasoning structure and content."""
    scores = []
    for text in completions:
        think, _ = _extract_components(text)
        if not think:
            scores.append(0.0)
            continue

        score = 1.0
        bad_count = sum(1 for p in _BAD_PHRASES_PATTERNS if p.search(think))
        if bad_count > 0:
            score -= min(0.4, 0.1 * bad_count)

        for token in _SPECIAL_TOKENS:
            if token in think:
                score -= 0.3

        if RE_STRUCTURED_LIST.search(think):
            score += 0.1

        approx_tokens = len(think.split())
        if approx_tokens < 20:
            score *= max(0.3, approx_tokens / 20)
        elif approx_tokens > 80:
            excess = (approx_tokens - 80) / 80
            score *= max(0.5, 1.0 - 0.2 * excess)
        elif 30 <= approx_tokens <= 60:
            score *= 1.1

        scores.append(max(0.0, min(1.0, score)))
    return scores


@register_reward_function("r1_answer_quality_reward")
def r1_answer_quality_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Answer Quality - Checks for unwanted content and gaming."""
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        _, ans = _extract_components(text)
        if len(ans) < 5:
            scores.append(0.0)
            continue

        score = 1.0
        ans_lower = ans.lower()

        # Check unwanted set
        ans_words = set(ans_lower.split())
        if ans_words & _UNWANTED_SET:
            matches = ans_words & _UNWANTED_SET
            if len(matches) > 1 or any(phrase in ans_lower for phrase in matches):
                scores.append(0.0)
                continue

        for phrase in _UNWANTED_SET:
            if " " in phrase and phrase in ans_lower:
                scores.append(0.0)
                break
        else:
            if RE_EMOJI.search(ans):
                score -= 0.1
            scores.append(max(0.0, min(1.0, score)))
            continue

    return scores


@register_reward_function("r1_semantic_similarity_reward")
def r1_semantic_similarity_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Semantic Similarity - TF-IDF with fallback."""
    scores = []

    def compute_tfidf(text1: str, text2: str) -> Optional[float]:
        if not SKLEARN_AVAILABLE:
            return None
        try:
            c1, c2 = _clean_text_basic(text1), _clean_text_basic(text2)
            if len(c1.split()) < 3 or len(c2.split()) < 3:
                return None
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf = vectorizer.fit_transform([c1, c2])
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except Exception:
            return None

    for gen, ref in zip(completions, answer):
        _, gen_ans = _extract_components(gen)
        _, ref_ans = _extract_components(ref)

        if not gen_ans or not ref_ans:
            scores.append(0.0)
            continue

        if gen_ans.strip().lower() == ref_ans.strip().lower():
            scores.append(1.0)
            continue

        sim = compute_tfidf(gen_ans, ref_ans)
        if sim is None:
            sim = max(_jaccard_similarity(gen_ans, ref_ans), 0.0)

        # Gentle verbosity penalty
        len_ratio = len(gen_ans) / max(len(ref_ans), 1)
        if len_ratio > 2.0:
            sim *= max(0.5, 1.0 - 0.1 * (len_ratio - 2.0))

        scores.append(max(0.0, min(1.0, sim)))
    return scores


@register_reward_function("r1_accuracy_reward_func")
def r1_accuracy_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Exact match accuracy reward."""
    scores = []
    for c, a in zip(completions, answer):
        pred = r1_extract_xml_answer(c).strip().lower()
        ref = a.strip().lower()
        scores.append(1.0 if pred == ref else 0.0)
    return scores


@register_reward_function("r1_conditional_content_reward")
def r1_conditional_content_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Hierarchical Safety and Style Reward."""
    scores = []
    for i, text in enumerate(completions):
        if not text:
            scores.append(0.0)
            continue
        _, ans = _extract_components(text)
        score = 1.0

        if _check_any_pattern(_FALSE_IDENTITY_PATTERNS, text):
            score -= 0.5
        if _check_any_pattern(_SAFETY_PATTERNS, ans):
            score -= 0.4
        if _check_any_pattern(_HUMILITY_PATTERNS, ans):
            score += 0.1
        if _check_any_pattern(_STYLE_BAD_PATTERNS, ans):
            score -= 0.1

        scores.append(max(0.0, min(1.0, score)))
    return scores


@register_reward_function("r1_code_execution_reward")
def r1_code_execution_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Code Execution Reward - AST check + Safe Execution."""
    scores = []

    def safe_execute(code_str: str) -> float:
        fname = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code_str)
                fname = f.name
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True,
                text=True,
                timeout=3,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return 1.0 if result.returncode == 0 else 0.0
        except Exception:
            return 0.0
        finally:
            if fname and os.path.exists(fname):
                os.remove(fname)

    for text in completions:
        blocks = RE_CODE_PYTHON.findall(text)
        if not blocks:
            blocks = RE_CODE_GENERIC.findall(text)
        if not blocks:
            scores.append(0.5)
            continue

        full_code = "\n".join(blocks)
        try:
            ast.parse(full_code)
            exec_score = safe_execute(full_code)
            scores.append(0.3 + (0.7 * exec_score))
        except SyntaxError:
            scores.append(0.0)

    return scores


@register_reward_function("r1_mcq_accuracy_reward")
def r1_mcq_accuracy_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """MCQ Accuracy - robust extraction of A/B/C/D."""
    scores = []
    for gen, ref in zip(completions, answer):
        _, gen_ans = _extract_components(gen)
        preds = []
        preds.extend(RE_MCQ_OPTION.findall(gen_ans))
        preds.extend(RE_MCQ_ANSWER.findall(gen_ans))

        if not preds:
            scores.append(0.0)
            continue

        pred = preds[-1].upper()
        ref_match = RE_MCQ_REF.search(ref)
        ref_clean = ref_match.group(1).upper() if ref_match else ref.strip().upper()[:1]

        scores.append(1.0 if pred == ref_clean else 0.0)
    return scores


@register_reward_function("r1_anti_gaming_reward")
def r1_anti_gaming_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Anti-Gaming - Catch exploits like copy-pasting prompt or gibberish."""
    scores = []
    for i, text in enumerate(completions):
        if not text or len(text) < 10:
            scores.append(0.0)
            continue

        # Gibberish check
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        if (vowels / len(text)) < 0.15:
            scores.append(0.0)
            continue

        # Tag stuffing
        if "<think></think>" in text:
            scores.append(0.0)
            continue

        scores.append(1.0)
    return scores


# =============================================================================
# VCTR REWARDS (Velocity to Correct Thinking)
# =============================================================================


@register_reward_function("r1_velocity_to_correct_thinking_reward")
def r1_velocity_to_correct_thinking_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """VCTR - Measures how quickly reasoning converges to correct answer."""
    CONVERGENCE_THRESHOLD = 0.6
    EARLY_BONUS = 1.0
    CONSISTENCY_WEIGHT = 0.25

    scores = []
    for completion, ref in zip(completions, answer):
        try:
            match = re.search(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
            thinking = (
                [l.strip() for l in match.group(1).split("\n") if l.strip()]
                if match
                else []
            )
            _, gen_ans = _extract_components(completion)

            # Consistency Score (Ratio)
            consistency = 0.0
            if thinking and gen_ans:
                last_thought = thinking[-1].lower()
                first_ans = gen_ans.split("\n")[0].strip().lower()
                if len(first_ans) > 4:
                    consistency = difflib.SequenceMatcher(
                        None, last_thought, first_ans
                    ).ratio()

            if not thinking or len(thinking) < 2:
                is_correct = gen_ans.strip().lower() == ref.strip().lower()
                base = 0.3 if is_correct else 0.0
                scores.append(min(1.0, base + CONSISTENCY_WEIGHT * consistency))
                continue

            sims = [_jaccard_similarity(line, ref) for line in thinking]
            k = next(
                (i + 1 for i, s in enumerate(sims) if s > CONVERGENCE_THRESHOLD), None
            )

            if k is None:
                is_correct = gen_ans.strip().lower() == ref.strip().lower()
                base = 0.3 if is_correct else 0.0
                scores.append(min(1.0, base + CONSISTENCY_WEIGHT * consistency))
                continue

            early = EARLY_BONUS / k
            exploration_penalty = 0.3 if k < 3 else 0.0
            trajectory = (sims[k - 1] - sims[0]) * 0.3
            is_correct = gen_ans.strip().lower() == ref.strip().lower()
            answer_bonus = 1.0 if is_correct else 0.0

            total = (
                early
                - exploration_penalty
                + trajectory
                + answer_bonus
                + (CONSISTENCY_WEIGHT * consistency)
            )
            scores.append(max(0.0, min(1.0, total / 2.75)))

        except Exception:
            scores.append(0.0)
    return scores


# =============================================================================
# RESTORED LEGACY FUNCTIONS (Fixes ImportError & CLI Errors)
# =============================================================================


@register_reward_function("r1_count_xml")
def r1_count_xml(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """
    Count XML tag presence (graduated scoring).
    Restored for compatibility with grpo_trainer.py.
    """
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        count = 0.0
        if "<think>" in text:
            count += 0.25
        if "</think>" in text:
            count += 0.25
        if "</think>" in text and len(text.split("</think>")[-1].strip()) > 0:
            count += 0.5
        scores.append(min(1.0, count))
    return scores


@register_reward_function("r1_soft_format_reward_func")
def r1_soft_format_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Soft format check - just requires think tags present."""
    return [
        1.0 if "<think>" in (c or "") and "</think>" in (c or "") else 0.0
        for c in completions
    ]


@register_reward_function("r1_strict_format_reward_func")
def r1_strict_format_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Strict format check."""
    return [
        1.0 if RE_STRICT_FORMAT.search((c or "").strip()) else 0.0 for c in completions
    ]


@register_reward_function("r1_int_reward_func")
def r1_int_reward_func(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Check if extracted answer is an integer."""
    return [
        0.5 if r1_extract_xml_answer(c).strip().isdigit() else 0.0 for c in completions
    ]


# =============================================================================
# ADDITIONAL UTILS & TESTING
# =============================================================================


@register_reward_function("r1_factual_accuracy_reward")
def r1_factual_accuracy_reward(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    types: Optional[List[str]] = None,
) -> List[float]:
    """Check response against specific factual constraints."""
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        score = 0.5
        text_lower = text.lower()

        if "death toll" in text_lower and any(
            x in text_lower for x in ["100", "200-300"]
        ):
            score -= 0.3

        scores.append(max(0.0, min(1.0, score)))
    return scores



@register_reward_function("hierarchical_rewards")
def hierarchical_rewards(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    types: Optional = None,
) -> List[float]:
    """
    Hierarchical rewards with answer match bonus.
    
    Combines:
    1. Base hierarchical reward (semantic similarity, format, quality)
    2. Answer match bonus (exact/substring match of answer section)
    
    This ensures that completions with correct answers get higher rewards
    than those with semantically similar but wrong content (e.g., wrong identity).
    """
    scores = []
    for prompt, completion, answer in zip(prompts, completions, answers):
        # Get base hierarchical score
        base_score, _ = hierarchical_reward(
            response=completion,
            expected=answer,
            question=prompt
        )
        
        # Extract answer sections for direct comparison
        _, completion_answer = _extract_components(completion)
        _, target_answer = _extract_components(answer)
        
        # Normalize for comparison
        comp_ans_norm = completion_answer.strip().lower()
        target_ans_norm = target_answer.strip().lower()
        
        # Calculate answer match bonus
        answer_bonus = 0.0
        if comp_ans_norm and target_ans_norm:
            # Exact match
            if comp_ans_norm == target_ans_norm:
                answer_bonus = 0.3
            # Target answer is contained in completion answer
            elif target_ans_norm in comp_ans_norm:
                answer_bonus = 0.25
            # Completion answer is contained in target (partial match)
            elif comp_ans_norm in target_ans_norm:
                ratio = len(comp_ans_norm) / len(target_ans_norm)
                answer_bonus = 0.2 * ratio
            # Word overlap check
            else:
                target_words = set(target_ans_norm.split())
                comp_words = set(comp_ans_norm.split())
                if target_words and comp_words:
                    overlap = len(target_words & comp_words) / len(target_words)
                    answer_bonus = 0.15 * overlap
        
        # Check for identity mismatch penalty
        # If target mentions specific identity but completion has different one
        identity_penalty = 0.0
        wrong_identities = ['qwen', 'gpt', 'claude', 'llama', 'gemini', 'mistral', 
                          'alibaba', 'openai', 'anthropic', 'google', 'meta']
        
        # Only apply penalty if target has a specific identity AND completion has wrong one
        target_lower = answer.lower()
        completion_lower = completion.lower()
        
        has_target_identity = any(name in target_lower for name in ['neuralai', 'adeel'])
        if has_target_identity:
            for wrong_id in wrong_identities:
                if wrong_id in completion_lower and wrong_id not in target_lower:
                    identity_penalty = 0.2
                    break
        
        # Combine scores
        final_score = base_score + answer_bonus - identity_penalty
        final_score = max(0.0, min(1.0, final_score))
        
        scores.append(final_score)

    return scores



if __name__ == "__main__":
    print("GRPO Reward Functions v4.2.0")
    print("=" * 50)
    print(f"Registered Functions: {len(REWARD_REGISTRY)}")

    # Simple Test
    prompts = ["What is 2+2?"]
    comps = ["<think>2+2 is 4</think>4"]
    ans = ["4"]

    for name, func in REWARD_REGISTRY.items():
        try:
            s = func(prompts, comps, ans, ["math"])
            print(f"✅ {name}: {s[0]:.3f}")
        except Exception as e:
            print(f"❌ {name}: {e}")
