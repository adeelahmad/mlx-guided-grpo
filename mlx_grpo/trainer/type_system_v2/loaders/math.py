"""
Math Dataset Loader - Data Loading for math tasks
=======================================

Expected format:
    {"prompt": "...", "answer": "...", "type": "math"}
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TYPE_CHECKING

from ..base.dataset_loader import BaseDatasetLoader, DatasetHooks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["MathDatasetLoader"]

logger = logging.getLogger(__name__)

# Type aliases that map to math
_MATH_TYPE_ALIASES = frozenset({
    "math", "arithmetic", "calculus", "algebra", "geometry",
    "number_theory", "combinatorics", "probability", "statistics",
    "gsm8k", "competition_math",
})

RE_NUMERIC_ANSWER = re.compile(r"^-?\d+(?:\.\d+)?$")
RE_BOXED_ANSWER = re.compile(r"\\boxed\{")
RE_GSM8K_ANSWER = re.compile(r"####\s*(.+)$", re.MULTILINE)
RE_MATH_EQUATION = re.compile(r"[=<>≤≥±]|\\(?:frac|sqrt|sum|int|lim)\b")
RE_FRACTION_ANSWER = re.compile(r"^-?\d+/\d+$")


class MathDatasetLoader(BaseDatasetLoader):
    """Dataset loader for math tasks."""

    type_name = "math"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(tokenizer=tokenizer, hooks=hooks, event_bus=event_bus)

    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate math sample - detects numeric, boxed, GSM8K, and equation content."""
        if "prompt" not in sample:
            return False, "Missing 'prompt' field"
        if "answer" not in sample:
            return False, "Missing 'answer' field"

        sample_type = str(sample.get("type", "")).lower()
        if sample_type in _MATH_TYPE_ALIASES:
            return True, None

        answer = str(sample["answer"]).strip()
        if RE_NUMERIC_ANSWER.match(answer):
            return True, None
        if RE_BOXED_ANSWER.search(answer):
            return True, None
        if RE_GSM8K_ANSWER.search(answer):
            return True, None
        if RE_FRACTION_ANSWER.match(answer):
            return True, None
        if RE_MATH_EQUATION.search(str(sample["prompt"])):
            return True, None

        return False, "Not a math sample"

    def preprocess_sample(self, sample: dict) -> dict:
        """Preprocess math sample - extracts GSM8K answers, normalizes format."""
        answer = str(sample["answer"]).strip()
        gsm_match = RE_GSM8K_ANSWER.search(answer)
        if gsm_match:
            answer = gsm_match.group(1).strip().replace(",", "")

        sample["answer"] = answer
        sample["type_info"] = {
            "type": "math",
            "is_boxed": bool(RE_BOXED_ANSWER.search(answer)),
            "is_numeric": bool(RE_NUMERIC_ANSWER.match(answer)),
        }
        return sample

    def get_system_prompt(self, sample: dict) -> str:
        return (
            "You are an expert mathematician. "
            "Work through the problem step by step in <think>...</think> tags. "
            "Show your reasoning clearly with equations. "
            "Then provide your final answer in \\boxed{answer} format."
        )

    def get_type_name(self) -> str:
        return "math"
