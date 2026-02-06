"""
MCQ Dataset Loader - Multiple Choice and Exam Data Loading
============================================================

Dataset loader for MCQ and exam-style tasks with ground truth answers.

Expected Formats:
    # MCQ with letter answer
    {"prompt": "What is 2+2? A) 3 B) 4 C) 5 D) 6", "answer": "B", "type": "mcq"}

    # Exam with ground truth
    {"prompt": "Solve: ...", "answer": "42", "ground_truth": "42", "type": "exam"}

    # AIME-style with boxed answers
    {"prompt": "Find x...", "answer": "\\boxed{7}", "possible_boxed_answers": ["7", "007"]}

Validation:
- Has prompt and answer
- Detects MCQ (letter answers A-D) or exam (ground_truth field)
- Accepts type variants: mcq, exam, aime, math500, olympiad
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TYPE_CHECKING

from ..base.dataset_loader import BaseDatasetLoader, DatasetHooks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["MCQDatasetLoader"]

logger = logging.getLogger(__name__)

# Type aliases that map to MCQ
_MCQ_TYPE_ALIASES = frozenset({
    "mcq", "exam", "aime", "math500", "exam_math", "exam_aime",
    "exam_olympiad", "olympiad", "multiple_choice",
})

RE_MCQ_LETTER = re.compile(r"^[A-Da-d]$")


class MCQDatasetLoader(BaseDatasetLoader):
    """Dataset loader for MCQ / exam-style tasks.

    Detects samples with:
    - Single letter answers (A-D)
    - Ground truth field
    - Exam-related type field
    - Possible boxed answers
    """

    type_name = "mcq"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(tokenizer=tokenizer, hooks=hooks, event_bus=event_bus)

    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate MCQ/exam sample."""
        if "prompt" not in sample:
            return False, "Missing 'prompt' field"

        if "answer" not in sample:
            return False, "Missing 'answer' field"

        # Accept if explicit MCQ/exam type
        sample_type = str(sample.get("type", "")).lower()
        if sample_type in _MCQ_TYPE_ALIASES:
            return True, None

        # Accept if has ground_truth field
        if "ground_truth" in sample:
            return True, None

        # Accept if has possible_boxed_answers
        if "possible_boxed_answers" in sample:
            return True, None

        # Accept if answer is a single letter
        answer = str(sample["answer"]).strip()
        if RE_MCQ_LETTER.match(answer):
            return True, None

        return False, "Not an MCQ/exam sample"

    def preprocess_sample(self, sample: dict) -> dict:
        """Preprocess MCQ/exam sample.

        Normalizes answer format and extracts metadata.
        """
        # Normalize answer
        answer = str(sample["answer"]).strip()

        # Extract ground_truth (use answer if not provided)
        ground_truth = sample.get("ground_truth", answer)

        # Extract possible boxed answers
        possible_boxed = sample.get("possible_boxed_answers", None)

        # Determine if MCQ-style (letter) or open-ended
        is_letter_mcq = bool(RE_MCQ_LETTER.match(answer))

        # Build enriched type_info
        sample["type_info"] = {
            "type": "mcq",
            "ground_truth": ground_truth,
            "possible_boxed_answers": possible_boxed,
            "is_exam": True,
            "is_letter_mcq": is_letter_mcq,
        }

        return sample

    def get_system_prompt(self, sample: dict) -> str:
        """System prompt for MCQ/exam tasks."""
        return (
            "You are an expert problem solver. "
            "Work through the problem step by step in <think>...</think> tags. "
            "Then provide your final answer. "
            "For multiple choice, state the letter. "
            "For math problems, put your answer in \\boxed{answer} format."
        )

    def get_type_name(self) -> str:
        return "mcq"
