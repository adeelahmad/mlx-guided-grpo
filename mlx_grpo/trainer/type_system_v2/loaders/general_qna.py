"""
GeneralQNA Dataset Loader - Default/Fallback Data Loading
==========================================================

Dataset loader for general Q&A, math reasoning, and thinking tasks.

This is the default loader - accepts any sample with prompt and answer fields
that doesn't match tool_call or MCQ patterns.

Expected Format:
    {"prompt": "What is the capital of France?", "answer": "<think>...</think>Paris"}
    {"prompt": "Solve x^2 = 4", "answer": "<think>x = +/-2</think>\\boxed{2}"}
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from ..base.dataset_loader import BaseDatasetLoader, DatasetHooks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["GeneralQNADatasetLoader"]

logger = logging.getLogger(__name__)


class GeneralQNADatasetLoader(BaseDatasetLoader):
    """Dataset loader for general Q&A and reasoning tasks.

    This is the default/fallback loader. Accepts most samples
    that have prompt and answer fields.
    """

    type_name = "general_qna"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        require_think_tags: bool = False,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(tokenizer=tokenizer, hooks=hooks, event_bus=event_bus)
        self.require_think_tags = require_think_tags

    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate general Q&A sample."""
        if "prompt" not in sample:
            return False, "Missing 'prompt' field"

        if "answer" not in sample:
            return False, "Missing 'answer' field"

        prompt = str(sample["prompt"]).strip()
        answer = str(sample["answer"]).strip()

        if len(prompt) < 5:
            return False, "Prompt too short"

        if len(answer) < 2:
            return False, "Answer too short"

        # Optionally require think tags
        if self.require_think_tags:
            if "<think>" not in answer:
                return False, "Missing <think> tags (required)"

        return True, None

    def preprocess_sample(self, sample: dict) -> dict:
        """Preprocess general Q&A sample."""
        # Build type_info
        sample["type_info"] = {
            "type": "general_qna",
            "is_exam": False,
        }

        return sample

    def get_system_prompt(self, sample: dict) -> str:
        """System prompt for general Q&A tasks."""
        return (
            "You are a helpful and knowledgeable assistant. "
            "Think through your reasoning step by step in <think>...</think> tags. "
            "Then provide your final answer clearly. "
            "For math problems, put your final answer in \\boxed{answer} format."
        )

    def get_type_name(self) -> str:
        return "general_qna"
