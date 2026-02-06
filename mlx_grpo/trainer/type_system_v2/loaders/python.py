"""
Python Dataset Loader - Data Loading for python tasks
=========================================

Expected format:
    {"prompt": "...", "answer": "...", "type": "python"}
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TYPE_CHECKING

from ..base.dataset_loader import BaseDatasetLoader, DatasetHooks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["PythonDatasetLoader"]

logger = logging.getLogger(__name__)

# Type aliases that map to python
_PYTHON_TYPE_ALIASES = frozenset({
    "python", "code", "coding", "programming", "py",
})

RE_CODE_MARKERS = re.compile(
    r"(?:def |class |import |from .+ import|```python|```py)", re.IGNORECASE
)


class PythonDatasetLoader(BaseDatasetLoader):
    """Dataset loader for python tasks."""

    type_name = "python"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(tokenizer=tokenizer, hooks=hooks, event_bus=event_bus)

    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate python/code sample - detects code markers in prompt or answer."""
        if "prompt" not in sample:
            return False, "Missing 'prompt' field"
        if "answer" not in sample:
            return False, "Missing 'answer' field"

        sample_type = str(sample.get("type", "")).lower()
        if sample_type in _PYTHON_TYPE_ALIASES:
            return True, None

        # Detect code content in answer
        answer = str(sample["answer"])
        if RE_CODE_MARKERS.search(answer):
            return True, None

        # Detect code request in prompt
        prompt = str(sample["prompt"]).lower()
        code_words = ["write a function", "implement", "write code", "python",
                      "def ", "class ", "algorithm"]
        if any(w in prompt for w in code_words):
            return True, None

        return False, "Not a python/code sample"

    def preprocess_sample(self, sample: dict) -> dict:
        """Preprocess python/code sample."""
        sample["type_info"] = {
            "type": "python",
            "has_code_block": "```" in str(sample.get("answer", "")),
        }
        return sample

    def get_system_prompt(self, sample: dict) -> str:
        return (
            "You are an expert Python programmer. "
            "Think through your approach in <think>...</think> tags. "
            "Then provide your solution as clean, working Python code "
            "in a ```python code block."
        )

    def get_type_name(self) -> str:
        return "python"
