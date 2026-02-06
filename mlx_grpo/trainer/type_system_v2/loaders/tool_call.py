"""
ToolCall Dataset Loader - Clean tool calling data
==================================================

Dataset loader for tool/function calling tasks.

Features:
- Validates function call format
- Normalizes function syntax
- Extracts function definitions from prompts
- Ensures NO thinking tags in answers
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TYPE_CHECKING

from ..base.dataset_loader import BaseDatasetLoader, DatasetHooks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["ToolCallDatasetLoader", "ToolCallCleaningHook"]

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL CALL DATASET LOADER
# =============================================================================

class ToolCallDatasetLoader(BaseDatasetLoader):
    """Dataset loader for tool/function calling tasks.

    Expected format:
        {
            "prompt": "Question with function definitions...",
            "answer": "function_name(param=value)",
            "type": "tool_call"
        }

    Validation:
    - Answer must contain function call pattern
    - Answer must NOT contain <think> tags
    - Answer must NOT contain \\boxed{} math syntax

    Preprocessing:
    - Normalizes function call syntax
    - Extracts and validates function definitions
    - Ensures clean tool calling format
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        strict: bool = True,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize ToolCallDatasetLoader.

        Args:
            tokenizer: HuggingFace tokenizer
            strict: If True, reject samples with contamination
            hooks: Observer hooks
            event_bus: Optional event bus
        """
        super().__init__(tokenizer=tokenizer, hooks=hooks, event_bus=event_bus)
        self.strict = strict

    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate tool call sample.

        Checks:
        1. Has prompt and answer fields
        2. Answer contains function call pattern
        3. Answer has NO thinking tags (if strict)
        4. Answer has NO boxed syntax (if strict)
        """
        # Check required fields
        if "prompt" not in sample:
            return False, "Missing 'prompt' field"

        if "answer" not in sample:
            return False, "Missing 'answer' field"

        answer = sample["answer"]

        # Check for function call pattern
        if not self._has_function_call(answer):
            return False, "Answer does not contain function call pattern"

        # Strict checks
        if self.strict:
            # Reject thinking tags
            if "<think>" in answer or "</think>" in answer:
                return False, "Answer contains <think> tags (contamination)"

            # Reject boxed syntax
            if "\\boxed{" in answer or r"\boxed{" in answer:
                return False, "Answer contains \\boxed{} (contamination)"

        return True, None

    def preprocess_sample(self, sample: dict) -> dict:
        """Preprocess tool call sample.

        - Normalizes function call syntax
        - Strips whitespace
        - Removes code block markers if present
        - Extracts function names
        """
        # Normalize answer
        answer = sample["answer"]
        answer = self._normalize_function_call(answer)
        sample["answer"] = answer

        # Extract function names for metadata
        function_names = self._extract_function_names(answer)
        sample["function_names"] = function_names

        # Extract available functions from prompt (if present)
        prompt = sample["prompt"]
        available_functions = self._extract_available_functions(prompt)
        if available_functions:
            sample["available_functions"] = available_functions

        return sample

    def get_system_prompt(self, sample: dict) -> str:
        """Get system prompt for tool calling.

        Returns a simple system message (functions are in the user prompt).
        """
        return "You are a helpful assistant with access to functions. Use them when appropriate."

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _has_function_call(self, text: str) -> bool:
        """Check if text contains function call pattern."""
        # Pattern: function_name(...) or function_name(key=value, ...)
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)'
        return bool(re.search(pattern, text))

    def _extract_function_names(self, text: str) -> list[str]:
        """Extract function names from text.

        Examples:
            "add(a=1, b=2)" -> ["add"]
            "sort(arr)\ncalc(x=5)" -> ["sort", "calc"]
        """
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, text)
        return list(set(matches))  # Unique

    def _normalize_function_call(self, text: str) -> str:
        """Normalize function call syntax.

        - Strip whitespace
        - Remove code block markers
        - Normalize spacing
        """
        text = text.strip()

        # Remove code blocks
        text = re.sub(r'```\w*\n?', '', text)
        text = re.sub(r'```', '', text)

        # Remove thinking tags if present (shouldn't be, but clean anyway)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _extract_available_functions(self, prompt: str) -> list[str]:
        """Extract available function names from prompt.

        Looks for function definitions in the prompt text.
        """
        # Look for "name": "function_name" pattern (common in API docs)
        name_pattern = r'"name"\s*:\s*"([a-zA-Z_][a-zA-Z0-9_]*)"'
        names = re.findall(name_pattern, prompt)

        if names:
            return list(set(names))

        # Fallback: look for function signature patterns
        # def function_name or function function_name
        sig_pattern = r'\b(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        names = re.findall(sig_pattern, prompt)

        return list(set(names)) if names else []


# =============================================================================
# CLEANING HOOK - Auto-clean contaminated samples
# =============================================================================

class ToolCallCleaningHook(DatasetHooks):
    """Hook that automatically cleans contamination from samples.

    Use this to salvage datasets that have some contamination.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.cleaned_count = 0

    def after_load(self, raw_data: list[dict]) -> list[dict]:
        """Clean all samples after loading."""
        cleaned = []
        for sample in raw_data:
            cleaned_sample = self._clean_sample(sample)
            cleaned.append(cleaned_sample)

        if self.cleaned_count > 0:
            logger.info(f"Cleaned {self.cleaned_count} contaminated samples")

        return cleaned

    def _clean_sample(self, sample: dict) -> dict:
        """Remove contamination from a single sample."""
        if "answer" not in sample:
            return sample

        original = sample["answer"]
        cleaned = original

        # Remove thinking tags
        if "<think>" in cleaned or "</think>" in cleaned:
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
            self.cleaned_count += 1
            if self.verbose:
                logger.info(f"Removed thinking tags from sample")

        # Remove boxed answers
        if "\\boxed{" in cleaned or r"\boxed{" in cleaned:
            cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cleaned)
            self.cleaned_count += 1
            if self.verbose:
                logger.info(f"Removed boxed answer from sample")

        # Update if changed
        if cleaned != original:
            sample["answer"] = cleaned.strip()
            sample["was_cleaned"] = True

        return sample


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
# Basic usage
loader = ToolCallDatasetLoader(tokenizer, strict=True)
dataset = loader.load("data/tool_calls.jsonl")

# With auto-cleaning
cleaning_hook = ToolCallCleaningHook(verbose=True)
loader = ToolCallDatasetLoader(
    tokenizer,
    strict=False,  # Allow contaminated samples (will be cleaned)
    hooks=cleaning_hook
)
dataset = loader.load("data/contaminated_tool_calls.jsonl")
"""
