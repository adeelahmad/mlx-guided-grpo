"""
ToolCall Dataset Loader - Qwen-native Hermes format
====================================================

Dataset loader for tool/function calling using Qwen's native format.

Expected dataset format (v15+):
    {
        "prompt": "User question only",
        "answer": "<tool_call>\\n{\\"name\\": \\"func\\", \\"arguments\\": {...}}\\n</tool_call>",
        "type": "tool_call",
        "tools": [{"type": "function", "function": {"name": "...", ...}}]
    }

Features:
- Validates Hermes <tool_call> format
- Extracts and validates tool definitions
- Backward-compatible with legacy python-style answers
- Ensures NO thinking tags in answers
"""

from __future__ import annotations

import json
import re
import logging
from typing import Optional, TYPE_CHECKING

from ..base.dataset_loader import BaseDatasetLoader, DatasetHooks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["ToolCallDatasetLoader", "ToolCallCleaningHook"]

logger = logging.getLogger(__name__)

# Regex to match <tool_call>...</tool_call> blocks
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)


# =============================================================================
# TOOL CALL DATASET LOADER
# =============================================================================

class ToolCallDatasetLoader(BaseDatasetLoader):
    """Dataset loader for Qwen-native tool calling.

    Expected format:
        {
            "prompt": "User question",
            "answer": "<tool_call>\\n{...}\\n</tool_call>",
            "type": "tool_call",
            "tools": [{"type": "function", "function": {...}}]
        }

    Also supports legacy format (backward compat):
        {
            "prompt": "Question with embedded function defs...",
            "answer": "function_name(param=value)",
            "type": "tool_call"
        }
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        strict: bool = True,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(tokenizer=tokenizer, hooks=hooks, event_bus=event_bus)
        self.strict = strict

    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate tool call sample.

        Checks:
        1. Has prompt and answer fields
        2. Answer contains Hermes <tool_call> format OR legacy function call
        3. Answer has NO thinking tags (if strict)
        4. Answer has NO boxed syntax (if strict)
        """
        if "prompt" not in sample:
            return False, "Missing 'prompt' field"
        if "answer" not in sample:
            return False, "Missing 'answer' field"

        answer = sample["answer"]

        # Check for Hermes format or legacy function call
        has_hermes = "<tool_call>" in answer
        has_legacy = bool(re.search(r'\b[a-zA-Z_]\w*\s*\([^)]*\)', answer))

        if not has_hermes and not has_legacy:
            return False, "Answer has no tool call pattern (Hermes or legacy)"

        if self.strict:
            if "<think>" in answer or "</think>" in answer:
                return False, "Answer contains <think> tags (contamination)"
            if "\\boxed{" in answer or r"\boxed{" in answer:
                return False, "Answer contains \\boxed{} (contamination)"

        return True, None

    def preprocess_sample(self, sample: dict) -> dict:
        """Preprocess tool call sample.

        - Validates Hermes JSON structure
        - Extracts function names for metadata
        - Validates tools field if present
        """
        answer = sample["answer"]

        # Extract function names from answer
        if "<tool_call>" in answer:
            function_names = self._extract_hermes_function_names(answer)
        else:
            function_names = self._extract_legacy_function_names(answer)

        sample["function_names"] = function_names

        # Validate tools field if present
        tools = sample.get("tools", [])
        if tools:
            tool_names = []
            for t in tools:
                if isinstance(t, dict):
                    func = t.get("function", t)
                    tool_names.append(func.get("name", ""))
            sample["available_functions"] = tool_names

        return sample

    def get_system_prompt(self, sample: dict) -> str:
        """Get minimal system prompt for tool calling.

        Tools are injected by Qwen's chat template via the tools= parameter,
        so we only need a minimal system message here.
        """
        return "You are a helpful assistant."

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _extract_hermes_function_names(self, text: str) -> list[str]:
        """Extract function names from Hermes <tool_call> blocks."""
        names = []
        for match in TOOL_CALL_PATTERN.findall(text):
            try:
                parsed = json.loads(match.strip())
                name = parsed.get("name", "")
                if name:
                    names.append(name)
            except (json.JSONDecodeError, ValueError):
                continue
        return list(set(names))

    def _extract_legacy_function_names(self, text: str) -> list[str]:
        """Extract function names from legacy python-style calls."""
        pattern = r'\b([a-zA-Z_]\w*)\s*\('
        return list(set(re.findall(pattern, text)))


# =============================================================================
# CLEANING HOOK
# =============================================================================

class ToolCallCleaningHook(DatasetHooks):
    """Hook that cleans contamination from tool call samples."""

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

        if "<think>" in cleaned or "</think>" in cleaned:
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
            self.cleaned_count += 1

        if "\\boxed{" in cleaned or r"\boxed{" in cleaned:
            cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cleaned)
            self.cleaned_count += 1

        if cleaned != original:
            sample["answer"] = cleaned.strip()
            sample["was_cleaned"] = True

        return sample
