"""
Built-in Type Handlers
======================

Concrete implementations of type handlers for common use cases.

Each handler is automatically registered via decorators and provides:
- Type-specific reward configurations
- Optimized generation strategies
- Optional curriculum learning

Usage:
    # Handlers auto-register on import
    from mlx_grpo.trainer.type_system import get_type_handler

    handler = get_type_handler("tool_call")
    rewards = handler.get_reward_config()
    strategy = handler.get_generation_strategy()
"""

from __future__ import annotations

import re
from typing import Any

from .registry import (
    BaseDataTypeHandler,
    GenerationStrategy,
    RewardConfig,
    CurriculumConfig,
    register_type,
)

__all__ = [
    "ToolCallingTypeHandler",
    "ThinkingTypeHandler",
    "CodeGenerationTypeHandler",
    "MathReasoningTypeHandler",
    "DefaultTypeHandler",
]


# =============================================================================
# TOOL/FUNCTION CALLING
# =============================================================================

@register_type("tool_call")
class ToolCallingTypeHandler(BaseDataTypeHandler):
    """Handler for function/tool calling tasks.

    Optimized for:
    - Short, structured outputs
    - Exact parameter matching
    - Multi-function call support
    """

    type_name = "tool_call"

    def get_reward_config(self) -> RewardConfig:
        """Use tool calling specific rewards."""
        return RewardConfig(
            functions=(
                "tool_call_exact",
                "tool_call_function",
                "tool_call_overall",
            ),
            weights=(0.3, 0.4, 0.3),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Short generations, no thinking phase."""
        return GenerationStrategy(
            max_length=256,
            temperature=0.7,
            top_p=0.95,
            two_phase=False,
            enforce_thinking=False,
        )

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Check if sample is a tool calling task."""
        # Check explicit type
        if sample.get("type") == "tool_call":
            return True

        # Check for function call pattern in answer
        answer = sample.get("answer", "")
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)', answer):
            return True

        # Check for function definitions in prompt
        prompt = sample.get("prompt", "")
        if "access to the following functions" in prompt.lower():
            return True

        return False


# =============================================================================
# THINKING/REASONING
# =============================================================================

@register_type("thinking")
class ThinkingTypeHandler(BaseDataTypeHandler):
    """Handler for tasks requiring explicit thinking process.

    Optimized for:
    - Two-phase generation (thinking + answer)
    - Reasoning quality evaluation
    - Longer generation lengths
    """

    type_name = "thinking"

    def get_reward_config(self) -> RewardConfig:
        """Use thinking-aware rewards."""
        return RewardConfig(
            functions=(
                "r1_correctness",
                "r1_format",
                "r1_thinking_quality",
                "r1_conciseness",
            ),
            weights=(0.4, 0.2, 0.3, 0.1),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Enable two-phase thinking generation."""
        return GenerationStrategy(
            max_length=1024,
            temperature=0.8,
            top_p=0.95,
            two_phase=True,
            enforce_thinking=True,
            continuation_tokens=256,
            stop_sequences=["</think>"],
        )

    def get_curriculum_config(self) -> CurriculumConfig:
        """Enable curriculum learning for thinking."""
        return CurriculumConfig(
            enabled=True,
            start_ratio=1.0,
            end_ratio=0.0,
            warmup_steps=50,
            strategy="cosine",
        )

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Check if sample requires thinking."""
        # Check explicit type
        if sample.get("type") == "thinking":
            return True

        # Check for <think> tags in answer
        answer = sample.get("answer", "")
        if "<think>" in answer and "</think>" in answer:
            return True

        return False


# =============================================================================
# CODE GENERATION
# =============================================================================

@register_type("code")
class CodeGenerationTypeHandler(BaseDataTypeHandler):
    """Handler for code generation tasks.

    Optimized for:
    - Code syntax validation
    - Execution correctness
    - Code quality metrics
    """

    type_name = "code"

    def get_reward_config(self) -> RewardConfig:
        """Use code-specific rewards."""
        return RewardConfig(
            functions=(
                "r1_correctness",
                "r1_code_syntax",
                "r1_code_execution",
            ),
            weights=(0.4, 0.3, 0.3),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Longer generation for code."""
        return GenerationStrategy(
            max_length=1024,
            temperature=0.7,
            top_p=0.95,
            two_phase=False,
            stop_sequences=["```\n\n", "# End of code"],
        )

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Check if sample is code generation."""
        # Check explicit type
        if sample.get("type") in ("code", "programming"):
            return True

        # Check for code blocks in answer
        answer = sample.get("answer", "")
        if "```" in answer or re.search(r'(def |class |import |function )', answer):
            return True

        # Check for programming keywords in prompt
        prompt = sample.get("prompt", "").lower()
        code_keywords = [
            "write a function",
            "implement",
            "code",
            "program",
            "algorithm",
        ]
        if any(kw in prompt for kw in code_keywords):
            return True

        return False


# =============================================================================
# MATH REASONING
# =============================================================================

@register_type("math")
class MathReasoningTypeHandler(BaseDataTypeHandler):
    """Handler for mathematical reasoning tasks.

    Optimized for:
    - Step-by-step reasoning
    - Numerical accuracy
    - Formula validation
    """

    type_name = "math"

    def get_reward_config(self) -> RewardConfig:
        """Use math-specific rewards."""
        return RewardConfig(
            functions=(
                "r1_correctness",
                "r1_boxed_match",
                "r1_thinking_quality",
            ),
            weights=(0.5, 0.3, 0.2),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Medium-length generation with thinking."""
        return GenerationStrategy(
            max_length=768,
            temperature=0.8,
            top_p=0.95,
            two_phase=True,
            enforce_thinking=True,
            continuation_tokens=256,
        )

    def get_curriculum_config(self) -> CurriculumConfig:
        """Enable curriculum for math."""
        return CurriculumConfig(
            enabled=True,
            start_ratio=0.8,
            end_ratio=0.0,
            warmup_steps=100,
            strategy="linear",
        )

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Check if sample is math reasoning."""
        # Check explicit type
        if sample.get("type") == "math":
            return True

        # Check for boxed answer
        answer = sample.get("answer", "")
        if "\\boxed{" in answer or r"\boxed{" in answer:
            return True

        # Check for math keywords
        prompt = sample.get("prompt", "").lower()
        math_keywords = [
            "calculate",
            "solve",
            "equation",
            "formula",
            "mathematics",
            "prove",
            "factorial",
        ]
        if any(kw in prompt for kw in math_keywords):
            return True

        return False


# =============================================================================
# MCQ (Multiple Choice Questions)
# =============================================================================

@register_type("mcq")
class MCQTypeHandler(BaseDataTypeHandler):
    """Handler for multiple choice questions."""

    type_name = "mcq"

    def get_reward_config(self) -> RewardConfig:
        """MCQ-specific rewards."""
        return RewardConfig(
            functions=(
                "r1_mcq_exact",
                "r1_thinking_quality",
            ),
            weights=(0.7, 0.3),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Short generation for MCQ."""
        return GenerationStrategy(
            max_length=512,
            temperature=0.8,
            two_phase=True,
            enforce_thinking=True,
        )

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Check if sample is MCQ."""
        # Check for possible_boxed_answers (common in MCQ datasets)
        if "possible_boxed_answers" in sample:
            return True

        # Check answer is single letter
        answer = sample.get("answer", "")
        if re.match(r'^[A-D]$', answer.strip()):
            return True

        return False


# =============================================================================
# EXAM-STYLE (AIME, etc.)
# =============================================================================

@register_type("exam")
class ExamTypeHandler(BaseDataTypeHandler):
    """Handler for exam-style problems (AIME, etc.)."""

    type_name = "exam"

    def get_reward_config(self) -> RewardConfig:
        """Exam-specific rewards."""
        return RewardConfig(
            functions=(
                "exam_correctness",
                "exam_format",
                "exam_thinking",
            ),
            weights=(0.5, 0.25, 0.25),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Long generation with thinking."""
        return GenerationStrategy(
            max_length=1536,
            temperature=0.85,
            two_phase=True,
            enforce_thinking=True,
            continuation_tokens=384,
        )

    def get_curriculum_config(self) -> CurriculumConfig:
        """Strong curriculum for exams."""
        return CurriculumConfig(
            enabled=True,
            start_ratio=1.0,
            end_ratio=0.0,
            warmup_steps=100,
            strategy="cosine",
        )

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Check if sample is exam-style."""
        sample_type = sample.get("type", "")
        if sample_type.startswith("exam_") or sample_type in ("aime", "math500"):
            return True

        # Check source
        source = sample.get("source", "").lower()
        if any(x in source for x in ["aime", "exam", "olympiad"]):
            return True

        return False


# =============================================================================
# DEFAULT FALLBACK
# =============================================================================

@register_type("default")
class DefaultTypeHandler(BaseDataTypeHandler):
    """Default handler for unspecified types.

    Uses conservative, general-purpose settings.
    """

    type_name = "default"

    def get_reward_config(self) -> RewardConfig:
        """Basic correctness reward."""
        return RewardConfig(
            functions=("r1_correctness",),
            weights=(1.0,),
        )

    def get_generation_strategy(self) -> GenerationStrategy:
        """Standard generation."""
        return GenerationStrategy(
            max_length=512,
            temperature=0.8,
            two_phase=False,
            enforce_thinking=False,
        )

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Accept all samples (fallback)."""
        return True


# =============================================================================
# COMPOSITE HANDLER (Advanced)
# =============================================================================

class CompositeTypeHandler(BaseDataTypeHandler):
    """Compose multiple handlers together.

    Example:
        handler = CompositeTypeHandler(
            handlers=[
                ToolCallingTypeHandler(),
                ThinkingTypeHandler(),
            ],
            # Use first matching handler
        )
    """

    type_name = "composite"

    def __init__(self, handlers: list[BaseDataTypeHandler]):
        self.handlers = handlers

    def get_reward_config(self) -> RewardConfig:
        """Combine reward configs from all handlers."""
        from .registry import compose_rewards
        return compose_rewards(*[h.get_reward_config() for h in self.handlers])

    def get_generation_strategy(self) -> GenerationStrategy:
        """Use first handler's strategy."""
        return self.handlers[0].get_generation_strategy()

    def get_curriculum_config(self) -> CurriculumConfig:
        """Use first handler's curriculum."""
        return self.handlers[0].get_curriculum_config()

    def validate_sample(self, sample: dict[str, Any]) -> bool:
        """Sample valid if any handler accepts it."""
        return any(h.validate_sample(sample) for h in self.handlers)

    def preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply all preprocessors in sequence."""
        for handler in self.handlers:
            sample = handler.preprocess_sample(sample)
        return sample
