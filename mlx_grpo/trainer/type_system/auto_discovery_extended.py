"""
Extended Auto-Discovery: Curriculum & Phase Recovery
====================================================

Extends the auto-discovery system with:
- Curriculum templates (per type)
- 2nd phase recovery strategies (for incomplete thinking)
- Prompt templates
- Integration helpers

Pattern:
    type="math" → discovers:
    - MathCurriculum (scaffolding strategy)
    - MathPhaseRecovery (thinking continuation)
    - MathPromptTemplate (prompt formatting)

Usage:
    # Get curriculum for math
    curriculum = get_curriculum_for_type("math")
    scaffolded_prompt = curriculum.apply_scaffolding(prompt, ratio=0.5)

    # Get phase recovery for incomplete thinking
    recovery = get_phase_recovery_for_type("math")
    continuation = recovery.create_continuation(incomplete_output)
"""

from __future__ import annotations

from typing import Any, Optional
from .auto_discovery import (
    discover_class,
    BaseReward,
    BaseGenerationStrategy,
    BaseDataLoader,
    DiscoveryRegistry,
)

__all__ = [
    "BaseCurriculum",
    "BasePhaseRecovery",
    "BasePromptTemplate",
    "get_curriculum_for_type",
    "get_phase_recovery_for_type",
    "get_prompt_template_for_type",
    "get_all_for_type",
]


# =============================================================================
# BASE CLASSES
# =============================================================================

class BaseCurriculum:
    """Base curriculum learning strategy.

    Handles scaffolding (partial hints) for curriculum learning.

    Naming convention:
        - Math type: MathCurriculum
        - Code type: CodeCurriculum
    """

    def __init__(self):
        self.type_name = self.__class__.__name__.replace("Curriculum", "").lower()

    def apply_scaffolding(
        self,
        prompt: str,
        answer: str,
        ratio: float = 0.5
    ) -> str:
        """Apply scaffolding to prompt.

        Args:
            prompt: Original prompt
            answer: Ground truth answer
            ratio: How much of answer to provide (0.0-1.0)

        Returns:
            Prompt with scaffolding added
        """
        if ratio <= 0.0:
            return prompt

        if ratio >= 1.0:
            # Full scaffolding - include complete answer
            return f"{prompt}\n\nHint: {answer}"

        # Partial scaffolding - include portion of answer
        portion_length = int(len(answer) * ratio)
        partial = answer[:portion_length]
        return f"{prompt}\n\nHint: {partial}..."

    def get_start_ratio(self) -> float:
        """Starting scaffolding ratio."""
        return 1.0

    def get_end_ratio(self) -> float:
        """Ending scaffolding ratio."""
        return 0.0

    def get_warmup_steps(self) -> int:
        """Steps before curriculum starts."""
        return 0

    def get_strategy(self) -> str:
        """Curriculum decay strategy: 'linear', 'exponential', 'cosine'."""
        return "linear"


class BasePhaseRecovery:
    """Base 2-phase recovery strategy.

    Handles incomplete thinking outputs (missing </think> tags).

    Naming convention:
        - Math type: MathPhaseRecovery
        - Code type: CodePhaseRecovery
    """

    def __init__(self):
        self.type_name = self.__class__.__name__.replace("PhaseRecovery", "").lower()

    def is_incomplete(self, output: str) -> bool:
        """Check if thinking phase is incomplete.

        Default: checks for <think> without </think>
        """
        return "<think>" in output and "</think>" not in output

    def create_continuation_prompt(
        self,
        original_prompt: str,
        incomplete_output: str
    ) -> str:
        """Create prompt for phase 2 continuation.

        Args:
            original_prompt: Original prompt
            incomplete_output: Incomplete generation from phase 1

        Returns:
            Continuation prompt
        """
        # Default: close thinking, ask for answer
        return (
            f"{incomplete_output}\n"
            "</think>\n\n"
            "Now provide your final answer:"
        )

    def get_continuation_tokens(self) -> int:
        """Max tokens for phase 2."""
        return 256

    def should_use_two_phase(self) -> bool:
        """Whether this type benefits from 2-phase generation."""
        return False


class BasePromptTemplate:
    """Base prompt template system.

    Handles formatting prompts for specific task types.

    Naming convention:
        - Math type: MathPromptTemplate
        - Tool type: ToolPromptTemplate
    """

    def __init__(self):
        self.type_name = self.__class__.__name__.replace("PromptTemplate", "").lower()

    def format_prompt(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """Format a prompt for this type.

        Args:
            task: The task description
            context: Optional context (examples, functions, etc.)

        Returns:
            Formatted prompt
        """
        return task

    def get_system_message(self) -> str:
        """Get system message for this type."""
        return "You are a helpful AI assistant."

    def format_examples(self, examples: list[dict[str, str]]) -> str:
        """Format few-shot examples.

        Args:
            examples: List of {'prompt': ..., 'answer': ...}

        Returns:
            Formatted examples string
        """
        if not examples:
            return ""

        parts = ["Here are some examples:\n"]
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Q: {ex['prompt']}")
            parts.append(f"A: {ex['answer']}\n")

        return "\n".join(parts)


# =============================================================================
# DISCOVERY FUNCTIONS
# =============================================================================

# Register discovery paths for new component types
DiscoveryRegistry.add_path("curriculum", "mlx_grpo.trainer.type_system.curriculum")
DiscoveryRegistry.add_path("phase_recovery", "mlx_grpo.trainer.type_system.phase_recovery")
DiscoveryRegistry.add_path("prompt_template", "mlx_grpo.trainer.type_system.prompts")


def get_curriculum_for_type(data_type: str) -> BaseCurriculum:
    """Get curriculum strategy for data type.

    Args:
        data_type: Data type (e.g., "math", "tool_call")

    Returns:
        Curriculum instance

    Example:
        curriculum = get_curriculum_for_type("math")
        scaffolded = curriculum.apply_scaffolding(prompt, answer, ratio=0.5)
    """
    CurriculumClass = discover_class(
        data_type=data_type,
        component_type="curriculum",
        base_class=BaseCurriculum,
        suffix="Curriculum",
    )
    return CurriculumClass()


def get_phase_recovery_for_type(data_type: str) -> BasePhaseRecovery:
    """Get phase recovery strategy for data type.

    Args:
        data_type: Data type

    Returns:
        PhaseRecovery instance

    Example:
        recovery = get_phase_recovery_for_type("math")
        if recovery.is_incomplete(output):
            continuation = recovery.create_continuation_prompt(prompt, output)
    """
    RecoveryClass = discover_class(
        data_type=data_type,
        component_type="phase_recovery",
        base_class=BasePhaseRecovery,
        suffix="PhaseRecovery",
    )
    return RecoveryClass()


def get_prompt_template_for_type(data_type: str) -> BasePromptTemplate:
    """Get prompt template for data type.

    Args:
        data_type: Data type

    Returns:
        PromptTemplate instance

    Example:
        template = get_prompt_template_for_type("tool_call")
        prompt = template.format_prompt(task, context={'functions': [...]})
    """
    TemplateClass = discover_class(
        data_type=data_type,
        component_type="prompt_template",
        base_class=BasePromptTemplate,
        suffix="PromptTemplate",
    )
    return TemplateClass()


def get_all_for_type(data_type: str) -> dict[str, Any]:
    """Get ALL components for a data type.

    One-stop shop for complete type configuration.

    Args:
        data_type: Data type

    Returns:
        Dict with all components:
        - reward: Reward instance
        - generation: GenerationStrategy instance
        - loader: DataLoader instance
        - curriculum: Curriculum instance
        - phase_recovery: PhaseRecovery instance
        - prompt_template: PromptTemplate instance

    Example:
        components = get_all_for_type("math")

        # Use reward
        scores = components['reward'].compute(prompts, completions, answers)

        # Use generation config
        max_len = components['generation'].get_max_length()

        # Use curriculum
        if training_iter < warmup:
            ratio = components['curriculum'].get_start_ratio()
            scaffolded = components['curriculum'].apply_scaffolding(
                prompt, answer, ratio
            )

        # Use phase recovery
        if components['phase_recovery'].is_incomplete(output):
            continuation = components['phase_recovery'].create_continuation_prompt(
                prompt, output
            )
    """
    from .auto_discovery import (
        get_reward_for_type,
        get_generation_strategy_for_type,
        get_data_loader_for_type,
    )

    return {
        'reward': get_reward_for_type(data_type),
        'generation': get_generation_strategy_for_type(data_type),
        'loader': get_data_loader_for_type(data_type),
        'curriculum': get_curriculum_for_type(data_type),
        'phase_recovery': get_phase_recovery_for_type(data_type),
        'prompt_template': get_prompt_template_for_type(data_type),
    }


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

class TypeIntegration:
    """Complete integration for a custom type.

    Simplest way to add a new type - define everything in one class!

    Example:
        class MyTypeIntegration(TypeIntegration):
            type_name = "my_type"

            def compute_reward(self, prompts, completions, answers, types=None):
                return [1.0] * len(completions)

            def get_max_length(self):
                return 768

            def apply_scaffolding(self, prompt, answer, ratio=0.5):
                return f"{prompt}\\n\\nHint: {answer[:int(len(answer)*ratio)]}"

        # That's it! Auto-discovered for type='my_type'
    """

    type_name: str = "base"

    # Reward methods
    def compute_reward(
        self,
        prompts: list[str],
        completions: list[str],
        answers: list[str],
        types: Optional[list[str]] = None,
    ) -> list[float]:
        """Compute rewards (required)."""
        return [1.0] * len(completions)

    # Generation methods
    def get_max_length(self) -> int:
        return 512

    def get_temperature(self) -> float:
        return 0.8

    def use_two_phase(self) -> bool:
        return False

    # Curriculum methods
    def apply_scaffolding(self, prompt: str, answer: str, ratio: float) -> str:
        return prompt

    # Phase recovery methods
    def is_incomplete(self, output: str) -> bool:
        return "<think>" in output and "</think>" not in output

    def create_continuation_prompt(self, prompt: str, output: str) -> str:
        return f"{output}\n</think>\n\nFinal answer:"

    # Prompt template methods
    def format_prompt(self, task: str, context: Optional[dict] = None) -> str:
        return task


def create_type_from_integration(integration: TypeIntegration):
    """Create all component classes from an integration.

    This is a factory that generates Reward, Strategy, Curriculum, etc.
    from a single integration class.

    Args:
        integration: TypeIntegration instance

    Returns:
        Dict of generated classes
    """
    type_name = integration.type_name
    class_prefix = "".join(word.capitalize() for word in type_name.split("_"))

    # Create reward class
    class DynamicReward(BaseReward):
        def compute(self, prompts, completions, answers, types=None):
            return integration.compute_reward(prompts, completions, answers, types)

    DynamicReward.__name__ = f"{class_prefix}Reward"

    # Create generation strategy
    class DynamicStrategy(BaseGenerationStrategy):
        def get_max_length(self):
            return integration.get_max_length()

        def get_temperature(self):
            return integration.get_temperature()

        def use_two_phase(self):
            return integration.use_two_phase()

    DynamicStrategy.__name__ = f"{class_prefix}GenerationStrategy"

    # Create curriculum
    class DynamicCurriculum(BaseCurriculum):
        def apply_scaffolding(self, prompt, answer, ratio=0.5):
            return integration.apply_scaffolding(prompt, answer, ratio)

    DynamicCurriculum.__name__ = f"{class_prefix}Curriculum"

    # Create phase recovery
    class DynamicRecovery(BasePhaseRecovery):
        def is_incomplete(self, output):
            return integration.is_incomplete(output)

        def create_continuation_prompt(self, prompt, output):
            return integration.create_continuation_prompt(prompt, output)

    DynamicRecovery.__name__ = f"{class_prefix}PhaseRecovery"

    return {
        'reward': DynamicReward,
        'generation': DynamicStrategy,
        'curriculum': DynamicCurriculum,
        'phase_recovery': DynamicRecovery,
    }
