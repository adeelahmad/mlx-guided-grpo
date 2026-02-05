#!/usr/bin/env python3
"""
Complete Type System Example
=============================

Shows how to define a COMPLETE custom type in ~20 lines of code:
- Reward function
- Generation strategy
- Curriculum learning
- Phase recovery
- Prompt templates

Everything auto-discovered and ready to use!
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# METHOD 1: Individual Classes (Flexible)
# =============================================================================

print("\n" + "="*70)
print("METHOD 1: Define Individual Classes")
print("="*70)

from mlx_grpo.trainer.type_system.auto_discovery import (
    BaseReward,
    BaseGenerationStrategy,
)
from mlx_grpo.trainer.type_system.auto_discovery_extended import (
    BaseCurriculum,
    BasePhaseRecovery,
)


# Define reward (5 lines!)
class CodeReward(BaseReward):
    """Auto-discovered for type='code'"""

    def compute(self, prompts, completions, answers, types=None):
        # Simple: check if code runs
        scores = []
        for comp in completions:
            try:
                compile(comp, '<string>', 'exec')
                scores.append(1.0)  # Compiles!
            except:
                scores.append(0.0)  # Syntax error
        return scores


# Define generation strategy (4 lines!)
class CodeGenerationStrategy(BaseGenerationStrategy):
    """Auto-discovered for type='code'"""

    def get_max_length(self):
        return 2048  # Code can be long

    def get_temperature(self):
        return 0.7  # Less random for code


# Define curriculum (6 lines!)
class CodeCurriculum(BaseCurriculum):
    """Auto-discovered for type='code'"""

    def apply_scaffolding(self, prompt, answer, ratio=0.5):
        if ratio <= 0:
            return prompt
        # Provide function signature as hint
        lines = answer.split('\n')
        hint = lines[0] if lines else ""  # First line (def ...)
        return f"{prompt}\n\nHint: Start with: {hint}"


# Define phase recovery (5 lines!)
class CodePhaseRecovery(BasePhaseRecovery):
    """Auto-discovered for type='code'"""

    def is_incomplete(self, output: str) -> bool:
        # Code incomplete if missing closing braces
        return output.count('{') > output.count('}')

    def create_continuation_prompt(self, prompt, output):
        return f"{output}\n# Complete the code:"


print("\n✓ Defined CodeReward (5 lines)")
print("✓ Defined CodeGenerationStrategy (4 lines)")
print("✓ Defined CodeCurriculum (6 lines)")
print("✓ Defined CodePhaseRecovery (5 lines)")
print(f"\nTotal: ~20 lines for COMPLETE type support!")


# =============================================================================
# METHOD 2: All-in-One Integration (Simplest!)
# =============================================================================

print("\n" + "="*70)
print("METHOD 2: All-in-One Integration Class")
print("="*70)

from mlx_grpo.trainer.type_system.auto_discovery_extended import TypeIntegration


class SummarizationIntegration(TypeIntegration):
    """Complete type definition in ONE class!

    Auto-generates:
    - SummarizationReward
    - SummarizationGenerationStrategy
    - SummarizationCurriculum
    - SummarizationPhaseRecovery
    """

    type_name = "summarization"

    # Reward
    def compute_reward(self, prompts, completions, answers, types=None):
        return [
            1.0 if len(c) <= len(a) * 1.5 else 0.5
            for c, a in zip(completions, answers)
        ]

    # Generation
    def get_max_length(self):
        return 256

    def get_temperature(self):
        return 0.6

    # Curriculum
    def apply_scaffolding(self, prompt, answer, ratio=0.5):
        if ratio > 0:
            words = answer.split()
            hint_words = words[:int(len(words) * ratio)]
            return f"{prompt}\n\nStart: {' '.join(hint_words)}..."
        return prompt

    # Phase recovery
    def create_continuation_prompt(self, prompt, output):
        return f"{output}\nConclude the summary:"


print("\n✓ Defined complete SummarizationIntegration in ONE class")
print("  Includes: reward, generation, curriculum, phase recovery")
print(f"\nTotal: ~15 lines for EVERYTHING!")


# =============================================================================
# DEMO: Using the Types
# =============================================================================

print("\n" + "="*70)
print("DEMO: Auto-Discovery in Action")
print("="*70)

from mlx_grpo.trainer.type_system.auto_discovery_extended import get_all_for_type

# Get all components for 'math' type
print("\n1. Getting components for type='math'...")
math_components = get_all_for_type("math")

print(f"   ✓ Reward: {math_components['reward'].__class__.__name__}")
print(f"   ✓ Generation: {math_components['generation'].__class__.__name__}")
print(f"   ✓ Curriculum: {math_components['curriculum'].__class__.__name__}")
print(f"   ✓ Phase Recovery: {math_components['phase_recovery'].__class__.__name__}")

# Use them!
print("\n2. Using math components...")

# Reward
scores = math_components['reward'].compute(
    prompts=["What is 2+2?"],
    completions=["\\boxed{4}"],
    answers=["\\boxed{4}"]
)
print(f"   Reward score: {scores[0]}")

# Generation config
strategy = math_components['generation']
print(f"   Max length: {strategy.get_max_length()}")
print(f"   Two-phase: {strategy.use_two_phase()}")

# Curriculum
curriculum = math_components['curriculum']
scaffolded = curriculum.apply_scaffolding(
    prompt="Solve: 5! = ?",
    answer="<think>5! = 5 × 4 × 3 × 2 × 1 = 120</think>\\boxed{120}",
    ratio=0.5
)
print(f"   Scaffolding applied: {len(scaffolded)} chars")

# Phase recovery
recovery = math_components['phase_recovery']
incomplete = "<think>5! = 5 × 4 × 3"
if recovery.is_incomplete(incomplete):
    continuation = recovery.create_continuation_prompt("Solve 5!", incomplete)
    print(f"   Recovery prompt created: {len(continuation)} chars")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: How Simple Is It?")
print("="*70)

summary = """
To add a new type, you have TWO options:

OPTION 1: Individual Classes (flexible)
----------------------------------------
1. Create {Type}Reward - 5 lines
2. Create {Type}GenerationStrategy - 4 lines
3. Create {Type}Curriculum - 6 lines (optional)
4. Create {Type}PhaseRecovery - 5 lines (optional)

Total: ~20 lines for complete type support!

OPTION 2: Integration Class (simplest)
---------------------------------------
1. Create {Type}Integration - 15 lines
2. That's it!

Auto-generates all components for you.

BOTH OPTIONS:
- Drop files in mlx_grpo/trainer/type_system/{rewards,generation,curriculum,phase_recovery}/
- NO registration needed
- NO imports needed
- NO configuration needed
- Just use type='{type}' in your data!

POWER:
- Works with ANY dataset
- Works with agentic apps
- Works with custom applications
- Fully extensible
- Convention over configuration
- Dead simple API

EXAMPLE USAGE:
--------------
# In your data:
{"prompt": "...", "answer": "...", "type": "my_custom_type"}

# System automatically:
1. Discovers MyCustomTypeReward
2. Discovers MyCustomTypeGenerationStrategy
3. Discovers MyCustomTypeCurriculum
4. Discovers MyCustomTypePhaseRecovery
5. Applies optimal configuration
6. Trains with type-specific settings

NO MANUAL CONFIGURATION REQUIRED!
"""

print(summary)

print("="*70)
print("\nThe power of convention over configuration!")
print("The elegance of metaprogramming!")
print("The simplicity of Python!")
print("="*70 + "\n")
