#!/usr/bin/env python3
"""
Demo: Extensible Type System for GRPO
======================================

This demonstrates the powerful, elegant type system for MLX-GRPO.

Features:
- Auto-detection of dataset types
- Type-specific reward configurations
- Type-specific generation strategies
- Easy extension with new types
- Mixed-type dataset support

Run:
    python examples/demo_type_system.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_grpo.trainer.type_system import (
    register_type,
    get_type_handler,
    list_types,
    detect_dataset_type,
    BaseDataTypeHandler,
    RewardConfig,
    GenerationStrategy,
)


def demo_basic_usage():
    """Demo 1: Basic usage with built-in types."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Usage")
    print("="*70)

    # List available types
    print(f"\nRegistered types: {', '.join(list_types())}")

    # Get handler for tool calling
    handler = get_type_handler("tool_call")

    print(f"\nType: {handler.type_name}")
    print(f"Rewards: {handler.get_reward_config().functions}")
    print(f"Max Length: {handler.get_generation_strategy().max_length}")
    print(f"Two-Phase: {handler.get_generation_strategy().two_phase}")


def demo_auto_detection():
    """Demo 2: Auto-detection of sample types."""
    print("\n" + "="*70)
    print("DEMO 2: Auto-Detection")
    print("="*70)

    # Test samples
    samples = [
        {
            "prompt": "What is 2+2?",
            "answer": "<think>2+2=4</think>4",
        },
        {
            "prompt": "Calculate factorial",
            "answer": "calculate_factorial(n=5)",
        },
        {
            "prompt": "Write a function to reverse a string",
            "answer": "def reverse(s):\n    return s[::-1]",
        },
        {
            "prompt": "What is \\sqrt{16}?",
            "answer": "\\boxed{4}",
        },
    ]

    expected = ["thinking", "tool_call", "code", "math"]

    print("\nAuto-detecting sample types:\n")
    for i, (sample, exp) in enumerate(zip(samples, expected), 1):
        detected = detect_dataset_type(sample)
        match = "✓" if detected == exp else "✗"
        print(f"{match} Sample {i}: {detected:15s} (expected: {exp})")


def demo_custom_type():
    """Demo 3: Register custom type handler."""
    print("\n" + "="*70)
    print("DEMO 3: Custom Type Handler")
    print("="*70)

    # Define custom handler
    @register_type("summarization")
    class SummarizationHandler(BaseDataTypeHandler):
        """Handler for text summarization tasks."""

        type_name = "summarization"

        def get_reward_config(self) -> RewardConfig:
            return RewardConfig(
                functions=("r1_correctness", "r1_conciseness"),
                weights=(0.7, 0.3),
            )

        def get_generation_strategy(self) -> GenerationStrategy:
            return GenerationStrategy(
                max_length=256,
                temperature=0.7,
                two_phase=False,
            )

        def validate_sample(self, sample):
            prompt = sample.get("prompt", "").lower()
            return any(kw in prompt for kw in ["summarize", "summary", "tldr"])

    print("\n✓ Registered custom type: 'summarization'")

    # Use it
    handler = get_type_handler("summarization")
    print(f"\nRewards: {handler.get_reward_config().functions}")
    print(f"Weights: {handler.get_reward_config().normalized_weights}")

    # Test validation
    sample = {"prompt": "Summarize this article", "answer": "Short summary."}
    is_valid = handler.validate_sample(sample)
    print(f"\nValidation test: {is_valid}")


def demo_mixed_dataset():
    """Demo 4: Mixed-type dataset handling."""
    print("\n" + "="*70)
    print("DEMO 4: Mixed-Type Dataset")
    print("="*70)

    # Simulate mixed dataset
    samples = [
        {"type": "tool_call", "prompt": "...", "answer": "func()"},
        {"type": "tool_call", "prompt": "...", "answer": "func()"},
        {"type": "tool_call", "prompt": "...", "answer": "func()"},
        {"type": "math", "prompt": "...", "answer": "\\boxed{42}"},
        {"type": "math", "prompt": "...", "answer": "\\boxed{7}"},
        {"type": "thinking", "prompt": "...", "answer": "<think>...</think>x"},
    ]

    # Count types
    from collections import Counter
    type_counts = Counter(s["type"] for s in samples)

    print(f"\nDataset composition:")
    for type_name, count in type_counts.items():
        pct = 100 * count / len(samples)
        print(f"  {type_name:15s} {count:3d} ({pct:5.1f}%)")

    # Get configs for each type
    print(f"\nType-specific configurations:")
    for type_name in type_counts.keys():
        handler = get_type_handler(type_name)
        config = handler.get_reward_config()
        print(f"\n  {type_name}:")
        print(f"    Rewards: {', '.join(config.functions)}")


def demo_composition():
    """Demo 5: Composing multiple handlers."""
    print("\n" + "="*70)
    print("DEMO 5: Configuration Composition")
    print("="*70)

    from mlx_grpo.trainer.type_system.registry import compose_rewards

    # Get configs
    tool_config = get_type_handler("tool_call").get_reward_config()
    think_config = get_type_handler("thinking").get_reward_config()

    print(f"\nTool calling rewards: {tool_config.functions}")
    print(f"Thinking rewards: {think_config.functions}")

    # Compose
    combined = compose_rewards(tool_config, think_config)

    print(f"\nCombined rewards: {combined.functions}")
    print(f"Weights: {[f'{w:.3f}' for w in combined.normalized_weights]}")


def demo_real_world():
    """Demo 6: Real-world usage pattern."""
    print("\n" + "="*70)
    print("DEMO 6: Real-World Usage")
    print("="*70)

    # Simulate loading a dataset
    print("\n1. Loading dataset...")

    sample = {
        "prompt": "You are a helpful assistant with access to functions...",
        "answer": "calculate_factorial(n=5)",
        "type": "tool_call",
    }

    # Auto-detect
    detected_type = detect_dataset_type(sample)
    print(f"   Detected type: {detected_type}")

    # Get handler
    handler = get_type_handler(detected_type)

    # Get configuration
    print(f"\n2. Applying type-based configuration...")

    reward_config = handler.get_reward_config()
    gen_config = handler.get_generation_strategy()

    print(f"   Rewards: {reward_config.functions}")
    print(f"   Weights: {[f'{w:.2f}' for w in reward_config.normalized_weights]}")
    print(f"   Max length: {gen_config.max_length}")
    print(f"   Temperature: {gen_config.temperature}")
    print(f"   Two-phase: {gen_config.two_phase}")

    # Preprocess
    print(f"\n3. Preprocessing sample...")
    processed = handler.preprocess_sample(sample)
    print(f"   Type field: {processed.get('type')}")

    print(f"\n✓ Ready for training!")


def demo_extensibility():
    """Demo 7: Advanced extensibility."""
    print("\n" + "="*70)
    print("DEMO 7: Advanced Extensibility")
    print("="*70)

    # Dynamic registration
    class TranslationHandler(BaseDataTypeHandler):
        type_name = "translation"

        def get_reward_config(self):
            return RewardConfig(
                functions=("r1_correctness", "r1_fluency"),
                weights=(0.6, 0.4),
            )

    # Register without decorator
    from mlx_grpo.trainer.type_system.registry import TypeRegistry
    TypeRegistry.register()(TranslationHandler)

    print("✓ Dynamically registered 'translation' type")
    print(f"\nAll types: {', '.join(list_types())}")

    # Context manager for temporary override
    print("\n✓ Type system supports:")
    print("  - Decorator-based registration")
    print("  - Dynamic registration")
    print("  - Metaclass validation")
    print("  - Protocol-based interfaces")
    print("  - Composition patterns")
    print("  - Context managers")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("MLX-GRPO EXTENSIBLE TYPE SYSTEM DEMO")
    print("="*70)

    demos = [
        demo_basic_usage,
        demo_auto_detection,
        demo_custom_type,
        demo_mixed_dataset,
        demo_composition,
        demo_real_world,
        demo_extensibility,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Auto-detection makes dataset handling seamless")
    print("  2. Type-specific configs optimize training per task")
    print("  3. Easy to extend with custom types")
    print("  4. Mixed-type datasets work out of the box")
    print("  5. Clean, elegant API using advanced Python")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
