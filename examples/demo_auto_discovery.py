#!/usr/bin/env python3
"""
Demo: Auto-Discovery Type System
=================================

Shows the incredibly simple and elegant auto-discovery system.

Key Principle:
    type="math" → automatically finds MathReward, MathGenerationStrategy

Just extend base class and implement ONE method. That's it!
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_auto_discovery():
    """Demo 1: Auto-discovery in action."""
    print("\n" + "="*70)
    print("DEMO 1: Auto-Discovery")
    print("="*70)

    from mlx_grpo.trainer.type_system.auto_discovery import (
        get_reward_for_type,
        get_generation_strategy_for_type,
    )

    # Just specify the type - components are auto-discovered!
    print("\n1. Getting components for type='math'...")

    reward = get_reward_for_type("math")
    strategy = get_generation_strategy_for_type("math")

    print(f"   ✓ Found: {reward.__class__.__name__}")
    print(f"   ✓ Found: {strategy.__class__.__name__}")

    # Use them
    print("\n2. Using the components...")

    prompts = ["What is 2+2?"]
    completions = ["<think>2+2=4</think>\\boxed{4}"]
    answers = ["\\boxed{4}"]

    scores = reward.compute(prompts, completions, answers)
    print(f"   Reward score: {scores[0]}")

    print(f"   Max length: {strategy.get_max_length()}")
    print(f"   Temperature: {strategy.get_temperature()}")
    print(f"   Two-phase: {strategy.use_two_phase()}")


def demo_create_custom_type():
    """Demo 2: Creating a custom type - super simple!"""
    print("\n" + "="*70)
    print("DEMO 2: Create Custom Type (5 lines of code!)")
    print("="*70)

    from mlx_grpo.trainer.type_system.auto_discovery import (
        BaseReward,
        BaseGenerationStrategy,
        get_reward_for_type,
        get_generation_strategy_for_type,
    )

    # Step 1: Create a reward - just extend and implement ONE method!
    print("\n1. Creating SummarizationReward...")

    class SummarizationReward(BaseReward):
        """Reward for summarization tasks.

        That's it! Just implement compute(). Auto-discovered for type='summarization'.
        """
        def compute(self, prompts, completions, answers, types=None):
            # Simple length-based reward
            return [
                1.0 if len(c) <= len(a) * 1.5 else 0.5
                for c, a in zip(completions, answers)
            ]

    print("   ✓ Created (5 lines!)")

    # Step 2: Create a strategy - override only what you need!
    print("\n2. Creating SummarizationGenerationStrategy...")

    class SummarizationGenerationStrategy(BaseGenerationStrategy):
        """Strategy for summarization.

        Only override what's different from defaults!
        """
        def get_max_length(self):
            return 256  # Summaries are short

        def get_temperature(self):
            return 0.7  # Less creative

    print("   ✓ Created (3 lines!)")

    # Step 3: Register discovery paths
    print("\n3. Registering discovery paths...")

    from mlx_grpo.trainer.type_system.auto_discovery import register_discovery_path

    # Tell the system where to look (only needed if not in default paths)
    # In real usage, you'd put these in proper modules and skip this step
    import sys
    current_module = sys.modules[__name__]
    current_module.SummarizationReward = SummarizationReward
    current_module.SummarizationGenerationStrategy = SummarizationGenerationStrategy

    print("   ✓ Registered")

    # Step 4: Use it! (falls back to base classes for now since we added to this module)
    print("\n4. Using for type='summarization'...")
    print("   (Using base classes as fallback since we defined in __main__)")

    reward = get_reward_for_type("summarization")
    strategy = get_generation_strategy_for_type("summarization")

    print(f"   ✓ Got: {reward.__class__.__name__}")
    print(f"   ✓ Got: {strategy.__class__.__name__}")


def demo_tool_calling():
    """Demo 3: Tool calling type."""
    print("\n" + "="*70)
    print("DEMO 3: Tool Calling Type")
    print("="*70)

    from mlx_grpo.trainer.type_system.auto_discovery import (
        get_reward_for_type,
        get_generation_strategy_for_type,
    )

    print("\n1. Auto-discovering for type='tool_call'...")

    reward = get_reward_for_type("tool_call")
    strategy = get_generation_strategy_for_type("tool_call")

    print(f"   ✓ Found: {reward.__class__.__name__}")
    print(f"   ✓ Found: {strategy.__class__.__name__}")

    print("\n2. Tool calling specific settings:")
    print(f"   Max length: {strategy.get_max_length()} (short!)")
    print(f"   Temperature: {strategy.get_temperature()}")
    print(f"   Two-phase: {strategy.use_two_phase()} (direct)")

    print("\n3. Testing tool call reward...")

    prompts = ["Calculate factorial"]
    completions = ["calculate_factorial(n=5)"]
    answers = ["calculate_factorial(n=5)"]

    scores = reward.compute(prompts, completions, answers)
    print(f"   Score: {scores[0]:.2f}")


def demo_fallback():
    """Demo 4: Fallback to base classes."""
    print("\n" + "="*70)
    print("DEMO 4: Graceful Fallback")
    print("="*70)

    from mlx_grpo.trainer.type_system.auto_discovery import (
        get_reward_for_type,
        get_generation_strategy_for_type,
    )

    print("\n1. Requesting type that doesn't have specific classes...")

    # 'unknown' type doesn't have UnknownReward, etc.
    reward = get_reward_for_type("unknown")
    strategy = get_generation_strategy_for_type("unknown")

    print(f"   ✓ Gracefully fell back to: {reward.__class__.__name__}")
    print(f"   ✓ Gracefully fell back to: {strategy.__class__.__name__}")

    print("\n2. Base classes provide sensible defaults:")
    print(f"   Max length: {strategy.get_max_length()}")
    print(f"   Temperature: {strategy.get_temperature()}")


def demo_real_world_workflow():
    """Demo 5: Real-world workflow."""
    print("\n" + "="*70)
    print("DEMO 5: Real-World Workflow")
    print("="*70)

    from mlx_grpo.trainer.type_system.auto_discovery import discover_all_for_type

    print("\n1. Dataset has samples with type='math'...")

    sample = {
        "prompt": "What is 5!?",
        "answer": "\\boxed{120}",
        "type": "math"
    }

    print(f"   Sample type: {sample['type']}")

    print("\n2. Auto-discover ALL components for this type...")

    components = discover_all_for_type(sample['type'])

    print(f"   ✓ Reward: {components['reward'].__class__.__name__}")
    print(f"   ✓ Strategy: {components['generation'].__class__.__name__}")
    print(f"   ✓ Loader: {components['loader'].__class__.__name__}")

    print("\n3. Apply type-specific configuration...")

    reward = components['reward']
    strategy = components['generation']

    print(f"   Reward weight: {reward.get_weight()}")
    print(f"   Max tokens: {strategy.get_max_length()}")
    print(f"   Enforce thinking: {strategy.enforce_thinking()}")

    print("\n4. Train with optimal settings for this type!")
    print("   ✓ No manual configuration needed")
    print("   ✓ Each type gets its best settings")
    print("   ✓ Add new types by just creating {Type}Reward class")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("AUTO-DISCOVERY TYPE SYSTEM DEMO")
    print("="*70)
    print("\nKey Principles:")
    print("  1. Convention over configuration")
    print("  2. type='math' → MathReward, MathGenerationStrategy")
    print("  3. Just extend base class and implement ONE method")
    print("  4. Automatic discovery, graceful fallbacks")
    print("  5. Dead simple to extend")

    demos = [
        demo_auto_discovery,
        demo_create_custom_type,
        demo_tool_calling,
        demo_fallback,
        demo_real_world_workflow,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("SUMMARY: How to Add a New Type")
    print("="*70)
    print("""
1. Create a reward (optional):
    # In: mlx_grpo/trainer/type_system/rewards/my_type_reward.py

    from ..auto_discovery import BaseReward

    class MyTypeReward(BaseReward):
        def compute(self, prompts, completions, answers, types=None):
            # Your logic here
            return [1.0] * len(completions)

2. Create a strategy (optional):
    # In: mlx_grpo/trainer/type_system/generation/my_type_strategy.py

    from ..auto_discovery import BaseGenerationStrategy

    class MyTypeGenerationStrategy(BaseGenerationStrategy):
        def get_max_length(self):
            return 1024

3. That's it! Use with type='my_type' in your data:
    {"prompt": "...", "answer": "...", "type": "my_type"}

4. Everything auto-discovered and applied!
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
