"""
GRPO Integration Example
========================

Example showing how to integrate hierarchical_rewards_v3 with GRPO training.

This example assumes you have:
- MLX or similar framework for training
- A batch of prompts with multiple completions per prompt
- Expected answers for each prompt
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# Import the reward system
from hierarchical_rewards_v3 import (
    batch_hierarchical_reward,
    hierarchical_reward,
    RewardConfig,
    GateConfig,
    get_default_config,
)


def create_training_config() -> RewardConfig:
    """
    Create a reward configuration optimized for GRPO training.

    Adjustments from default:
    - Slightly lower thresholds for early training
    - Higher floors to ensure gradient flow
    """
    return RewardConfig(
        level_weights={
            "foundation": 0.10,
            "correctness": 0.45,
            "quality": 0.30,
            "polish": 0.15,
        },
        # Foundation: structure requirements
        foundation_gate=GateConfig(
            threshold=0.35,  # Slightly forgiving for 450-token limit
            floor=0.15,  # Always get some credit
            steepness=10.0,
        ),
        # Correctness: factual accuracy
        correctness_gate=GateConfig(threshold=0.10, floor=0.10, steepness=10.0),
        # Quality: reasoning depth
        quality_gate=GateConfig(threshold=0.08, floor=0.08, steepness=10.0),
        # Polish: style
        polish_gate=GateConfig(threshold=0.05, floor=0.05, steepness=10.0),
        min_score=0.05,
        max_score=1.0,
    )


def compute_grpo_rewards(
    prompts: List[str],
    completion_groups: List[List[str]],
    expected_answers: List[str],
    config: RewardConfig = None,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Compute rewards for GRPO training.

    Args:
        prompts: List of input prompts
        completion_groups: List of completion lists (one list per prompt)
        expected_answers: Expected answer for each prompt
        config: Reward configuration

    Returns:
        Tuple of (reward_groups, batch_stats)

    Example:
        prompts = ["What is 2+2?", "What is 3×4?"]
        completions = [
            ["<think>2+2=4</think><answer>4</answer>", "<think>adding</think><answer>5</answer>"],
            ["<think>3×4=12</think><answer>12</answer>", "<think>multiply</think><answer>11</answer>"]
        ]
        expected = ["4", "12"]

        rewards, stats = compute_grpo_rewards(prompts, completions, expected)
    """
    if config is None:
        config = create_training_config()

    all_rewards = []
    all_stats = []

    for prompt, completions, expected in zip(
        prompts, completion_groups, expected_answers
    ):
        result = batch_hierarchical_reward(
            responses=completions,
            expected=expected,
            question=prompt,
            config=config,
            ensure_ranking=True,
        )

        all_rewards.append(result.scores)
        all_stats.append(result.batch_stats)

    # Aggregate statistics
    batch_stats = {
        "mean_score": np.mean([s["mean"] for s in all_stats]),
        "mean_spread": np.mean([s["spread"] for s in all_stats]),
        "min_spread": min(s["spread"] for s in all_stats),
        "max_spread": max(s["spread"] for s in all_stats),
    }

    return all_rewards, batch_stats


def log_reward_diagnostics(
    response: str,
    expected: str,
    question: str,
    step: int = 0,
) -> Dict[str, Any]:
    """
    Get detailed diagnostics for logging during training.

    Returns structured data suitable for wandb/tensorboard logging.
    """
    score, diag = hierarchical_reward(response, expected, question)

    return {
        f"reward/final_score": score,
        f"reward/foundation_raw": diag["levels"]["foundation"]["raw_score"],
        f"reward/correctness_raw": diag["levels"]["correctness"]["raw_score"],
        f"reward/quality_raw": diag["levels"]["quality"]["raw_score"],
        f"reward/polish_raw": diag["levels"]["polish"]["raw_score"],
        f"reward/foundation_gate": diag["gates"]["foundation"],
        f"reward/correctness_gate": diag["gates"]["correctness"],
        f"reward/anti_gaming_penalty": diag["anti_gaming"]["penalty"],
        f"reward/has_anti_gaming_flags": len(diag["anti_gaming"]["flags"]) > 0,
    }


# ============================================
# Example Training Loop Integration
# ============================================


def example_grpo_step(
    model,
    batch: Dict[str, Any],
    config: RewardConfig = None,
) -> Dict[str, Any]:
    """
    Example GRPO training step with hierarchical rewards.

    This is a simplified example - adapt to your actual training setup.
    """
    prompts = batch["prompts"]
    expected_answers = batch["expected"]

    # Generate completions (group_size = 2 for GRPO)
    completions = []
    for prompt in prompts:
        # Your generation code here
        group = model.generate(prompt, n=2)  # Pseudo-code
        completions.append(group)

    # Compute rewards
    rewards, stats = compute_grpo_rewards(
        prompts=prompts,
        completion_groups=completions,
        expected_answers=expected_answers,
        config=config,
    )

    # Convert to training format
    # GRPO uses relative rankings, so we need advantages
    advantages = []
    for reward_group in rewards:
        # Center rewards for GRPO
        mean_reward = np.mean(reward_group)
        adv = [r - mean_reward for r in reward_group]
        advantages.append(adv)

    # Your GRPO loss computation here
    # loss = compute_grpo_loss(model, prompts, completions, advantages)

    return {
        "rewards": rewards,
        "advantages": advantages,
        "stats": stats,
    }


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Test data
    prompts = [
        "What is 6 × 7?",
        "What is the capital of France?",
    ]

    completions = [
        [
            "<think>\n6 × 7 = 42\nSimple multiplication.\n</think>\n<answer>42</answer>",
            "<think>\nLet me add 6 seven times: 6+6+6+6+6+6+6 = 42\n</think>\n<answer>42</answer>",
        ],
        [
            "<think>\nFrance's capital is Paris, the largest city.\n</think>\n<answer>Paris</answer>",
            "<think>\nI think it's Lyon.\n</think>\n<answer>Lyon</answer>",
        ],
    ]

    expected = ["42", "Paris"]

    # Compute rewards
    rewards, stats = compute_grpo_rewards(prompts, completions, expected)

    print("=" * 60)
    print("GRPO Reward Computation Example")
    print("=" * 60)

    for i, (prompt, reward_group) in enumerate(zip(prompts, rewards)):
        print(f"\nPrompt {i+1}: {prompt[:40]}...")
        for j, reward in enumerate(reward_group):
            print(f"  Completion {j+1}: {reward:.4f}")
        print(f"  Spread: {max(reward_group) - min(reward_group):.4f}")

    print(f"\nBatch Statistics:")
    print(f"  Mean score: {stats['mean_score']:.4f}")
    print(f"  Mean spread: {stats['mean_spread']:.4f}")
    print(f"  Min spread: {stats['min_spread']:.4f}")
