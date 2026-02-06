#!/usr/bin/env python3
"""Comprehensive tests for balanced type shuffling."""

import sys
from collections import Counter
from unittest.mock import Mock


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text):
        return [1, 2, 3]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3, 4, 5]


def test_balanced_shuffle_basic():
    """Test basic balanced shuffling with multiple types."""
    from mlx_grpo.trainer.datasets import GRPODataset

    test_data = []
    # 30 math, 20 tool_call, 10 exam
    for i in range(30):
        test_data.append({
            "prompt": f"Math {i}",
            "answer": f"<think>Sol {i}</think> Ans {i}",
            "type": "math"
        })
    for i in range(20):
        test_data.append({
            "prompt": f"Tool {i}",
            "answer": f'{{"name": "func_{i}", "arguments": "{{}}"}}',
            "type": "tool_call"
        })
    for i in range(10):
        test_data.append({
            "prompt": f"Exam {i}",
            "answer": f"<think>Think {i}</think> \\boxed{{ans_{i}}}",
            "type": "exam",
            "ground_truth": f"ans_{i}"
        })

    tokenizer = MockTokenizer()
    dataset = GRPODataset(
        data=test_data,
        tokenizer=tokenizer,
        require_think_tags=False,
        shuffle=True,
        balanced_shuffle=True,
        seed=42
    )

    # Verify total count
    assert len(dataset) == 60, f"Expected 60 samples, got {len(dataset)}"

    # Verify overall type distribution
    all_types = [dataset[i][4].get("type") for i in range(len(dataset))]
    type_counts = Counter(all_types)
    assert type_counts['math'] == 30, "Math samples lost!"
    assert type_counts['tool_call'] == 20, "Tool call samples lost!"
    assert type_counts['exam'] == 10, "Exam samples lost!"

    print("✓ Basic balanced shuffle test passed")


def test_balanced_shuffle_distribution():
    """Test that types are evenly distributed across chunks."""
    from mlx_grpo.trainer.datasets import GRPODataset

    test_data = []
    for i in range(50):
        test_data.append({"prompt": f"Math {i}", "answer": f"<think>S{i}</think> A{i}", "type": "math"})
    for i in range(30):
        test_data.append({"prompt": f"Tool {i}", "answer": f'{{"name": "f{i}"}}', "type": "tool_call"})
    for i in range(20):
        test_data.append({"prompt": f"Exam {i}", "answer": f"<think>T{i}</think> A{i}", "type": "exam", "ground_truth": f"a{i}"})

    dataset = GRPODataset(
        data=test_data,
        tokenizer=MockTokenizer(),
        require_think_tags=False,
        shuffle=True,
        balanced_shuffle=True,
        seed=123
    )

    # Check distribution in 10-sample chunks
    chunk_size = 10
    num_chunks = len(dataset) // chunk_size

    max_variance = 0
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size

        chunk_types = [dataset[i][4].get("type") for i in range(start, end)]
        type_counts = Counter(chunk_types)

        # Calculate variance from expected proportions
        # Expected: 50% math, 30% tool_call, 20% exam
        expected = {'math': 5, 'tool_call': 3, 'exam': 2}
        variance = sum(abs(type_counts.get(t, 0) - expected[t]) for t in expected)
        max_variance = max(max_variance, variance)

    # Variance should be reasonable (proportional distribution maintained)
    # Some variance is expected due to rounding and weighted distribution
    assert max_variance <= 10, f"Distribution variance too high: {max_variance}"

    print("✓ Distribution test passed")


def test_single_type_dataset():
    """Test balanced shuffle with only one type (should work normally)."""
    from mlx_grpo.trainer.datasets import GRPODataset

    test_data = [
        {"prompt": f"Math {i}", "answer": f"<think>S{i}</think> A{i}", "type": "math"}
        for i in range(100)
    ]

    dataset = GRPODataset(
        data=test_data,
        tokenizer=MockTokenizer(),
        require_think_tags=False,
        shuffle=True,
        balanced_shuffle=True,
        seed=42
    )

    assert len(dataset) == 100
    all_types = [dataset[i][4].get("type") for i in range(len(dataset))]
    assert all(t == "math" for t in all_types)

    print("✓ Single type test passed")


def test_unbalanced_types():
    """Test with highly unbalanced type distribution."""
    from mlx_grpo.trainer.datasets import GRPODataset

    test_data = []
    # 90 math, 5 tool_call, 5 exam (very unbalanced)
    for i in range(90):
        test_data.append({"prompt": f"Math {i}", "answer": f"<think>S{i}</think> A{i}", "type": "math"})
    for i in range(5):
        test_data.append({"prompt": f"Tool {i}", "answer": f'{{"name": "f{i}"}}', "type": "tool_call"})
    for i in range(5):
        test_data.append({"prompt": f"Exam {i}", "answer": f"<think>T{i}</think> A{i}", "type": "exam", "ground_truth": f"a{i}"})

    dataset = GRPODataset(
        data=test_data,
        tokenizer=MockTokenizer(),
        require_think_tags=False,
        shuffle=True,
        balanced_shuffle=True,
        seed=42
    )

    # Verify counts preserved
    all_types = [dataset[i][4].get("type") for i in range(len(dataset))]
    type_counts = Counter(all_types)
    assert type_counts['math'] == 90
    assert type_counts['tool_call'] == 5
    assert type_counts['exam'] == 5

    # Check that rare types appear early (not clustered at end)
    first_20_types = [dataset[i][4].get("type") for i in range(20)]
    first_20_counts = Counter(first_20_types)

    # Should have at least some tool_call and exam in first 20
    assert first_20_counts.get('tool_call', 0) > 0, "tool_call not in first 20"
    assert first_20_counts.get('exam', 0) > 0, "exam not in first 20"

    print("✓ Unbalanced types test passed")


def test_shuffle_disabled():
    """Test that disabling shuffle maintains original order."""
    from mlx_grpo.trainer.datasets import GRPODataset

    test_data = [
        {"prompt": f"Item {i}", "answer": f"<think>S{i}</think> A{i}", "type": f"type_{i % 3}"}
        for i in range(30)
    ]

    dataset = GRPODataset(
        data=test_data,
        tokenizer=MockTokenizer(),
        require_think_tags=False,
        shuffle=False,  # Disabled
        balanced_shuffle=True,
        seed=42
    )

    # Should maintain original order
    for i in range(len(dataset)):
        prompt_str = dataset[i][2]
        assert f"Item {i}" == prompt_str, f"Order not preserved at index {i}"

    print("✓ Shuffle disabled test passed")


def test_balanced_shuffle_disabled():
    """Test that disabling balanced_shuffle uses simple shuffle."""
    from mlx_grpo.trainer.datasets import GRPODataset

    test_data = []
    for i in range(30):
        test_data.append({"prompt": f"Math {i}", "answer": f"<think>S{i}</think> A{i}", "type": "math"})
    for i in range(20):
        test_data.append({"prompt": f"Tool {i}", "answer": f'{{"name": "f{i}"}}', "type": "tool_call"})

    dataset = GRPODataset(
        data=test_data,
        tokenizer=MockTokenizer(),
        require_think_tags=False,
        shuffle=True,
        balanced_shuffle=False,  # Use simple shuffle
        seed=42
    )

    # Counts should be preserved
    all_types = [dataset[i][4].get("type") for i in range(len(dataset))]
    type_counts = Counter(all_types)
    assert type_counts['math'] == 30
    assert type_counts['tool_call'] == 20

    # With simple shuffle and this seed, distribution is likely less balanced
    # (just verify it ran successfully)

    print("✓ Balanced shuffle disabled test passed")


def test_reproducibility():
    """Test that same seed produces same shuffle."""
    from mlx_grpo.trainer.datasets import GRPODataset

    test_data = [
        {"prompt": f"Item {i}", "answer": f"<think>S{i}</think> A{i}", "type": f"type_{i % 3}"}
        for i in range(30)
    ]

    dataset1 = GRPODataset(
        data=test_data.copy(),
        tokenizer=MockTokenizer(),
        require_think_tags=False,
        shuffle=True,
        balanced_shuffle=True,
        seed=999
    )

    dataset2 = GRPODataset(
        data=test_data.copy(),
        tokenizer=MockTokenizer(),
        require_think_tags=False,
        shuffle=True,
        balanced_shuffle=True,
        seed=999  # Same seed
    )

    # Should produce identical order
    for i in range(len(dataset1)):
        prompt1 = dataset1[i][2]
        prompt2 = dataset2[i][2]
        assert prompt1 == prompt2, f"Order differs at index {i}"

    print("✓ Reproducibility test passed")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_balanced_shuffle_basic,
        test_balanced_shuffle_distribution,
        test_single_type_dataset,
        test_unbalanced_types,
        test_shuffle_disabled,
        test_balanced_shuffle_disabled,
        test_reproducibility,
    ]

    print("Running comprehensive balanced shuffle tests...")
    print("=" * 60)

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("=" * 60)
    print(f"\n✓✓✓ All {len(tests)} tests passed! ✓✓✓\n")
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
