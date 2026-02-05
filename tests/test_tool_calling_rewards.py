"""
Test suite for tool calling reward functions.

Run with: python -m pytest tests/test_tool_calling_rewards.py -v
Or: python tests/test_tool_calling_rewards.py
"""

import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_grpo.trainer.tool_calling_reward import (
    extract_function_calls,
    parse_parameters,
    compare_function_calls,
    tool_call_exact_match,
    tool_call_function_match,
    tool_call_parameter_match,
    tool_call_overall_quality,
)


def test_extract_simple_function():
    """Test extracting a simple function call."""
    text = "calculate_factorial(n=5)"
    calls = extract_function_calls(text)

    assert len(calls) == 1
    assert calls[0]['name'] == 'calculate_factorial'
    assert calls[0]['params'] == {'n': 5}
    print("✓ test_extract_simple_function")


def test_extract_multiple_functions():
    """Test extracting multiple function calls."""
    text = """calculate_factorial(n=5)
calculate_factorial(n=7)
calculate_factorial(n=10)"""

    calls = extract_function_calls(text)

    assert len(calls) == 3
    assert all(c['name'] == 'calculate_factorial' for c in calls)
    assert calls[0]['params'] == {'n': 5}
    assert calls[1]['params'] == {'n': 7}
    assert calls[2]['params'] == {'n': 10}
    print("✓ test_extract_multiple_functions")


def test_extract_string_params():
    """Test extracting functions with string parameters."""
    text = 'get_balance(address="0x123456", token="WBNB")'
    calls = extract_function_calls(text)

    assert len(calls) == 1
    assert calls[0]['name'] == 'get_balance'
    assert calls[0]['params']['address'] == '0x123456'
    assert calls[0]['params']['token'] == 'WBNB'
    print("✓ test_extract_string_params")


def test_extract_mixed_params():
    """Test extracting functions with mixed parameter types."""
    text = 'bacterial_growth(initial_population=1000, growth_rate=0.03, time=90, doubling_time=25)'
    calls = extract_function_calls(text)

    assert len(calls) == 1
    assert calls[0]['name'] == 'bacterial_growth'
    assert calls[0]['params']['initial_population'] == 1000
    assert calls[0]['params']['growth_rate'] == 0.03
    assert calls[0]['params']['time'] == 90
    assert calls[0]['params']['doubling_time'] == 25
    print("✓ test_extract_mixed_params")


def test_extract_list_params():
    """Test extracting functions with list parameters."""
    text = 'euclidean_distance(point_a=[1, 1], point_b=[4, 5])'
    calls = extract_function_calls(text)

    assert len(calls) == 1
    assert calls[0]['name'] == 'euclidean_distance'
    assert calls[0]['params']['point_a'] == [1, 1]
    assert calls[0]['params']['point_b'] == [4, 5]
    print("✓ test_extract_list_params")


def test_parse_parameters():
    """Test parameter parsing."""
    # Simple params
    params = parse_parameters('n=5')
    assert params == {'n': 5}

    # Multiple params
    params = parse_parameters('a=1, b=2, c=3')
    assert params == {'a': 1, 'b': 2, 'c': 3}

    # String params
    params = parse_parameters('name="test", value=123')
    assert params == {'name': 'test', 'value': 123}

    # List params
    params = parse_parameters('points=[1, 2, 3]')
    assert params == {'points': [1, 2, 3]}

    print("✓ test_parse_parameters")


def test_compare_exact_match():
    """Test comparing identical function calls."""
    pred = [{'name': 'func', 'params': {'a': 1, 'b': 2}}]
    exp = [{'name': 'func', 'params': {'a': 1, 'b': 2}}]

    result = compare_function_calls(pred, exp)

    assert result['exact_match'] == 1.0
    assert result['function_match'] == 1.0
    assert result['param_match'] == 1.0
    assert result['overall'] == 1.0
    print("✓ test_compare_exact_match")


def test_compare_wrong_function():
    """Test comparing different function names."""
    pred = [{'name': 'func1', 'params': {'a': 1}}]
    exp = [{'name': 'func2', 'params': {'a': 1}}]

    result = compare_function_calls(pred, exp)

    assert result['exact_match'] == 0.0
    assert result['function_match'] == 0.0
    print("✓ test_compare_wrong_function")


def test_compare_correct_function_wrong_params():
    """Test correct function but wrong parameters."""
    pred = [{'name': 'func', 'params': {'a': 1, 'b': 999}}]
    exp = [{'name': 'func', 'params': {'a': 1, 'b': 2}}]

    result = compare_function_calls(pred, exp)

    assert result['exact_match'] == 0.0
    assert result['function_match'] == 1.0
    assert 0.0 < result['param_match'] < 1.0  # Partial credit
    print("✓ test_compare_correct_function_wrong_params")


def test_reward_exact_match():
    """Test exact match reward function."""
    prompts = ["test"]
    completions = ["calculate_factorial(n=5)"]
    answers = ["calculate_factorial(n=5)"]

    scores = tool_call_exact_match(prompts, completions, answers)

    assert scores == [1.0]
    print("✓ test_reward_exact_match")


def test_reward_exact_match_fails():
    """Test exact match fails on different calls."""
    prompts = ["test"]
    completions = ["calculate_factorial(n=5)"]
    answers = ["calculate_factorial(n=7)"]

    scores = tool_call_exact_match(prompts, completions, answers)

    assert scores == [0.0]
    print("✓ test_reward_exact_match_fails")


def test_reward_function_match():
    """Test function name match reward."""
    prompts = ["test"]
    completions = ["calculate_factorial(n=5)"]
    answers = ["calculate_factorial(n=7)"]  # Different param, same function

    scores = tool_call_function_match(prompts, completions, answers)

    assert scores == [1.0]  # Function name matches
    print("✓ test_reward_function_match")


def test_reward_function_mismatch():
    """Test function name mismatch."""
    prompts = ["test"]
    completions = ["func1(n=5)"]
    answers = ["func2(n=5)"]

    scores = tool_call_function_match(prompts, completions, answers)

    assert scores == [0.0]
    print("✓ test_reward_function_mismatch")


def test_reward_multiple_functions():
    """Test reward for multiple function calls."""
    prompts = ["test"]
    completions = ["calculate_factorial(n=5)\ncalculate_factorial(n=7)"]
    answers = ["calculate_factorial(n=5)\ncalculate_factorial(n=7)"]

    scores = tool_call_exact_match(prompts, completions, answers)

    assert scores == [1.0]
    print("✓ test_reward_multiple_functions")


def test_reward_overall_quality():
    """Test overall quality reward."""
    prompts = ["test", "test", "test"]
    completions = [
        "func(a=1, b=2)",  # Perfect
        "func(a=1, b=999)",  # Right function, partial params
        "wrong_func(a=1)",  # Wrong function
    ]
    answers = [
        "func(a=1, b=2)",
        "func(a=1, b=2)",
        "func(a=1, b=2)",
    ]

    scores = tool_call_overall_quality(prompts, completions, answers)

    assert scores[0] == 1.0  # Perfect
    assert 0.0 < scores[1] < 1.0  # Partial
    assert scores[2] == 0.0  # Wrong function
    print("✓ test_reward_overall_quality")


def test_real_world_example():
    """Test with real dataset example."""
    prompts = [
        "You are a helpful assistant with access to the following functions. Use them if required.\n\n[\n {\n \"name\": \"calculate_factorial\",\n \"description\": \"Calculates the factorial of a non-negative integer.\",\n \"parameters\": {\n \"n\": {\n \"description\": \"The non-negative integer.\",\n \"type\": \"int\"\n }\n }\n }\n]\n\nWhat are the factorials of 5, 7, and 10?"
    ]
    completions = [
        "calculate_factorial(n=5)\ncalculate_factorial(n=7)\ncalculate_factorial(n=10)"
    ]
    answers = [
        "calculate_factorial(n=5)\ncalculate_factorial(n=7)\ncalculate_factorial(n=10)"
    ]

    # Test all reward functions
    exact = tool_call_exact_match(prompts, completions, answers)
    function = tool_call_function_match(prompts, completions, answers)
    overall = tool_call_overall_quality(prompts, completions, answers)

    assert exact[0] == 1.0
    assert function[0] == 1.0
    assert overall[0] == 1.0
    print("✓ test_real_world_example")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Tool Calling Reward Tests")
    print("="*60 + "\n")

    tests = [
        test_extract_simple_function,
        test_extract_multiple_functions,
        test_extract_string_params,
        test_extract_mixed_params,
        test_extract_list_params,
        test_parse_parameters,
        test_compare_exact_match,
        test_compare_wrong_function,
        test_compare_correct_function_wrong_params,
        test_reward_exact_match,
        test_reward_exact_match_fails,
        test_reward_function_match,
        test_reward_function_mismatch,
        test_reward_multiple_functions,
        test_reward_overall_quality,
        test_real_world_example,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
