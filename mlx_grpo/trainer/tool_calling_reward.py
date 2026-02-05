"""
Tool/Function Calling Reward Functions for GRPO Training
========================================================

Specialized reward functions for evaluating function calling capabilities.
Handles single and multi-function calls with parameter validation.

Features:
- Exact match rewards for perfect function calls
- Function name extraction and matching
- Parameter validation with partial credit
- Multi-function call support
- Normalized scores [0.0, 1.0]
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from .rewards.registry import reward

# =============================================================================
# REGEX PATTERNS FOR FUNCTION CALL PARSING
# =============================================================================

# Matches function_name(param1=value1, param2=value2)
RE_FUNC_CALL = re.compile(
    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)",
    re.MULTILINE | re.DOTALL
)

# Matches parameter assignments: param=value
RE_PARAM = re.compile(
    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)(?:,\s*|\s*$)",
    re.DOTALL
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_function_calls(text: str) -> list[dict[str, Any]]:
    """
    Extract function calls from text.

    Parses text like:
        calculate_factorial(n=5)
        get_balance(address="0x123", token="WBNB")

    Returns:
        List of dicts with 'name' and 'params' keys
        Example: [{'name': 'calculate_factorial', 'params': {'n': 5}}]
    """
    calls = []

    for match in RE_FUNC_CALL.finditer(text):
        func_name = match.group(1)
        params_str = match.group(2).strip()

        # Parse parameters
        params = {}
        if params_str:
            params = parse_parameters(params_str)

        calls.append({
            'name': func_name,
            'params': params
        })

    return calls


def parse_parameters(params_str: str) -> dict[str, Any]:
    """
    Parse function parameters from string.

    Handles:
    - Strings: "value" or 'value'
    - Numbers: 123, 123.45
    - Lists: [1, 2, 3]
    - Dicts: {"key": "value"}
    - Booleans: True, False
    - None

    Args:
        params_str: Parameter string like 'n=5, name="test"'

    Returns:
        Dict mapping parameter names to values
    """
    params = {}

    # Try to parse as Python literal first (safer than eval)
    # Wrap in dict format for ast.literal_eval
    try:
        # Build a dict string: {param1: value1, param2: value2}
        dict_str = "{" + params_str + "}"
        parsed = ast.literal_eval(dict_str)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except (ValueError, SyntaxError):
        pass

    # Fallback: manual parsing
    # Split by commas, but be careful with nested structures
    current_pos = 0
    while current_pos < len(params_str):
        # Find next parameter
        match = re.match(
            r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*',
            params_str[current_pos:]
        )

        if not match:
            break

        param_name = match.group(1)
        value_start = current_pos + match.end()

        # Find value end (next comma at same nesting level or end of string)
        value_end = find_value_end(params_str, value_start)
        value_str = params_str[value_start:value_end].strip()

        # Parse value
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Keep as string if can't parse
            value = value_str.strip('"\'')

        params[param_name] = value
        current_pos = value_end + 1  # Skip comma

    return params


def find_value_end(text: str, start: int) -> int:
    """
    Find the end of a parameter value considering nesting.

    Handles brackets [], braces {}, and quotes.
    """
    depth = 0
    in_quote = None
    i = start

    while i < len(text):
        char = text[i]

        # Handle quotes
        if char in ('"', "'"):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char

        # Handle nesting (only when not in quote)
        elif in_quote is None:
            if char in ('[', '{', '('):
                depth += 1
            elif char in (']', '}', ')'):
                depth -= 1
            elif char == ',' and depth == 0:
                return i

        i += 1

    return i


def normalize_value(value: Any) -> Any:
    """
    Normalize a value for comparison.

    - Converts floats to rounded values
    - Normalizes string case and whitespace
    - Handles None/null equivalence
    """
    if value is None or value == "null":
        return None

    if isinstance(value, str):
        # Strip quotes and whitespace
        return value.strip().strip('"\'').lower()

    if isinstance(value, float):
        # Round to 6 decimal places to avoid float comparison issues
        return round(value, 6)

    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]

    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}

    return value


def compare_function_calls(
    predicted_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]]
) -> dict[str, float]:
    """
    Compare predicted and expected function calls.

    Returns:
        Dict with scores:
        - exact_match: 1.0 if perfect match, else 0.0
        - function_match: Ratio of correct function names
        - param_match: Ratio of correct parameters
        - overall: Weighted combination
    """
    if not predicted_calls and not expected_calls:
        return {
            'exact_match': 1.0,
            'function_match': 1.0,
            'param_match': 1.0,
            'overall': 1.0
        }

    if not predicted_calls or not expected_calls:
        return {
            'exact_match': 0.0,
            'function_match': 0.0,
            'param_match': 0.0,
            'overall': 0.0
        }

    # Check if same number of calls
    if len(predicted_calls) != len(expected_calls):
        # Partial credit for any correct calls found
        function_match = sum(
            1 for pred in predicted_calls
            if any(pred['name'] == exp['name'] for exp in expected_calls)
        ) / max(len(predicted_calls), len(expected_calls))

        return {
            'exact_match': 0.0,
            'function_match': function_match,
            'param_match': 0.0,
            'overall': function_match * 0.5
        }

    # Compare each call
    function_matches = 0
    param_matches = 0
    perfect_matches = 0

    for pred, exp in zip(predicted_calls, expected_calls):
        # Check function name
        if pred['name'] == exp['name']:
            function_matches += 1

            # Check parameters
            pred_params = {k: normalize_value(v) for k, v in pred['params'].items()}
            exp_params = {k: normalize_value(v) for k, v in exp['params'].items()}

            if pred_params == exp_params:
                param_matches += 1
                perfect_matches += 1
            else:
                # Partial credit for partial parameter match
                if exp_params:
                    matching_params = sum(
                        1 for k, v in exp_params.items()
                        if k in pred_params and pred_params[k] == v
                    )
                    param_matches += matching_params / len(exp_params)

    n_calls = len(expected_calls)
    exact_match = 1.0 if perfect_matches == n_calls else 0.0
    function_match = function_matches / n_calls
    param_match = param_matches / n_calls

    # Overall score: weighted combination
    # Function name is most important, then params, then exact match
    overall = (
        0.4 * function_match +
        0.4 * param_match +
        0.2 * exact_match
    )

    return {
        'exact_match': exact_match,
        'function_match': function_match,
        'param_match': param_match,
        'overall': overall
    }


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

@reward("tool_call_exact", default=True)
def tool_call_exact_match(
    prompts: list[str],
    completions: list[str],
    answers: list[str],
    types: Optional[list[str]] = None
) -> list[float]:
    """
    Exact match reward for tool/function calling.

    Returns 1.0 only if the completion exactly matches the expected answer.
    Handles multi-line function calls and whitespace normalization.

    Args:
        prompts: User prompts (not used)
        completions: Model-generated function calls
        answers: Expected function calls
        types: Optional type indicators (not used)

    Returns:
        List of scores in [0.0, 1.0]
    """
    scores = []

    for completion, answer in zip(completions, answers):
        # Normalize whitespace and compare
        comp_norm = re.sub(r'\s+', ' ', completion.strip().lower())
        ans_norm = re.sub(r'\s+', ' ', answer.strip().lower())

        score = 1.0 if comp_norm == ans_norm else 0.0
        scores.append(score)

    return scores


@reward("tool_call_function", default=True)
def tool_call_function_match(
    prompts: list[str],
    completions: list[str],
    answers: list[str],
    types: Optional[list[str]] = None
) -> list[float]:
    """
    Reward based on correct function name(s) being called.

    Returns 1.0 if all expected function names are present,
    with partial credit for partial matches.

    Args:
        prompts: User prompts (not used)
        completions: Model-generated function calls
        answers: Expected function calls
        types: Optional type indicators (not used)

    Returns:
        List of scores in [0.0, 1.0]
    """
    scores = []

    for completion, answer in zip(completions, answers):
        pred_calls = extract_function_calls(completion)
        exp_calls = extract_function_calls(answer)

        if not exp_calls:
            # No expected calls - score 1.0 if no predictions, else 0.0
            scores.append(1.0 if not pred_calls else 0.0)
            continue

        if not pred_calls:
            scores.append(0.0)
            continue

        # Check function name matches
        pred_names = {call['name'] for call in pred_calls}
        exp_names = {call['name'] for call in exp_calls}

        # Ratio of correct names
        correct = len(pred_names & exp_names)
        total = len(exp_names)
        score = correct / total if total > 0 else 0.0

        scores.append(score)

    return scores


@reward("tool_call_params", default=False)
def tool_call_parameter_match(
    prompts: list[str],
    completions: list[str],
    answers: list[str],
    types: Optional[list[str]] = None
) -> list[float]:
    """
    Reward based on correct parameters being provided.

    Requires correct function name AND correct parameters.
    Provides partial credit for partially correct parameters.

    Args:
        prompts: User prompts (not used)
        completions: Model-generated function calls
        answers: Expected function calls
        types: Optional type indicators (not used)

    Returns:
        List of scores in [0.0, 1.0]
    """
    scores = []

    for completion, answer in zip(completions, answers):
        pred_calls = extract_function_calls(completion)
        exp_calls = extract_function_calls(answer)

        comparison = compare_function_calls(pred_calls, exp_calls)
        scores.append(comparison['param_match'])

    return scores


@reward("tool_call_overall", default=False)
def tool_call_overall_quality(
    prompts: list[str],
    completions: list[str],
    answers: list[str],
    types: Optional[list[str]] = None
) -> list[float]:
    """
    Overall tool calling quality reward.

    Weighted combination of:
    - Function name correctness (40%)
    - Parameter correctness (40%)
    - Exact match bonus (20%)

    Args:
        prompts: User prompts (not used)
        completions: Model-generated function calls
        answers: Expected function calls
        types: Optional type indicators (not used)

    Returns:
        List of scores in [0.0, 1.0]
    """
    scores = []

    for completion, answer in zip(completions, answers):
        pred_calls = extract_function_calls(completion)
        exp_calls = extract_function_calls(answer)

        comparison = compare_function_calls(pred_calls, exp_calls)
        scores.append(comparison['overall'])

    return scores


@reward("tool_call_parseable", default=False)
def tool_call_parseable(
    prompts: list[str],
    completions: list[str],
    answers: list[str],
    types: Optional[list[str]] = None
) -> list[float]:
    """
    Reward for generating parseable function calls.

    Returns 1.0 if the completion contains at least one parseable
    function call, else 0.0.

    Args:
        prompts: User prompts (not used)
        completions: Model-generated function calls
        answers: Expected function calls (not used)
        types: Optional type indicators (not used)

    Returns:
        List of scores in [0.0, 1.0]
    """
    scores = []

    for completion in completions:
        pred_calls = extract_function_calls(completion)
        scores.append(1.0 if pred_calls else 0.0)

    return scores
