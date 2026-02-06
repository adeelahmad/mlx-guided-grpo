"""
Tool Calling Dataset Loader for GRPO Training
==============================================

Specialized dataset processing for function/tool calling tasks.

Features:
- JSONL format with function definitions + user queries
- Automatic prompt formatting
- Function signature extraction
- Multi-function call support
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

__all__ = [
    "load_tool_calling_dataset",
    "format_tool_calling_prompt",
]

logger = logging.getLogger(__name__)


def load_tool_calling_dataset(
    data_path: str | Path,
    system_message: str | None = None,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    type_key: str = "type",
    ground_truth_key: str = "ground_truth",
) -> list[dict[str, Any]]:
    """
    Load tool calling dataset from JSONL file.

    Expected format per line:
    {
        "prompt": "You are a helpful assistant with access to...",
        "answer": "function_name(param1=value1, param2=value2)",
        "type": "tool_call",
        "ground_truth": "function_name",
        "ground_truth_text": "function_name",
        "possible_boxed_answers": ["func1", "func2", ...],
        "is_multi_answer": false,
        "confidence": 1.0,
        "source": "tool_calling"
    }

    Args:
        data_path: Path to JSONL file
        system_message: Optional system message to prepend
        prompt_key: Key for prompt field (default: "prompt")
        answer_key: Key for answer field (default: "answer")
        type_key: Key for type field (default: "type")
        ground_truth_key: Key for ground truth (default: "ground_truth")

    Returns:
        List of formatted dataset samples ready for GRPODataset
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info(f"Loading tool calling dataset from {data_path}")

    samples = []
    skipped = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: invalid JSON - {e}")
                skipped += 1
                continue

            # Extract fields
            prompt = item.get(prompt_key, "")
            answer = item.get(answer_key, "")
            sample_type = item.get(type_key, "tool_call")
            ground_truth = item.get(ground_truth_key, None)

            if not prompt or not answer:
                logger.warning(f"Skipping line {line_num}: missing prompt or answer")
                skipped += 1
                continue

            # Format the sample
            formatted_sample = {
                "prompt": prompt,
                "answer": answer,
                "type": sample_type,
                "ground_truth": ground_truth or answer,
                "ground_truth_text": item.get("ground_truth_text", ground_truth),
                "possible_boxed_answers": item.get("possible_boxed_answers", []),
                "is_multi_answer": item.get("is_multi_answer", False),
                "confidence": item.get("confidence", 1.0),
                "source": item.get("source", "tool_calling"),
            }

            # Add optional system message
            if system_message:
                formatted_sample["system"] = system_message

            samples.append(formatted_sample)

    logger.info(
        f"Loaded {len(samples)} samples from {data_path}"
        + (f" (skipped {skipped})" if skipped else "")
    )

    return samples


def format_tool_calling_prompt(
    user_query: str,
    available_functions: list[dict[str, Any]],
    system_message: str | None = None,
) -> str:
    """
    Format a tool calling prompt with function definitions.

    Args:
        user_query: The user's question/request
        available_functions: List of function definitions
        system_message: Optional system message

    Returns:
        Formatted prompt string

    Example:
        >>> functions = [
        ...     {
        ...         "name": "calculate_factorial",
        ...         "description": "Calculates factorial of n",
        ...         "parameters": {
        ...             "n": {"type": "int", "description": "The number"}
        ...         }
        ...     }
        ... ]
        >>> format_tool_calling_prompt("What is 5 factorial?", functions)
        "You are a helpful assistant with access to the following functions...
    """
    if system_message is None:
        system_message = "You are a helpful assistant with access to the following functions. Use them if required."

    # Format function definitions
    func_defs = []
    for func in available_functions:
        func_defs.append(json.dumps(func, indent=2))

    functions_block = "[\n" + ",\n".join(func_defs) + "\n]"

    # Combine into prompt
    prompt = f"{system_message}\n\n{functions_block}\n\n{user_query}"

    return prompt


def extract_functions_from_prompt(prompt: str) -> list[dict[str, Any]]:
    """
    Extract function definitions from a tool calling prompt.

    Looks for JSON array of function definitions in the prompt.

    Args:
        prompt: The prompt containing function definitions

    Returns:
        List of function definition dicts
    """
    import re

    # Try to find JSON array pattern
    # Look for [ ... ] containing function definitions
    match = re.search(r'\[\s*\{.*?\}\s*\]', prompt, re.DOTALL)

    if not match:
        return []

    try:
        functions = json.loads(match.group(0))
        if isinstance(functions, list):
            return functions
    except json.JSONDecodeError:
        pass

    return []


def validate_tool_call(
    function_call: str,
    available_functions: list[str],
) -> dict[str, Any]:
    """
    Validate a generated function call.

    Args:
        function_call: The generated function call string
        available_functions: List of available function names

    Returns:
        Dict with validation results:
        - is_valid: Whether the call is valid
        - function_name: Extracted function name
        - parameters: Extracted parameters dict
        - errors: List of error messages
    """
    from .tool_calling_reward import extract_function_calls

    result = {
        "is_valid": False,
        "function_name": None,
        "parameters": {},
        "errors": []
    }

    # Parse the call
    calls = extract_function_calls(function_call)

    if not calls:
        result["errors"].append("No function call found")
        return result

    if len(calls) > 1:
        result["errors"].append(f"Multiple calls found: {len(calls)}")

    # Check first call
    call = calls[0]
    func_name = call['name']
    params = call['params']

    result["function_name"] = func_name
    result["parameters"] = params

    # Validate function exists
    if func_name not in available_functions:
        result["errors"].append(f"Unknown function: {func_name}")
        result["errors"].append(f"Available: {', '.join(available_functions)}")
        return result

    # If we got here, it's valid
    result["is_valid"] = True
    return result
