#!/usr/bin/env python3
"""
Convert v14 dataset to Qwen-native tool calling format.

Changes for tool_call samples:
1. Extracts tool definitions from prompt text -> stores as "tools" field (OpenAI schema)
2. Extracts user question -> stores as clean "prompt"
3. Converts answers to Hermes <tool_call> format
4. Passes non-tool_call samples through unchanged

Usage:
    python scripts/convert_v14_to_qwen_tools.py /path/to/v14 /path/to/v15
    python scripts/convert_v14_to_qwen_tools.py /path/to/v14  # outputs to v14_qwen
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Reuse parser from existing converter
sys.path.insert(0, str(Path(__file__).parent))
from convert_to_hermes_format import parse_python_call_to_hermes, parse_value


# =============================================================================
# TYPE MAPPING: Dataset types -> JSON Schema types
# =============================================================================

TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
    # Handle compound types like "int, optional"
    "int, optional": "integer",
    "str, optional": "string",
    "float, optional": "number",
    "bool, optional": "boolean",
    "list, optional": "array",
}


def map_param_type(raw_type: str) -> str:
    """Map dataset parameter type to JSON Schema type."""
    raw = raw_type.strip().lower()
    return TYPE_MAP.get(raw, "string")


# =============================================================================
# TOOL DEFINITION CONVERSION
# =============================================================================

def convert_tool_to_openai_schema(tool_def: dict) -> dict:
    """Convert a simplified tool definition to OpenAI/Qwen format.

    From:
        {"name": "func", "description": "...", "parameters": {
            "param1": {"description": "...", "type": "str", "default": "val"},
            "param2": {"description": "...", "type": "int"}
        }}

    To:
        {"type": "function", "function": {"name": "func", "description": "...",
            "parameters": {"type": "object", "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "description": "..."}
            }, "required": ["param2"]}
        }}
    """
    name = tool_def.get("name", "unknown")
    description = tool_def.get("description", "")
    raw_params = tool_def.get("parameters", {})

    properties = {}
    required = []

    for param_name, param_info in raw_params.items():
        if isinstance(param_info, dict):
            prop = {
                "type": map_param_type(param_info.get("type", "string")),
                "description": param_info.get("description", ""),
            }
            # Add enum if present
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            # Add default if present
            if "default" in param_info:
                prop["default"] = param_info["default"]

            properties[param_name] = prop

            # Parameters without default are required
            if "default" not in param_info or param_info["default"] == "":
                required.append(param_name)
        else:
            # Simple type string
            properties[param_name] = {"type": map_param_type(str(param_info))}
            required.append(param_name)

    parameters_schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters_schema,
        },
    }


# =============================================================================
# PROMPT PARSING
# =============================================================================

def _find_matching_bracket(text: str, start: int) -> int:
    """Find the closing ] that matches the [ at position start.

    Properly handles nested brackets, strings, and escapes.
    """
    depth = 0
    in_string = False
    string_char = None
    i = start

    while i < len(text):
        char = text[i]

        if in_string:
            if char == "\\" and i + 1 < len(text):
                i += 2  # Skip escaped character
                continue
            if char == string_char:
                in_string = False
        else:
            if char in ('"', "'"):
                in_string = True
                string_char = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    return -1  # No matching bracket found


def extract_tools_and_question(prompt: str) -> tuple[list[dict], str]:
    """Extract tool definitions and user question from prompt text.

    The v14 format embeds tools in the prompt:
        "You are a helpful assistant with access to the following functions...
        [{...tool defs...}]
        <user question>"

    Returns:
        (tools_openai_format, user_question)
    """
    # Find the JSON array of tool definitions - must match brackets properly
    bracket_start = prompt.find("[")
    if bracket_start < 0:
        return [], prompt

    bracket_end = _find_matching_bracket(prompt, bracket_start)
    if bracket_end < 0:
        return [], prompt

    json_str = prompt[bracket_start : bracket_end + 1]

    try:
        raw_tools = json.loads(json_str)
    except json.JSONDecodeError:
        return [], prompt

    if not isinstance(raw_tools, list) or not raw_tools:
        return [], prompt

    # Verify these look like tool definitions (have "name" field)
    if not all(isinstance(t, dict) and "name" in t for t in raw_tools):
        return [], prompt

    # Convert each tool to OpenAI schema
    tools = [convert_tool_to_openai_schema(t) for t in raw_tools]

    # Extract user question (everything after the closing bracket)
    user_question = prompt[bracket_end + 1 :].strip()

    if not user_question:
        # Fallback: maybe the question is before the tools
        user_question = prompt[:bracket_start].strip()
        # Remove the preamble
        preamble_end = user_question.rfind("\n")
        if preamble_end > 0:
            user_question = user_question[preamble_end:].strip()

    return tools, user_question


# =============================================================================
# ANSWER CONVERSION
# =============================================================================

def convert_answer_to_hermes(answer: str) -> str:
    """Convert answer to Hermes <tool_call> format.

    Handles:
    - Python-style: func(a=5, b=3) -> <tool_call>{"name":"func","arguments":{"a":5,"b":3}}</tool_call>
    - JSON-style: {"name":"func","arguments":"{...}"} -> <tool_call>{"name":"func","arguments":{...}}</tool_call>
    - Multi-call: func1()\nfunc2() -> multiple <tool_call> blocks
    """
    answer = answer.strip()

    # Check if already in Hermes format
    if "<tool_call>" in answer:
        return answer

    # Check if JSON-style (single or multi)
    if answer.startswith("{") and '"name"' in answer:
        return _convert_json_answer(answer)

    # Python-style (single or multi-line)
    return _convert_python_answer(answer)


def _convert_json_answer(answer: str) -> str:
    """Convert JSON-style answer to Hermes format."""
    # Could be multiple JSON objects on separate lines
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    tool_calls = []

    for line in lines:
        try:
            parsed = json.loads(line)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})

            # Arguments might be a JSON string - parse it
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass

            tool_call_obj = {"name": name, "arguments": arguments}
            tool_calls.append(
                f"<tool_call>\n{json.dumps(tool_call_obj)}\n</tool_call>"
            )
        except json.JSONDecodeError:
            # Try python-style fallback
            hermes_json = _python_call_to_hermes_obj(line)
            if hermes_json:
                tool_calls.append(
                    f"<tool_call>\n{json.dumps(hermes_json)}\n</tool_call>"
                )

    return "\n".join(tool_calls) if tool_calls else answer


def _convert_python_answer(answer: str) -> str:
    """Convert Python-style answer to Hermes format."""
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    tool_calls = []

    for line in lines:
        hermes_obj = _python_call_to_hermes_obj(line)
        if hermes_obj:
            tool_calls.append(
                f"<tool_call>\n{json.dumps(hermes_obj)}\n</tool_call>"
            )
        else:
            # Can't parse - keep raw (will be caught by validation later)
            tool_calls.append(line)

    return "\n".join(tool_calls)


def _python_call_to_hermes_obj(call_str: str) -> dict | None:
    """Convert a Python-style function call to a Hermes dict.

    Returns {"name": "func", "arguments": {...}} or None if can't parse.
    """
    call_str = call_str.strip()
    match = re.match(r"(\w+)\((.*)\)$", call_str, re.DOTALL)
    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2).strip()

    if not params_str:
        return {"name": func_name, "arguments": {}}

    # Parse parameters using the existing parser logic
    params_dict = _parse_params(params_str)
    return {"name": func_name, "arguments": params_dict}


def _parse_params(params_str: str) -> dict:
    """Parse Python-style parameters into a dict."""
    params_dict = {}
    current_param = ""
    current_value = ""
    in_quotes = False
    quote_char = None
    after_equals = False
    paren_depth = 0
    bracket_depth = 0

    i = 0
    while i < len(params_str):
        char = params_str[i]

        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and (i == 0 or params_str[i - 1] != "\\"):
                in_quotes = False
                quote_char = None

        if not in_quotes:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1

        if (
            char == "="
            and not in_quotes
            and paren_depth == 0
            and bracket_depth == 0
        ):
            after_equals = True
            i += 1
            continue

        if (
            char == ","
            and not in_quotes
            and paren_depth == 0
            and bracket_depth == 0
        ):
            if current_param and after_equals:
                params_dict[current_param.strip()] = parse_value(
                    current_value.strip()
                )
            current_param = ""
            current_value = ""
            after_equals = False
            i += 1
            continue

        if after_equals:
            current_value += char
        else:
            current_param += char

        i += 1

    if current_param and after_equals:
        params_dict[current_param.strip()] = parse_value(current_value.strip())

    return params_dict


# =============================================================================
# MAIN CONVERSION
# =============================================================================

def convert_sample(sample: dict) -> dict:
    """Convert a single tool_call sample to Qwen format."""
    sample = sample.copy()

    # Extract tools from prompt
    tools, user_question = extract_tools_and_question(sample["prompt"])

    if tools:
        sample["tools"] = tools
        sample["prompt"] = user_question

    # Convert answer to Hermes format
    original_answer = sample["answer"]
    hermes_answer = convert_answer_to_hermes(original_answer)
    sample["answer"] = hermes_answer

    # Preserve original if different
    if hermes_answer != original_answer and "original_answer" not in sample:
        sample["original_answer"] = original_answer

    return sample


def convert_file(input_path: Path, output_path: Path) -> dict:
    """Convert a dataset file."""
    stats = {
        "total": 0,
        "converted": 0,
        "passed_through": 0,
        "errors": 0,
        "no_tools_found": 0,
    }

    output_lines = []

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            stats["total"] += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Line {line_num}: Invalid JSON, skipping")
                stats["errors"] += 1
                continue

            sample_type = sample.get("type", "")

            if sample_type == "tool_call":
                try:
                    converted = convert_sample(sample)
                    if "tools" in converted:
                        stats["converted"] += 1
                    else:
                        stats["no_tools_found"] += 1

                    output_lines.append(json.dumps(converted) + "\n")

                    if stats["converted"] <= 2:
                        print(f"\n  Example (line {line_num}):")
                        print(f"    prompt: {converted['prompt'][:80]}...")
                        print(f"    tools:  {len(converted.get('tools', []))} functions")
                        print(f"    answer: {converted['answer'][:100]}...")

                except Exception as e:
                    print(f"  Line {line_num}: Error: {e}")
                    stats["errors"] += 1
                    output_lines.append(line)
            else:
                # Pass through non-tool_call samples
                stats["passed_through"] += 1
                output_lines.append(line)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(output_lines)

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_v14_to_qwen_tools.py <input_dir> [output_dir]")
        print("\nConverts tool_call samples to Qwen-native format:")
        print("  - Extracts tool definitions from prompt -> tools field")
        print("  - Converts answers to Hermes <tool_call> format")
        print("  - Passes exam samples through unchanged")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.parent / f"{input_path.name}_qwen"

    print("=" * 70)
    print("CONVERT TO QWEN TOOL CALLING FORMAT")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    total_stats = {
        "total": 0,
        "converted": 0,
        "passed_through": 0,
        "errors": 0,
        "no_tools_found": 0,
    }

    for filename in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        input_file = input_path / filename
        if not input_file.exists():
            continue

        output_file = output_path / filename
        print(f"\nProcessing {filename}...")

        stats = convert_file(input_file, output_file)

        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"  Converted:    {stats['converted']}")
        print(f"  Passed thru:  {stats['passed_through']}")
        print(f"  No tools:     {stats['no_tools_found']}")
        if stats["errors"]:
            print(f"  Errors:       {stats['errors']}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples:   {total_stats['total']}")
    print(f"Tool calls conv: {total_stats['converted']}")
    print(f"Passed through:  {total_stats['passed_through']}")
    print(f"No tools found:  {total_stats['no_tools_found']}")
    print(f"Errors:          {total_stats['errors']}")
    print(f"\nOutput: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
