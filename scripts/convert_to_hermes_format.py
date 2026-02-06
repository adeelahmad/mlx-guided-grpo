#!/usr/bin/env python3
"""
Convert tool calling data from Python format to Hermes format.

From: function_name(param1=value1, param2=value2)
To:   {"name": "function_name", "arguments": "{\"param1\": \"value1\", \"param2\": \"value2\"}"}
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict


def parse_python_call_to_hermes(call_str: str) -> str:
    """
    Convert Python-style function call to Hermes JSON format.

    Examples:
        Input:  is_hotel_available(hotel="Queens Hotel", city="Berlin, Germany", checkin="2022-03-16", checkout="2022-03-22")
        Output: {"name": "is_hotel_available", "arguments": "{\"hotel\": \"Queens Hotel\", \"city\": \"Berlin, Germany\", \"checkin\": \"2022-03-16\", \"checkout\": \"2022-03-22\"}"}
    """
    # Match function name and parameters
    match = re.match(r'(\w+)\((.*)\)$', call_str.strip())
    if not match:
        print(f"⚠️  Cannot parse: {call_str[:100]}")
        return call_str  # Return as-is if can't parse

    func_name = match.group(1)
    params_str = match.group(2).strip()

    if not params_str:
        # No parameters
        return json.dumps({
            "name": func_name,
            "arguments": "{}"
        })

    # Parse parameters more carefully
    # Handle nested quotes and commas
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

        # Track quote state
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and (i == 0 or params_str[i-1] != '\\'):
                in_quotes = False
                quote_char = None

        # Track nesting depth
        if not in_quotes:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1

        # Parameter parsing
        if char == '=' and not in_quotes and paren_depth == 0 and bracket_depth == 0:
            after_equals = True
            i += 1
            continue

        if char == ',' and not in_quotes and paren_depth == 0 and bracket_depth == 0:
            # End of parameter
            if current_param and after_equals:
                params_dict[current_param.strip()] = parse_value(current_value.strip())
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

    # Handle last parameter
    if current_param and after_equals:
        params_dict[current_param.strip()] = parse_value(current_value.strip())

    # Convert to Hermes format
    # Arguments must be a JSON-formatted string
    arguments_str = json.dumps(params_dict)

    hermes_obj = {
        "name": func_name,
        "arguments": arguments_str
    }

    return json.dumps(hermes_obj)


def parse_value(value_str: str):
    """Parse a parameter value, handling different types."""
    value_str = value_str.strip()

    # String (remove quotes)
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    # Boolean
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    # None/null
    if value_str.lower() in ['none', 'null']:
        return None

    # Number
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # List (simple)
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            return json.loads(value_str)
        except:
            pass

    # Dict (simple)
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            return json.loads(value_str)
        except:
            pass

    # Default: return as string
    return value_str


def convert_dataset(input_path: Path, output_path: Path, dry_run: bool = False):
    """Convert a dataset file from Python to Hermes format."""

    converted_count = 0
    skipped_count = 0
    error_count = 0

    lines_out = []

    with open(input_path) as f_in:
        for line_num, line in enumerate(f_in, 1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Line {line_num}: Invalid JSON, skipping")
                error_count += 1
                continue

            sample_type = sample.get("type", "unknown")

            # Only convert tool_call types
            if sample_type in ["tool_call", "tool", "function_calling"]:
                answer = sample.get("answer", "")

                # Check if already in Hermes format
                try:
                    parsed = json.loads(answer)
                    if "name" in parsed and "arguments" in parsed:
                        # Already Hermes format
                        skipped_count += 1
                        lines_out.append(line)
                        continue
                except:
                    pass

                # Convert to Hermes
                try:
                    hermes_answer = parse_python_call_to_hermes(answer)
                    sample["answer"] = hermes_answer

                    # Store original for reference
                    sample["original_answer"] = answer

                    converted_count += 1

                    if line_num <= 3:  # Show first 3 conversions
                        print(f"\n📝 Example conversion (line {line_num}):")
                        print(f"   Original: {answer[:80]}...")
                        print(f"   Hermes:   {hermes_answer[:80]}...")

                except Exception as e:
                    print(f"❌ Line {line_num}: Conversion error: {e}")
                    print(f"   Answer: {answer[:100]}...")
                    error_count += 1
                    lines_out.append(line)  # Keep original
                    continue

            lines_out.append(json.dumps(sample) + "\n")

    # Write output
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f_out:
            f_out.writelines(lines_out)

    return converted_count, skipped_count, error_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_hermes_format.py <input_dir_or_file> [output_dir]")
        print("\nExamples:")
        print("  python convert_to_hermes_format.py /path/to/dataset/v12")
        print("  python convert_to_hermes_format.py /path/to/dataset/v12 /path/to/dataset/v12_hermes")
        print("\nAdd --dry-run to preview without writing")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    # Determine output path
    if len(sys.argv) >= 3 and sys.argv[2] != "--dry-run":
        output_path = Path(sys.argv[2])
    else:
        # Default: add _hermes suffix
        if input_path.is_dir():
            output_path = input_path.parent / f"{input_path.name}_hermes"
        else:
            output_path = input_path.parent / f"{input_path.stem}_hermes{input_path.suffix}"

    print("\n" + "=" * 80)
    print("CONVERT TO HERMES FORMAT")
    print("=" * 80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    if dry_run:
        print("\n🔍 DRY RUN MODE (no files will be written)")
    print("\n" + "=" * 80)

    total_converted = 0
    total_skipped = 0
    total_errors = 0

    if input_path.is_dir():
        # Convert all .jsonl files
        for filename in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            input_file = input_path / filename
            if not input_file.exists():
                continue

            output_file = output_path / filename

            print(f"\n📁 Processing {filename}...")
            converted, skipped, errors = convert_dataset(input_file, output_file, dry_run)

            total_converted += converted
            total_skipped += skipped
            total_errors += errors

            print(f"   ✅ Converted: {converted}")
            print(f"   ⏭️  Skipped:   {skipped}")
            if errors > 0:
                print(f"   ❌ Errors:    {errors}")

    elif input_path.is_file():
        converted, skipped, errors = convert_dataset(input_path, output_path, dry_run)
        total_converted = converted
        total_skipped = skipped
        total_errors = errors

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total converted: {total_converted}")
    print(f"Total skipped:   {total_skipped} (already Hermes format)")
    print(f"Total errors:    {total_errors}")

    if not dry_run and total_converted > 0:
        print(f"\n✅ Conversion complete!")
        print(f"   Output saved to: {output_path}")
        print(f"\n📝 Next steps:")
        print(f"   1. Review the converted data")
        print(f"   2. Update train.sh to use new data path:")
        print(f"      DATA=\"{output_path}\"")
        print(f"   3. Update curriculum for Hermes format")
        print(f"   4. Restart training")
    elif dry_run:
        print(f"\n💡 Remove --dry-run to actually convert the data")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
