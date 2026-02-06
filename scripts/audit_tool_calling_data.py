#!/usr/bin/env python3
"""
Audit tool calling data for Qwen alignment.

Checks for:
1. Mixed formats in tool_call samples
2. <think> tags in tool_call samples (should not exist!)
3. Format compatibility with Qwen's Hermes-style pre-training
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def parse_python_function_call(call_str: str) -> Dict[str, any] | None:
    """Parse Python-style function call into name and arguments."""
    # Match: function_name(param1=value1, param2=value2, ...)
    match = re.match(r'(\w+)\((.*)\)', call_str.strip())
    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2)

    # Try to parse parameters (simplified)
    # This is a rough approximation
    return {
        "name": func_name,
        "raw_params": params_str
    }


def check_hermes_format(answer: str) -> bool:
    """Check if answer is in Hermes/JSON format."""
    try:
        parsed = json.loads(answer)
        return "name" in parsed and "arguments" in parsed
    except:
        return False


def audit_dataset(file_path: Path) -> Dict:
    """Audit a dataset file for Qwen alignment issues."""

    stats = {
        "total_samples": 0,
        "by_type": Counter(),
        "tool_call_issues": [],
        "format_by_type": defaultdict(Counter),
    }

    issues = {
        "tool_call_with_think_tags": [],
        "tool_call_non_hermes_format": [],
        "mixed_formats": [],
    }

    with open(file_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Line {line_num}: Invalid JSON")
                continue

            stats["total_samples"] += 1

            sample_type = sample.get("type", "unknown")
            stats["by_type"][sample_type] += 1

            answer = sample.get("answer", "")

            # Check format
            is_hermes = check_hermes_format(answer)
            is_python_call = bool(parse_python_function_call(answer))
            has_think_tags = "<think>" in answer or "</think>" in answer

            if is_hermes:
                stats["format_by_type"][sample_type]["hermes"] += 1
            elif is_python_call:
                stats["format_by_type"][sample_type]["python_call"] += 1
            elif has_think_tags:
                stats["format_by_type"][sample_type]["thinking"] += 1
            else:
                stats["format_by_type"][sample_type]["other"] += 1

            # Check for issues specific to tool_call type
            if sample_type in ["tool_call", "tool", "function_calling"]:
                if has_think_tags:
                    issues["tool_call_with_think_tags"].append({
                        "line": line_num,
                        "answer_preview": answer[:100],
                    })

                if not is_hermes and is_python_call:
                    issues["tool_call_non_hermes_format"].append({
                        "line": line_num,
                        "answer_preview": answer[:100],
                    })

    return stats, issues


def print_report(stats: Dict, issues: Dict, file_path: Path):
    """Print audit report."""
    print("\n" + "=" * 80)
    print(f"AUDIT REPORT: {file_path.name}")
    print("=" * 80)

    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"\n   By type:")
    for type_name, count in stats['by_type'].most_common():
        pct = count / stats['total_samples'] * 100
        print(f"      {type_name:30s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n📝 Format Distribution:")
    for type_name in stats['by_type'].keys():
        format_counts = stats['format_by_type'][type_name]
        if not format_counts:
            continue
        print(f"\n   {type_name}:")
        total_for_type = sum(format_counts.values())
        for format_name, count in format_counts.most_common():
            pct = count / total_for_type * 100
            print(f"      {format_name:20s}: {count:6d} ({pct:5.1f}%)")

    print(f"\n🚨 Issues Found:")

    # Issue 1: tool_call with <think> tags
    if issues["tool_call_with_think_tags"]:
        print(f"\n   ❌ Tool calling samples with <think> tags: {len(issues['tool_call_with_think_tags'])}")
        print(f"      (These should NOT have thinking tags!)")
        for issue in issues["tool_call_with_think_tags"][:3]:
            print(f"         Line {issue['line']}: {issue['answer_preview']}...")
        if len(issues["tool_call_with_think_tags"]) > 3:
            print(f"         ... and {len(issues['tool_call_with_think_tags']) - 3} more")
    else:
        print(f"\n   ✅ No tool calling samples with <think> tags")

    # Issue 2: tool_call with non-Hermes format
    if issues["tool_call_non_hermes_format"]:
        print(f"\n   ⚠️  Tool calling samples NOT in Hermes format: {len(issues['tool_call_non_hermes_format'])}")
        print(f"      (Qwen was pre-trained with Hermes format!)")
        for issue in issues["tool_call_non_hermes_format"][:3]:
            print(f"         Line {issue['line']}: {issue['answer_preview']}...")
        if len(issues["tool_call_non_hermes_format"]) > 3:
            print(f"         ... and {len(issues['tool_call_non_hermes_format']) - 3} more")
    else:
        print(f"\n   ✅ All tool calling samples use Hermes format")

    print("\n" + "=" * 80)

    # Recommendations
    print("\n💡 Recommendations:")

    if issues["tool_call_with_think_tags"]:
        print("\n   1. CRITICAL: Remove <think> tags from tool_call samples")
        print("      - Qwen uses structured 'reasoning_content' for thinking, not <think> tags")
        print("      - Tool calling and thinking use different formats")

    if issues["tool_call_non_hermes_format"]:
        print("\n   2. HIGH PRIORITY: Convert to Hermes format for Qwen compatibility")
        print("      - Your format: function_name(param1=value1, param2=value2)")
        print("      - Hermes format: {\"name\": \"function_name\", \"arguments\": \"{...}\"}")
        print("      - Matches Qwen's pre-training")
        print("      - Compatible with OpenAI API, vLLM, etc.")

    if not issues["tool_call_with_think_tags"] and not issues["tool_call_non_hermes_format"]:
        print("\n   ✅ Dataset looks good for Qwen training!")

    print("\n" + "=" * 80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_tool_calling_data.py <dataset_file.jsonl>")
        print("   or: python audit_tool_calling_data.py <dataset_dir>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_dir():
        # Audit all .jsonl files in directory
        for file in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            file_path = path / file
            if file_path.exists():
                stats, issues = audit_dataset(file_path)
                print_report(stats, issues, file_path)
    elif path.is_file():
        stats, issues = audit_dataset(path)
        print_report(stats, issues, path)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
