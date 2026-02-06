"""
Type Scaffold Generator - Rails-style boilerplate for new types
================================================================

Generates the 3 required files (reward, loader, generator) for a new
type, wires up __init__.py imports, and registers aliases in coordinator.

Usage:
    python -m mlx_grpo.scaffold <type_name> [--thinking] [--aliases a,b,c]
    mlx-grpo-scaffold <type_name> [--thinking] [--aliases a,b,c]

Examples:
    python -m mlx_grpo.scaffold math --thinking --aliases math,arithmetic,calculus
    python -m mlx_grpo.scaffold python --thinking --aliases python,code,coding,programming
    python -m mlx_grpo.scaffold sql --aliases sql,database,query
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

# Base directory for type_system_v2
TYPE_SYSTEM_DIR = Path(__file__).parent / "trainer" / "type_system_v2"


# =============================================================================
# TEMPLATES
# =============================================================================

def _reward_template(type_name: str, class_prefix: str, thinking: bool) -> str:
    """Generate reward module source."""
    title = f"{class_prefix} Reward"
    underline = "=" * (len(title) + 20)

    if thinking:
        extract_block = textwrap.dedent("""\
            RE_THINK_EXTRACT = re.compile(r"<think>(.*?)</think>\\s*(.*)", re.DOTALL)
            RE_BOXED = re.compile(r"\\\\boxed\\{([^}]*)\\}")
            RE_STRICT_FORMAT = re.compile(r"^<think>\\n[\\s\\S]*?\\n</think>\\n[\\s\\S]*$")
        """)
        extract_method = textwrap.dedent("""\
            def _extract_components(self, text: str) -> tuple[str | None, str]:
                \"\"\"Extract thinking and answer content.\"\"\"
                if not text:
                    return None, ""
                match = RE_THINK_EXTRACT.search(text)
                if match:
                    return match.group(1).strip(), match.group(2).strip()
                return None, text.strip()
        """)
        format_method = textwrap.dedent("""\
            def _score_format(self, completion: str) -> float:
                \"\"\"Score format quality (think/answer structure).\"\"\"
                if not completion:
                    return 0.0
                if RE_STRICT_FORMAT.search(completion.strip()):
                    return 1.0
                score = 0.0
                if "<think>" in completion:
                    score += 0.25
                if "</think>" in completion:
                    score += 0.25
                open_pos = completion.find("<think>")
                close_pos = completion.find("</think>")
                if open_pos >= 0 and close_pos > open_pos:
                    between = completion[open_pos + 7:close_pos].strip()
                    if len(between) >= 5:
                        score += 0.25
                if close_pos >= 0:
                    after = completion[close_pos + 8:].strip()
                    if len(after) >= 2:
                        score += 0.25
                return score
        """)
        weights = textwrap.dedent("""\
            def get_component_weights(self) -> dict[str, float]:
                return {
                    "correctness": 0.40,
                    "format": 0.30,
                    "thinking_quality": 0.30,
                }
        """)
        compute_body = textwrap.dedent("""\
            component_scores = {}
            metadata = {}

            thinking, answer_content = self._extract_components(completion)
            metadata["has_thinking"] = bool(thinking)

            # Component 1: Correctness
            component_scores["correctness"] = self._score_correctness(
                answer_content, answer
            )

            # Component 2: Format
            component_scores["format"] = self._score_format(completion)

            # Component 3: Thinking Quality
            component_scores["thinking_quality"] = (
                self._score_thinking(thinking) if thinking else 0.0
            )

            total_score = self.combine_component_scores(component_scores)
            return RewardResult(
                total_score=total_score,
                component_scores=component_scores,
                metadata=metadata,
                valid=True,
            )
        """)
    else:
        extract_block = ""
        extract_method = ""
        format_method = ""
        weights = textwrap.dedent("""\
            def get_component_weights(self) -> dict[str, float]:
                return {
                    "correctness": 0.60,
                    "format": 0.40,
                }
        """)
        compute_body = textwrap.dedent("""\
            component_scores = {}
            metadata = {}

            # Component 1: Correctness
            component_scores["correctness"] = self._score_correctness(
                completion, answer
            )

            # Component 2: Format
            component_scores["format"] = 1.0 if len(completion.strip()) > 5 else 0.0

            total_score = self.combine_component_scores(component_scores)
            return RewardResult(
                total_score=total_score,
                component_scores=component_scores,
                metadata=metadata,
                valid=True,
            )
        """)

    return f'''"""
{title} - Scoring for {type_name} tasks
{underline}

Scoring Components:
{_weights_doc(thinking)}
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.reward import BaseReward, RewardResult, RewardHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["{class_prefix}Reward"]

logger = logging.getLogger(__name__)

{extract_block}

class {class_prefix}Reward(BaseReward):
    """{title} for {type_name} tasks."""

    type_name = "{type_name}"

    def __init__(
        self,
        hooks: Optional[RewardHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)

    {textwrap.indent(weights, "    ").strip()}

    def validate_completion(
        self,
        completion: str,
        type_info: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        if not completion or len(completion.strip()) < 2:
            return False, "Empty or too short completion"
        return True, None

    def compute_single(
        self,
        prompt: str,
        completion: str,
        answer: str,
        type_info: Optional[Any] = None,
    ) -> RewardResult:
        {textwrap.indent(compute_body, "        ").strip()}

    # =========================================================================
    # SCORING HELPERS
    # =========================================================================

    {textwrap.indent(extract_method, "    ").strip()}

    def _score_correctness(self, completion_answer: str, ground_truth: str) -> float:
        """Score answer correctness."""
        if not completion_answer or not ground_truth:
            return 0.0
        if completion_answer.strip().lower() == ground_truth.strip().lower():
            return 1.0
        if ground_truth.strip().lower() in completion_answer.strip().lower():
            return 0.7
        return 0.0

    {textwrap.indent(format_method, "    ").strip()}

    def _score_thinking(self, thinking: str) -> float:
        """Score thinking quality."""
        if not thinking:
            return 0.0
        words = len(thinking.split())
        if words < 10:
            return max(0.3, words / 10)
        if words > 200:
            return 0.5
        return 1.0
'''


def _weights_doc(thinking: bool) -> str:
    if thinking:
        return "1. Correctness (40%)\n2. Format (30%)\n3. Thinking Quality (30%)"
    return "1. Correctness (60%)\n2. Format (40%)"


def _loader_template(type_name: str, class_prefix: str, thinking: bool) -> str:
    """Generate loader module source."""
    title = f"{class_prefix} Dataset Loader"
    underline = "=" * (len(title) + 20)

    system_prompt = (
        "You are an expert problem solver. "
        "Work through the problem step by step in <think>...</think> tags. "
        "Then provide your final answer."
    ) if thinking else (
        f"You are a helpful assistant specializing in {type_name} tasks."
    )

    return f'''"""
{title} - Data Loading for {type_name} tasks
{underline}

Expected format:
    {{"prompt": "...", "answer": "...", "type": "{type_name}"}}
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TYPE_CHECKING

from ..base.dataset_loader import BaseDatasetLoader, DatasetHooks

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from ..events import EventBus

__all__ = ["{class_prefix}DatasetLoader"]

logger = logging.getLogger(__name__)

# Type aliases that map to {type_name}
_{type_name.upper()}_TYPE_ALIASES = frozenset({{
    "{type_name}",
}})


class {class_prefix}DatasetLoader(BaseDatasetLoader):
    """Dataset loader for {type_name} tasks."""

    type_name = "{type_name}"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        hooks: Optional[DatasetHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(tokenizer=tokenizer, hooks=hooks, event_bus=event_bus)

    def validate_sample(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Validate {type_name} sample."""
        if "prompt" not in sample:
            return False, "Missing \'prompt\' field"
        if "answer" not in sample:
            return False, "Missing \'answer\' field"

        # Accept if explicit type match
        sample_type = str(sample.get("type", "")).lower()
        if sample_type in _{type_name.upper()}_TYPE_ALIASES:
            return True, None

        return False, "Not a {type_name} sample"

    def preprocess_sample(self, sample: dict) -> dict:
        """Preprocess {type_name} sample."""
        sample["type_info"] = {{
            "type": "{type_name}",
        }}
        return sample

    def get_system_prompt(self, sample: dict) -> str:
        return (
            "{system_prompt}"
        )

    def get_type_name(self) -> str:
        return "{type_name}"
'''


def _generator_template(
    type_name: str, class_prefix: str, thinking: bool
) -> str:
    """Generate generator module source."""
    title = f"{class_prefix} Rollout Generator"
    underline = "=" * (len(title) + 20)

    if thinking:
        return f'''"""
{title} - Generation for {type_name} tasks
{underline}

Extends ThinkingBasedGenerator with {type_name}-specific settings.
Uses two-phase generation with thinking structure.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .thinking_based import ThinkingBasedGenerator
from ..base.rollout_generator import GeneratorHooks

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["{class_prefix}RolloutGenerator"]

logger = logging.getLogger(__name__)


class {class_prefix}RolloutGenerator(ThinkingBasedGenerator):
    """{title} for {type_name} tasks.

    Uses thinking-based two-phase generation.
    Generation: max_length=1024, temperature=0.8, continuation_tokens=256
    """

    type_name = "{type_name}"

    def __init__(
        self,
        max_length: int = 1024,
        temperature: float = 0.8,
        continuation_tokens: int = 256,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(
            max_length=max_length,
            temperature=temperature,
            continuation_tokens=continuation_tokens,
            hooks=hooks,
            event_bus=event_bus,
        )
'''
    else:
        return f'''"""
{title} - Generation for {type_name} tasks
{underline}

Simple generator without thinking phase.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from ..base.rollout_generator import (
    BaseRolloutGenerator,
    GenerationConfig,
    GeneratorHooks,
)

if TYPE_CHECKING:
    from ..events import EventBus

__all__ = ["{class_prefix}RolloutGenerator"]

logger = logging.getLogger(__name__)


class {class_prefix}RolloutGenerator(BaseRolloutGenerator):
    """{title} for {type_name} tasks.

    Simple generation without thinking phase.
    Generation: max_length=512, temperature=0.8
    """

    type_name = "{type_name}"

    def __init__(
        self,
        max_length: int = 512,
        temperature: float = 0.8,
        hooks: Optional[GeneratorHooks] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(hooks=hooks, event_bus=event_bus)
        self.max_length = max_length
        self.temperature = temperature

    def get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_length=self.max_length,
            temperature=self.temperature,
            two_phase=False,
        )

    def apply_curriculum(self, answer: str, ratio: float) -> str:
        if ratio <= 0:
            return ""
        if ratio >= 1.0:
            return answer
        cutoff = max(1, int(len(answer) * ratio))
        return answer[:cutoff]

    def is_generation_complete(
        self, text: str, phase: int
    ) -> tuple[bool, Optional[str]]:
        if len(text.strip()) >= 2:
            return True, "complete"
        return False, None

    def needs_phase_recovery(self) -> bool:
        return False
'''


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def _to_class_prefix(type_name: str) -> str:
    """Convert type_name to PascalCase class prefix.

    math -> Math, python -> Python, general_qna -> GeneralQna
    """
    return "".join(word.capitalize() for word in type_name.split("_"))


def _update_init_file(init_path: Path, import_line: str, all_entry: str) -> bool:
    """Add import and __all__ entry to an __init__.py file.

    Returns True if file was modified.
    """
    content = init_path.read_text()

    if import_line.split("import ")[-1].split(",")[0].strip() in content:
        return False  # Already imported

    # Add import before __all__
    lines = content.split("\n")
    new_lines = []
    added_import = False
    added_all = False

    for line in lines:
        # Insert import before __all__
        if line.startswith("__all__") and not added_import:
            new_lines.append(import_line)
            added_import = True

        # Add to __all__ list (before closing bracket)
        if line.strip() == "]" and not added_all:
            new_lines.append(f'    "{all_entry}",')
            added_all = True

        new_lines.append(line)

    init_path.write_text("\n".join(new_lines))
    return True


def scaffold_type(
    type_name: str,
    thinking: bool = True,
    aliases: list[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Generate all boilerplate files for a new type.

    Args:
        type_name: Canonical type name (e.g. "math", "python")
        thinking: Whether this type uses <think> structure
        aliases: Optional type aliases
        dry_run: If True, only print what would be created

    Returns:
        List of created/modified file paths
    """
    class_prefix = _to_class_prefix(type_name)
    created_files = []

    # Directories
    rewards_dir = TYPE_SYSTEM_DIR / "rewards"
    loaders_dir = TYPE_SYSTEM_DIR / "loaders"
    generators_dir = TYPE_SYSTEM_DIR / "generators"

    # Check directories exist
    for d in [rewards_dir, loaders_dir, generators_dir]:
        if not d.exists():
            print(f"  ERROR: Directory not found: {d}")
            return []

    # Check if type already exists
    reward_file = rewards_dir / f"{type_name}.py"
    loader_file = loaders_dir / f"{type_name}.py"
    generator_file = generators_dir / f"{type_name}.py"

    for f in [reward_file, loader_file, generator_file]:
        if f.exists() and not dry_run:
            print(f"  WARNING: {f.name} already exists, skipping")

    # Generate files
    files_to_create = [
        (reward_file, _reward_template(type_name, class_prefix, thinking)),
        (loader_file, _loader_template(type_name, class_prefix, thinking)),
        (generator_file, _generator_template(type_name, class_prefix, thinking)),
    ]

    for file_path, content in files_to_create:
        if dry_run:
            print(f"  [DRY RUN] Would create: {file_path}")
            continue

        if file_path.exists():
            continue

        file_path.write_text(content)
        created_files.append(str(file_path))
        print(f"  Created: {file_path.name}")

    # Update __init__.py files
    if not dry_run:
        inits = [
            (
                rewards_dir / "__init__.py",
                f"from .{type_name} import {class_prefix}Reward",
                f"{class_prefix}Reward",
            ),
            (
                loaders_dir / "__init__.py",
                f"from .{type_name} import {class_prefix}DatasetLoader",
                f"{class_prefix}DatasetLoader",
            ),
            (
                generators_dir / "__init__.py",
                f"from .{type_name} import {class_prefix}RolloutGenerator",
                f"{class_prefix}RolloutGenerator",
            ),
        ]

        for init_path, import_line, all_entry in inits:
            if _update_init_file(init_path, import_line, all_entry):
                created_files.append(str(init_path))
                print(f"  Updated: {init_path.parent.name}/__init__.py")

    # Print alias instructions
    if aliases:
        print(f"\n  Aliases to add to coordinator.py _TYPE_ALIASES:")
        for alias in aliases:
            if alias != type_name:
                print(f'    "{alias}": "{type_name}",')

    # Print registration instructions
    print(f"\n  Registration in coordinator.py auto_register_builtin_types():")
    print(f"    from .rewards.{type_name} import {class_prefix}Reward")
    print(f"    from .loaders.{type_name} import {class_prefix}DatasetLoader")
    print(f"    from .generators.{type_name} import {class_prefix}RolloutGenerator")

    return created_files


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-grpo-scaffold",
        description="Generate boilerplate for a new GRPO type (like Rails scaffold)",
    )
    parser.add_argument(
        "type_name",
        help="Canonical type name (e.g. math, python, sql)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Use <think>...</think> generation structure (two-phase)",
    )
    parser.add_argument(
        "--aliases",
        type=str,
        default="",
        help="Comma-separated type aliases (e.g. math,arithmetic,calculus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be created without writing files",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    type_name = args.type_name.strip().lower().replace("-", "_")
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()] if args.aliases else []
    thinking = args.thinking

    print(f"\n  Scaffolding type: {type_name}")
    print(f"  Thinking: {thinking}")
    print(f"  Aliases: {aliases or '(none)'}")
    print()

    created = scaffold_type(
        type_name=type_name,
        thinking=thinking,
        aliases=aliases,
        dry_run=args.dry_run,
    )

    if created:
        print(f"\n  Done! Created {len(created)} files.")
    elif not args.dry_run:
        print("\n  No files created (already exist?).")

    print()


if __name__ == "__main__":
    main()
