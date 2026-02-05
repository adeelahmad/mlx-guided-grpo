"""
Foundation Level Rewards
========================

Level 1: Structure and format requirements.

This level checks basic requirements:
    - Think tags present and properly formed
    - Answer section exists
    - Content is complete (not truncated)
    - Basic structural integrity

Design Philosophy:
    - Soft scoring (not binary pass/fail)
    - Gradients always flow
    - Calibrated for 450-token limit
    - Partial credit for partial compliance
"""

import re
import logging
from typing import Tuple, Dict, Any, Optional

from ..core.config import get_config
from ..core.base import LevelResult, DiagnosticInfo
from ..utils.text_processing import (
    extract_sections,
    check_completion,
    check_tag_balance,
    normalize_text,
)

logger = logging.getLogger(__name__)


def check_structure(
    completion: str,
    require_think: bool = True,
    require_answer: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check structural requirements (think tags, answer section).

    Returns:
        (score, diagnostics) where score is continuous [0, 1]
    """
    config = get_config()

    diagnostics = {
        "has_think_open": False,
        "has_think_close": False,
        "has_answer": False,
        "think_length": 0,
        "answer_length": 0,
        "tag_balance": 0,
        "checks_passed": 0,
        "checks_total": 0,
    }

    if not completion:
        return 0.0, diagnostics

    # Check for think tags
    has_think_open = "<think>" in completion
    has_think_close = "</think>" in completion

    diagnostics["has_think_open"] = has_think_open
    diagnostics["has_think_close"] = has_think_close

    # Extract sections
    thinking, answer = extract_sections(completion)

    diagnostics["think_length"] = len(thinking)
    diagnostics["answer_length"] = len(answer)
    diagnostics["has_answer"] = bool(answer.strip())

    # Check tag balance
    is_balanced, imbalance, _ = check_tag_balance(completion)
    diagnostics["tag_balance"] = imbalance

    # Score components (each worth equal amount)
    score_components = []

    # 1. Think tags present
    if require_think:
        if has_think_open and has_think_close:
            score_components.append(1.0)
        elif has_think_open or has_think_close:
            score_components.append(0.5)  # Partial credit
        else:
            score_components.append(0.0)
        diagnostics["checks_total"] += 1
        if has_think_open and has_think_close:
            diagnostics["checks_passed"] += 1

    # 2. Thinking content exists
    if require_think:
        min_think = config.min_thinking_tokens
        if len(thinking.split()) >= min_think:
            score_components.append(1.0)
        elif thinking.strip():
            # Partial credit based on length
            ratio = len(thinking.split()) / max(1, min_think)
            score_components.append(min(1.0, ratio))
        else:
            score_components.append(0.0)
        diagnostics["checks_total"] += 1
        if len(thinking.split()) >= min_think:
            diagnostics["checks_passed"] += 1

    # 3. Answer section exists
    if require_answer:
        min_answer = config.min_answer_tokens
        if len(answer.split()) >= min_answer:
            score_components.append(1.0)
        elif answer.strip():
            # Partial credit
            ratio = len(answer.split()) / max(1, min_answer)
            score_components.append(min(1.0, ratio))
        else:
            # No answer but has thinking - some credit
            if thinking.strip():
                score_components.append(0.3)
            else:
                score_components.append(0.0)
        diagnostics["checks_total"] += 1
        if len(answer.split()) >= min_answer:
            diagnostics["checks_passed"] += 1

    # 4. Tag balance
    if is_balanced:
        score_components.append(1.0)
        diagnostics["checks_passed"] += 1
    else:
        score_components.append(max(0.3, 1.0 - imbalance * 0.2))
    diagnostics["checks_total"] += 1

    # Aggregate
    if score_components:
        score = sum(score_components) / len(score_components)
    else:
        score = 0.0

    return score, diagnostics


def check_format(completion: str) -> Tuple[float, Dict[str, Any]]:
    """
    Check format requirements (length, encoding, etc.).

    Returns:
        (score, diagnostics)
    """
    config = get_config()

    diagnostics = {
        "total_length": 0,
        "within_limits": True,
        "encoding_ok": True,
        "excessive_whitespace": False,
    }

    if not completion:
        return 0.0, diagnostics

    diagnostics["total_length"] = len(completion)

    score_components = []

    # 1. Length within limits
    total_len = len(completion.split())
    min_len = 10  # Minimum viable response
    max_len = 500  # Maximum before penalty

    if total_len < min_len:
        length_score = total_len / min_len
    elif total_len <= max_len:
        length_score = 1.0
    else:
        # Soft penalty for exceeding
        excess = (total_len - max_len) / max_len
        length_score = max(0.5, 1.0 - excess * 0.3)

    score_components.append(length_score)
    diagnostics["within_limits"] = min_len <= total_len <= max_len

    # 2. Check for excessive whitespace (possible gaming)
    whitespace_ratio = completion.count(" ") / max(1, len(completion))
    if whitespace_ratio > 0.5:
        diagnostics["excessive_whitespace"] = True
        score_components.append(0.5)
    else:
        score_components.append(1.0)

    # 3. Check for encoding issues
    try:
        completion.encode("utf-8").decode("utf-8")
        score_components.append(1.0)
    except UnicodeError:
        diagnostics["encoding_ok"] = False
        score_components.append(0.5)

    if score_components:
        score = sum(score_components) / len(score_components)
    else:
        score = 0.0

    return score, diagnostics


def check_completeness(completion: str) -> Tuple[float, Dict[str, Any]]:
    """
    Check if completion is complete (not truncated).

    Returns:
        (score, diagnostics)
    """
    diagnostics = {
        "is_complete": False,
        "reason": "",
        "confidence": 0.0,
    }

    if not completion:
        return 0.0, diagnostics

    # Check overall completion
    is_complete, reason, confidence = check_completion(completion)

    diagnostics["is_complete"] = is_complete
    diagnostics["reason"] = reason
    diagnostics["confidence"] = confidence

    # Extract sections and check each
    thinking, answer = extract_sections(completion)

    # Check if thinking is complete
    if thinking:
        think_complete, think_reason, think_conf = check_completion(thinking)
        diagnostics["thinking_complete"] = think_complete
        diagnostics["thinking_reason"] = think_reason
    else:
        think_complete = True
        think_conf = 0.5

    # Check if answer is complete
    if answer:
        answer_complete, answer_reason, answer_conf = check_completion(answer)
        diagnostics["answer_complete"] = answer_complete
        diagnostics["answer_reason"] = answer_reason
    else:
        answer_complete = False
        answer_conf = 0.0

    # Combine scores
    # Answer completion is more important
    if answer:
        score = 0.3 * think_conf + 0.7 * answer_conf
    else:
        score = 0.5 * confidence  # No answer section

    return score, diagnostics


def foundation_check(completion: str) -> Tuple[float, Dict[str, Any]]:
    """
    Complete foundation level check.

    Combines structure, format, and completeness checks.

    Returns:
        (score, diagnostics) with soft score in [0, 1]
    """
    config = get_config()

    # Run all checks
    structure_score, structure_diag = check_structure(
        completion,
        require_think=config.require_think_tags,
        require_answer=config.require_answer_section,
    )

    format_score, format_diag = check_format(completion)

    completeness_score, completeness_diag = check_completeness(completion)

    # Combine diagnostics
    diagnostics = {
        "structure": structure_diag,
        "format": format_diag,
        "completeness": completeness_diag,
        "component_scores": {
            "structure": structure_score,
            "format": format_score,
            "completeness": completeness_score,
        },
    }

    # Copy key fields to top level for convenience
    diagnostics["has_think_tags"] = (
        structure_diag["has_think_open"] and structure_diag["has_think_close"]
    )
    diagnostics["has_answer"] = structure_diag["has_answer"]
    diagnostics["is_complete"] = completeness_diag.get("is_complete", False)
    diagnostics["answer_complete"] = completeness_diag.get("answer_complete", False)
    diagnostics["checks_passed"] = structure_diag["checks_passed"]
    diagnostics["checks_total"] = structure_diag["checks_total"]

    # Weight components
    # Structure is most important, then completeness, then format
    weights = {
        "structure": 0.50,
        "format": 0.15,
        "completeness": 0.35,
    }

    final_score = (
        weights["structure"] * structure_score
        + weights["format"] * format_score
        + weights["completeness"] * completeness_score
    )

    # Ensure minimum score if there's any content
    if completion and len(completion.strip()) > 20:
        final_score = max(0.1, final_score)

    diagnostics["final_score"] = final_score

    return final_score, diagnostics


def create_foundation_result(
    completion: str,
    gate_config: Optional[Any] = None,
) -> LevelResult:
    """
    Create a complete LevelResult for foundation level.

    Includes soft gating computation.
    """
    config = get_config()
    gate = gate_config or config.foundation_gate

    # Get raw score
    raw_score, diagnostics = foundation_check(completion)

    # Apply soft gate
    gate_value = gate.compute_gate(raw_score)
    gated_score = raw_score * gate_value

    # Create result
    result = LevelResult(
        level="foundation",
        raw_score=raw_score,
        gated_score=gated_score,
        gate_value=gate_value,
        passed_soft_gate=gate_value > 0.5,
        diagnostics=diagnostics,
    )

    # Add components
    result.add_component(
        "structure",
        diagnostics["component_scores"]["structure"],
        0.50,
        diagnostics["structure"],
    )
    result.add_component(
        "format", diagnostics["component_scores"]["format"], 0.15, diagnostics["format"]
    )
    result.add_component(
        "completeness",
        diagnostics["component_scores"]["completeness"],
        0.35,
        diagnostics["completeness"],
    )

    return result


def compute_foundation_reward(
    response: str, expected: str, question: str, config: Any, **kwargs
) -> "SimpleRewardResult":
    """
    Compute foundation reward with simple result interface.

    Args:
        response: Model response
        expected: Expected answer (not used for foundation)
        question: Original question (not used for foundation)
        config: RewardConfig
        **kwargs: Additional arguments (ignored)

    Returns:
        SimpleRewardResult with score, components, and details
    """
    from ..core.base import ComponentResult
    from dataclasses import dataclass, field
    from typing import List, Dict, Any

    @dataclass
    class SimpleRewardResult:
        """Simple reward result for inter-level communication."""

        score: float
        components: List[ComponentResult] = field(default_factory=list)
        details: Dict[str, Any] = field(default_factory=dict)

    # Get the foundation check result
    score, diagnostics = foundation_check(response)

    # Build components from diagnostics
    components = []
    if "component_scores" in diagnostics:
        for comp_name, comp_score in diagnostics["component_scores"].items():
            weight = {"structure": 0.50, "format": 0.15, "completeness": 0.35}.get(
                comp_name, 0.33
            )
            components.append(
                ComponentResult(
                    name=comp_name,
                    raw_score=comp_score,
                    weight=weight,
                    weighted_score=comp_score * weight,
                    diagnostics=diagnostics.get(comp_name, {}),
                )
            )

    return SimpleRewardResult(score=score, components=components, details=diagnostics)
