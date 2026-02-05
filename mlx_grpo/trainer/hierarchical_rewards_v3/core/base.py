"""
Reward System Data Structures
=============================

Comprehensive data structures for reward computation results,
diagnostics, and batch-level analysis.

Design Principles:
    - Immutable results for thread safety
    - Rich diagnostics for debugging
    - Serialization support for logging
    - Hierarchical structure matching reward levels
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RewardLevel(Enum):
    """Reward levels for categorization."""

    FOUNDATION = "foundation"
    CORRECTNESS = "correctness"
    QUALITY = "quality"
    POLISH = "polish"


@dataclass
class ComponentResult:
    """
    Result from a single reward component.

    Components are the atomic units of reward computation,
    like "factual_accuracy" or "reasoning_depth".
    """

    name: str
    raw_score: float
    weight: float
    weighted_score: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def contribution(self) -> float:
        """Contribution to parent level score."""
        return self.weighted_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_score": round(self.raw_score, 4),
            "weight": round(self.weight, 4),
            "weighted_score": round(self.weighted_score, 4),
            "diagnostics": self.diagnostics,
        }


@dataclass
class LevelResult:
    """
    Result from a complete reward level.

    Levels aggregate multiple components and apply gating.
    """

    level: str
    raw_score: float  # Before gating
    gated_score: float  # After soft gating
    gate_value: float  # The gate multiplier applied
    passed_soft_gate: bool  # Whether gate > 0.5
    components: Dict[str, ComponentResult] = field(default_factory=dict)
    upstream_gate: float = 1.0  # Gate from previous level
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def add_component(
        self,
        name: str,
        raw_score: float,
        weight: float,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a component result."""
        weighted = raw_score * weight
        self.components[name] = ComponentResult(
            name=name,
            raw_score=raw_score,
            weight=weight,
            weighted_score=weighted,
            diagnostics=diagnostics or {},
        )

    @property
    def component_scores(self) -> Dict[str, float]:
        """Get all component raw scores."""
        return {name: comp.raw_score for name, comp in self.components.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "raw_score": round(self.raw_score, 4),
            "gated_score": round(self.gated_score, 4),
            "gate_value": round(self.gate_value, 4),
            "passed_soft_gate": self.passed_soft_gate,
            "upstream_gate": round(self.upstream_gate, 4),
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "diagnostics": self.diagnostics,
        }


@dataclass
class DiagnosticInfo:
    """
    Detailed diagnostic information for analysis.

    Captures all intermediate values for debugging
    and training analysis.
    """

    # Input information
    completion_length: int = 0
    thinking_length: int = 0
    answer_length: int = 0
    expected_length: int = 0

    # Structural analysis
    has_think_tags: bool = False
    has_answer_section: bool = False
    is_complete: bool = False
    tag_balance: int = 0

    # Content analysis
    unique_token_ratio: float = 0.0
    trigram_repetition: float = 0.0
    phrase_repetition: float = 0.0
    information_density: float = 0.0

    # Reasoning analysis
    reasoning_steps: int = 0
    reasoning_depth: int = 0
    branch_count: int = 0

    # Match analysis
    answer_similarity: float = 0.0
    keyword_overlap: float = 0.0
    numerical_match: bool = False
    exact_match: bool = False

    # Anti-gaming flags
    repetition_flag: bool = False
    keyword_stuffing_flag: bool = False
    template_flag: bool = False
    length_gaming_flag: bool = False

    # Timing (optional)
    computation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Completion: {self.completion_length} chars "
            f"(think: {self.thinking_length}, answer: {self.answer_length})",
            f"Structure: tags={self.has_think_tags}, answer={self.has_answer_section}, "
            f"complete={self.is_complete}",
            f"Content: unique_ratio={self.unique_token_ratio:.2f}, "
            f"info_density={self.information_density:.2f}",
            f"Reasoning: steps={self.reasoning_steps}, depth={self.reasoning_depth}",
            f"Match: similarity={self.answer_similarity:.2f}, "
            f"exact={self.exact_match}, numerical={self.numerical_match}",
        ]

        flags = []
        if self.repetition_flag:
            flags.append("repetition")
        if self.keyword_stuffing_flag:
            flags.append("keyword_stuffing")
        if self.template_flag:
            flags.append("template")
        if self.length_gaming_flag:
            flags.append("length_gaming")

        if flags:
            lines.append(f"Flags: {', '.join(flags)}")

        return "\n".join(lines)


@dataclass
class AntiGamingResult:
    """
    Results from anti-gaming detection.

    Tracks all detected gaming attempts and their penalties.
    """

    # Detection flags
    repetition_detected: bool = False
    keyword_stuffing_detected: bool = False
    template_detected: bool = False
    length_gaming_detected: bool = False
    low_diversity_detected: bool = False
    single_dominance_detected: bool = False

    # Measured values
    repetition_ratio: float = 0.0
    keyword_density: float = 0.0
    template_similarity: float = 0.0
    diversity_score: float = 1.0
    dominance_ratio: float = 0.0

    # Penalties (individual)
    repetition_penalty: float = 0.0
    keyword_penalty: float = 0.0
    template_penalty: float = 0.0
    length_penalty: float = 0.0
    diversity_penalty: float = 0.0
    dominance_penalty: float = 0.0

    # Total
    total_penalty: float = 0.0

    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    def add_penalty(self, source: str, penalty: float) -> None:
        """Add a penalty from a source."""
        setattr(self, f"{source}_penalty", penalty)
        self.total_penalty = min(0.8, self.total_penalty + penalty)
        self.details[f"{source}_penalty"] = penalty

    @property
    def any_gaming_detected(self) -> bool:
        """Check if any gaming was detected."""
        return any(
            [
                self.repetition_detected,
                self.keyword_stuffing_detected,
                self.template_detected,
                self.length_gaming_detected,
                self.low_diversity_detected,
                self.single_dominance_detected,
            ]
        )

    @property
    def multiplier(self) -> float:
        """Get the score multiplier (1 - penalty)."""
        return max(0.2, 1.0 - self.total_penalty)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "any_gaming_detected": self.any_gaming_detected,
            "repetition": {
                "detected": self.repetition_detected,
                "ratio": round(self.repetition_ratio, 4),
                "penalty": round(self.repetition_penalty, 4),
            },
            "keyword_stuffing": {
                "detected": self.keyword_stuffing_detected,
                "density": round(self.keyword_density, 4),
                "penalty": round(self.keyword_penalty, 4),
            },
            "template": {
                "detected": self.template_detected,
                "similarity": round(self.template_similarity, 4),
                "penalty": round(self.template_penalty, 4),
            },
            "length_gaming": {
                "detected": self.length_gaming_detected,
                "penalty": round(self.length_penalty, 4),
            },
            "diversity": {
                "detected": self.low_diversity_detected,
                "score": round(self.diversity_score, 4),
                "penalty": round(self.diversity_penalty, 4),
            },
            "dominance": {
                "detected": self.single_dominance_detected,
                "ratio": round(self.dominance_ratio, 4),
                "penalty": round(self.dominance_penalty, 4),
            },
            "total_penalty": round(self.total_penalty, 4),
            "multiplier": round(self.multiplier, 4),
            "details": self.details,
        }


@dataclass
class RewardResult:
    """
    Complete result from hierarchical reward computation.

    Contains all level results, diagnostics, and the final score.
    """

    # Final scores
    final_score: float = 0.0  # After all processing
    raw_score: float = 0.0  # Before anti-gaming
    scaled_score: float = 0.0  # After scaling

    # Level results
    foundation: Optional[LevelResult] = None
    correctness: Optional[LevelResult] = None
    quality: Optional[LevelResult] = None
    polish: Optional[LevelResult] = None

    # Anti-gaming
    anti_gaming: Optional[AntiGamingResult] = None

    # Diagnostics
    diagnostics: Optional[DiagnosticInfo] = None

    # Input references (truncated for logging)
    _completion_preview: str = ""
    _expected_preview: str = ""

    # Metadata
    computation_time_ms: float = 0.0
    config_hash: str = ""

    def get_level(self, level: str) -> Optional[LevelResult]:
        """Get result for a specific level."""
        return getattr(self, level.lower(), None)

    def get_all_component_scores(self) -> Dict[str, float]:
        """Get all component scores across all levels."""
        scores = {}
        for level_name in ["foundation", "correctness", "quality", "polish"]:
            level = self.get_level(level_name)
            if level:
                for name, comp in level.components.items():
                    scores[f"{level_name}.{name}"] = comp.raw_score
        return scores

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "REWARD RESULT SUMMARY",
            "=" * 60,
            f"Final Score: {self.final_score:.4f}",
            f"Raw Score:   {self.raw_score:.4f}",
            f"Scaled:      {self.scaled_score:.4f}",
            "-" * 60,
        ]

        for level_name in ["foundation", "correctness", "quality", "polish"]:
            level = self.get_level(level_name)
            if level:
                lines.append(
                    f"{level_name.upper():12s}: "
                    f"raw={level.raw_score:.3f}, "
                    f"gated={level.gated_score:.3f}, "
                    f"gate={level.gate_value:.3f}"
                )
                for name, comp in level.components.items():
                    lines.append(f"  - {name}: {comp.raw_score:.3f} (w={comp.weight:.2f})")

        if self.anti_gaming and self.anti_gaming.any_gaming_detected:
            lines.append("-" * 60)
            lines.append(f"ANTI-GAMING: penalty={self.anti_gaming.total_penalty:.3f}")
            if self.anti_gaming.repetition_detected:
                lines.append(f"  - Repetition: {self.anti_gaming.repetition_ratio:.3f}")
            if self.anti_gaming.keyword_stuffing_detected:
                lines.append(f"  - Keyword stuffing: {self.anti_gaming.keyword_density:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_score": round(self.final_score, 4),
            "raw_score": round(self.raw_score, 4),
            "scaled_score": round(self.scaled_score, 4),
            "foundation": self.foundation.to_dict() if self.foundation else None,
            "correctness": self.correctness.to_dict() if self.correctness else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "polish": self.polish.to_dict() if self.polish else None,
            "anti_gaming": self.anti_gaming.to_dict() if self.anti_gaming else None,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "computation_time_ms": round(self.computation_time_ms, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class BatchResult:
    """
    Results for a batch of completions.

    Used for batch-level analysis and diversity checking.
    """

    results: List[RewardResult] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    adjusted_scores: List[float] = field(default_factory=list)

    # Batch statistics
    diversity_score: float = 0.0
    diversity_passed: bool = True

    # Score statistics
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    score_spread: float = 0.0

    # Ranking information
    rankings: List[int] = field(default_factory=list)
    ranking_adjusted: bool = False

    def compute_statistics(self) -> None:
        """Compute batch statistics from scores."""
        if not self.scores:
            return

        import numpy as np

        scores_arr = np.array(self.scores)

        self.mean_score = float(np.mean(scores_arr))
        self.std_score = float(np.std(scores_arr))
        self.min_score = float(np.min(scores_arr))
        self.max_score = float(np.max(scores_arr))
        self.score_spread = self.max_score - self.min_score

        # Compute rankings (higher score = lower rank number)
        self.rankings = list((-scores_arr).argsort().argsort() + 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": len(self.results),
            "diversity_score": round(self.diversity_score, 4),
            "diversity_passed": self.diversity_passed,
            "statistics": {
                "mean": round(self.mean_score, 4),
                "std": round(self.std_score, 4),
                "min": round(self.min_score, 4),
                "max": round(self.max_score, 4),
                "spread": round(self.score_spread, 4),
            },
            "scores": [round(s, 4) for s in self.scores],
            "adjusted_scores": [round(s, 4) for s in self.adjusted_scores],
            "rankings": self.rankings,
            "ranking_adjusted": self.ranking_adjusted,
            "results": [r.to_dict() for r in self.results],
        }


# Type aliases for clarity
ScoreDict = Dict[str, float]
DiagDict = Dict[str, Any]
RewardTuple = Tuple[float, DiagDict]
