"""
Reward System Configuration
===========================

Central configuration with soft gating support and calibrated thresholds
designed for GRPO training with limited generation length.

Key Design Principles:
    1. Soft gating - gradients always flow
    2. Calibrated thresholds for 450-token generations
    3. Multi-scale weighting
    4. Information-theoretic grounding
"""

import os
import math
import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Dict, Any, Tuple
import threading

logger = logging.getLogger(__name__)


class RewardLevel(IntEnum):
    """
    Hierarchical reward levels with priority ordering.

    Uses IntEnum for comparison and ordering operations.
    Lower values = higher priority (must pass first).
    """

    FOUNDATION = 1  # Structure and format requirements
    CORRECTNESS = 2  # Factual accuracy and answer matching
    QUALITY = 3  # Reasoning depth and coherence
    POLISH = 4  # Style and presentation

    @classmethod
    def from_string(cls, name: str) -> "RewardLevel":
        name = name.upper().strip()
        try:
            return cls[name]
        except KeyError:
            valid = [level.name for level in cls]
            raise ValueError(f"Invalid reward level: {name}. Valid: {valid}")

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class GateConfig:
    """
    Soft gate configuration for a single level.

    Soft gating formula:
        gate_value = sigmoid(steepness * (score - threshold))
        output = max(floor, score * gate_value)

    This ensures:
        1. Gradients always flow (floor > 0)
        2. Smooth transition around threshold
        3. No sudden jumps in reward landscape
    """

    threshold: float = 0.5  # Center of sigmoid transition
    steepness: float = 10.0  # How sharp the transition is
    floor: float = 0.05  # Minimum output (ensures gradient flow)
    ceiling: float = 1.0  # Maximum output
    weight: float = 1.0  # Level weight in final aggregation

    def compute_gate(self, score: float) -> float:
        """
        Compute soft gate value using sigmoid.

        Returns value in [floor, ceiling].
        """
        # Sigmoid centered at threshold
        x = self.steepness * (score - self.threshold)
        # Clip to prevent overflow
        x = max(-20.0, min(20.0, x))
        gate = 1.0 / (1.0 + math.exp(-x))

        # Scale to [floor, ceiling]
        return self.floor + (self.ceiling - self.floor) * gate

    def apply_gate(self, score: float, upstream_gate: float = 1.0) -> float:
        """
        Apply soft gate to a score, with upstream gating.

        Args:
            score: Raw score for this level
            upstream_gate: Gate value from previous level (cascading)

        Returns:
            Gated score in [floor * upstream, ceiling * upstream]
        """
        gate = self.compute_gate(score)
        return score * gate * upstream_gate


@dataclass
class RewardConfig:
    """
    Central configuration for the hierarchical reward system.

    Calibrated for:
        - Max generation: 450 tokens
        - Group size: 2
        - GRPO training (needs ranking signal)

    Design Philosophy:
        - Soft gating ensures gradient flow
        - Multi-scale analysis captures different aspects
        - Information-theoretic measures resist gaming
        - Relative scoring for GRPO compatibility
    """

    # ==========================================
    # LEVEL 1: FOUNDATION CONFIGURATION
    # ==========================================
    # Relaxed for 450-token limit
    require_think_tags: bool = True
    require_answer_section: bool = True
    min_thinking_tokens: int = 20  # Reduced from higher values
    max_thinking_tokens: int = 350  # Leave room for answer
    min_answer_tokens: int = 5  # Very short answers OK
    max_answer_tokens: int = 200  # Reasonable answer length
    require_complete_tags: bool = True

    # Soft gate for foundation
    foundation_gate: GateConfig = field(
        default_factory=lambda: GateConfig(
            threshold=0.4,  # Lowered from 0.8 - more forgiving
            steepness=8.0,  # Moderate transition
            floor=0.15,  # Always give some credit
            ceiling=1.0,
            weight=0.10,
        )
    )

    # ==========================================
    # LEVEL 2: CORRECTNESS CONFIGURATION
    # ==========================================
    # Factual accuracy weights
    factual_weight: float = 0.35
    semantic_weight: float = 0.25
    answer_match_weight: float = 0.20
    numerical_tolerance: float = 0.02  # 2% tolerance for numbers

    # Soft gate for correctness
    correctness_gate: GateConfig = field(
        default_factory=lambda: GateConfig(
            threshold=0.10,  # Low threshold - want quality signal even with partial correctness
            steepness=6.0,
            floor=0.10,
            ceiling=1.0,
            weight=0.45,
        )
    )

    # ==========================================
    # LEVEL 3: QUALITY CONFIGURATION
    # ==========================================
    # Reasoning quality weights
    reasoning_depth_weight: float = 0.25
    reasoning_coherence_weight: float = 0.20
    efficiency_weight: float = 0.15
    step_clarity_weight: float = 0.15

    # Information-theoretic thresholds
    min_information_density: float = 0.3  # Bits per token
    max_redundancy_ratio: float = 0.4  # Max repetition

    # Soft gate for quality
    quality_gate: GateConfig = field(
        default_factory=lambda: GateConfig(
            threshold=0.08, steepness=5.0, floor=0.08, ceiling=1.0, weight=0.30
        )
    )

    # ==========================================
    # LEVEL 4: POLISH CONFIGURATION
    # ==========================================
    style_weight: float = 0.10
    format_weight: float = 0.05

    # Soft gate for polish
    polish_gate: GateConfig = field(
        default_factory=lambda: GateConfig(
            threshold=0.05, steepness=4.0, floor=0.05, ceiling=1.0, weight=0.15
        )
    )

    # ==========================================
    # ANTI-GAMING CONFIGURATION
    # ==========================================
    # Calibrated for real outputs, not oversensitive
    enable_anti_gaming: bool = True

    # Repetition detection (from static analysis: clone detection)
    max_trigram_repetition: float = 0.25  # Max ratio of repeated trigrams
    max_phrase_repetition: float = 0.20  # Max ratio of repeated phrases (4+ words)
    repetition_penalty_scale: float = 0.3  # How much to penalize

    # Information density (from static analysis: Halstead complexity)
    min_unique_token_ratio: float = 0.35  # Vocabulary richness
    min_information_gain: float = 0.1  # Bits per new token

    # Structural complexity (from static analysis: cyclomatic complexity)
    min_reasoning_branches: int = 1  # At least some conditional reasoning
    max_linear_ratio: float = 0.85  # Not purely linear

    # Pattern exploitation
    max_template_similarity: float = 0.70  # Allow some structure
    template_penalty_scale: float = 0.2

    # Keyword stuffing
    max_keyword_density: float = 0.20  # Raised from 0.15
    keyword_penalty_scale: float = 0.2

    # Length normalization
    length_norm_target: int = 300  # Target length for normalization
    length_penalty_scale: float = 0.1  # Mild length penalty

    # ==========================================
    # BATCH & DIVERSITY CONFIGURATION
    # ==========================================
    enable_batch_diversity: bool = True
    min_batch_diversity: float = 0.15  # Lowered for group_size=2
    diversity_penalty_scale: float = 0.10

    # Ensure ranking signal for GRPO
    ensure_ranking: bool = True
    min_score_spread: float = 0.05  # Minimum difference for ranking
    ranking_epsilon: float = 0.02  # Small adjustment to ensure spread

    # ==========================================
    # SCORE SCALING CONFIGURATION
    # ==========================================
    # Map raw scores to better range for training
    score_scaling: str = "sigmoid"  # "linear", "sigmoid", "sqrt"
    score_scale_center: float = 0.3  # Center for sigmoid scaling
    score_scale_spread: float = 0.4  # Spread for sigmoid scaling
    min_output_score: float = 0.02  # Never go below this
    max_output_score: float = 0.95  # Never go above this

    # ==========================================
    # OPERATIONAL SETTINGS
    # ==========================================
    log_diagnostics: bool = True
    log_level: str = "DEBUG"
    strict_validation: bool = False  # Warn instead of fail
    cache_embeddings: bool = True

    def __post_init__(self):
        """Apply environment overrides and validate."""
        self._apply_env_overrides()
        self._validate()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            "REWARD_FOUNDATION_THRESHOLD": ("foundation_gate.threshold", float),
            "REWARD_CORRECTNESS_THRESHOLD": ("correctness_gate.threshold", float),
            "REWARD_ENABLE_ANTI_GAMING": (
                "enable_anti_gaming",
                lambda x: x.lower() in ("1", "true"),
            ),
            "REWARD_LOG_DIAGNOSTICS": (
                "log_diagnostics",
                lambda x: x.lower() in ("1", "true"),
            ),
            "REWARD_MIN_BATCH_DIVERSITY": ("min_batch_diversity", float),
        }

        for env_var, (attr_path, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    if "." in attr_path:
                        obj_name, field_name = attr_path.split(".")
                        obj = getattr(self, obj_name)
                        setattr(obj, field_name, converter(value))
                    else:
                        setattr(self, attr_path, converter(value))
                    logger.debug(f"Applied env override: {env_var}={value}")
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Invalid env var {env_var}={value}: {e}")

    def _validate(self) -> None:
        """Validate configuration consistency."""
        errors = []
        warnings = []

        # Check weight sums
        correctness_weights = (
            self.factual_weight + self.semantic_weight + self.answer_match_weight
        )
        if abs(correctness_weights - 0.80) > 0.1:
            warnings.append(
                f"Correctness weights sum to {correctness_weights:.2f}, expected ~0.80"
            )

        quality_weights = (
            self.reasoning_depth_weight
            + self.reasoning_coherence_weight
            + self.efficiency_weight
            + self.step_clarity_weight
        )
        if abs(quality_weights - 0.75) > 0.1:
            warnings.append(
                f"Quality weights sum to {quality_weights:.2f}, expected ~0.75"
            )

        # Check gate configurations
        for gate_name in [
            "foundation_gate",
            "correctness_gate",
            "quality_gate",
            "polish_gate",
        ]:
            gate = getattr(self, gate_name)
            if gate.floor >= gate.ceiling:
                errors.append(
                    f"{gate_name}: floor ({gate.floor}) >= ceiling ({gate.ceiling})"
                )
            if gate.threshold < 0 or gate.threshold > 1:
                errors.append(f"{gate_name}: threshold ({gate.threshold}) not in [0,1]")

        # Check anti-gaming thresholds
        if self.max_trigram_repetition > 0.5:
            warnings.append(
                f"max_trigram_repetition ({self.max_trigram_repetition}) may be too lenient"
            )

        if errors:
            msg = "Configuration errors:\n" + "\n".join(errors)
            if self.strict_validation:
                raise ValueError(msg)
            else:
                logger.error(msg)

        if warnings:
            for w in warnings:
                logger.warning(f"Configuration warning: {w}")

        # logger.debug("Configuration validated successfully")

    def get_gate(self, level: RewardLevel) -> GateConfig:
        """Get gate configuration for a level."""
        gate_map = {
            RewardLevel.FOUNDATION: self.foundation_gate,
            RewardLevel.CORRECTNESS: self.correctness_gate,
            RewardLevel.QUALITY: self.quality_gate,
            RewardLevel.POLISH: self.polish_gate,
        }
        return gate_map[level]

    def get_level_weight(self, level: RewardLevel) -> float:
        """Get weight for a level."""
        return self.get_gate(level).weight

    def get_all_weights(self) -> Dict[RewardLevel, float]:
        """Get all level weights."""
        return {level: self.get_level_weight(level) for level in RewardLevel}

    @property
    def level_weights(self) -> Dict[str, float]:
        """Get level weights as string-keyed dict for easy access."""
        return {
            "foundation": self.foundation_gate.weight,
            "correctness": self.correctness_gate.weight,
            "quality": self.quality_gate.weight,
            "polish": self.polish_gate.weight,
        }

    @property
    def min_score(self) -> float:
        """Minimum output score."""
        return self.min_output_score

    @property
    def max_score(self) -> float:
        """Maximum output score."""
        return self.max_output_score

    def scale_score(self, raw_score: float) -> float:
        """
        Scale raw score to output range using configured method.

        This helps ensure scores are in a good range for GRPO training.
        """
        if self.score_scaling == "linear":
            scaled = raw_score
        elif self.score_scaling == "sigmoid":
            # Sigmoid centered at score_scale_center
            x = (raw_score - self.score_scale_center) / self.score_scale_spread
            x = max(-10.0, min(10.0, x))
            scaled = 1.0 / (1.0 + math.exp(-x))
        elif self.score_scaling == "sqrt":
            scaled = math.sqrt(max(0, raw_score))
        else:
            scaled = raw_score

        # Clamp to output range
        return max(self.min_output_score, min(self.max_output_score, scaled))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, GateConfig):
                result[key] = {
                    "threshold": value.threshold,
                    "steepness": value.steepness,
                    "floor": value.floor,
                    "ceiling": value.ceiling,
                    "weight": value.weight,
                }
            elif isinstance(value, Enum):
                result[key] = value.name
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardConfig":
        """Create config from dictionary."""
        # Handle GateConfig fields
        gate_fields = [
            "foundation_gate",
            "correctness_gate",
            "quality_gate",
            "polish_gate",
        ]
        processed = {}

        for key, value in data.items():
            if key in gate_fields and isinstance(value, dict):
                processed[key] = GateConfig(**value)
            elif key.startswith("_"):
                continue
            else:
                processed[key] = value

        return cls(**processed)


# ==========================================
# Global Configuration Management (Thread-Safe)
# ==========================================

_config_lock = threading.RLock()
_global_config: Optional[RewardConfig] = None


def get_config() -> RewardConfig:
    """
    Get the global reward configuration.

    Thread-safe. Creates default config if none exists.
    """
    global _global_config

    with _config_lock:
        if _global_config is None:
            _global_config = RewardConfig()
            logger.info("Initialized default reward configuration")
        return _global_config


def set_config(config: RewardConfig) -> None:
    """
    Set the global reward configuration.

    Thread-safe.
    """
    global _global_config

    with _config_lock:
        _global_config = config
        logger.info("Updated global reward configuration")


def reset_config() -> None:
    """Reset to default configuration."""
    global _global_config

    with _config_lock:
        _global_config = None
        logger.debug("Reset reward configuration")


def update_config(**kwargs) -> RewardConfig:
    """
    Update specific configuration values.

    Returns the updated config.
    """
    config = get_config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config key: {key}")

    config._validate()
    return config


def get_default_config() -> RewardConfig:
    """
    Create a fresh default configuration.

    Unlike get_config(), this always returns a new instance
    rather than the global singleton.
    """
    return RewardConfig()
