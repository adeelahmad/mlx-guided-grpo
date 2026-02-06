"""
Type System V2 - Clean SOLID Architecture
==========================================

Extensible type system following SOLID principles with proper separation of concerns.

Architecture:
- EventBus: Lightweight pub/sub for lifecycle events
- TypeComponentMeta: Auto-registration via metaclass
- BaseReward / BaseDatasetLoader / BaseRolloutGenerator: Abstract base classes
- TypeCoordinator: Central orchestration

Built-in Types:
- tool_call: Tool/function calling tasks
- mcq: Multiple choice / exam tasks
- general_qna: General Q&A and reasoning (default)

Usage:
    from mlx_grpo.trainer.type_system_v2 import TypeCoordinator, auto_register_builtin_types

    coordinator = TypeCoordinator()
    auto_register_builtin_types(coordinator, tokenizer)

    reward = coordinator.get_reward("tool_call")
    scores = reward.compute(prompts, completions, answers)

Bridge (backward-compatible adapter):
    from mlx_grpo.trainer.type_system_v2.bridge import create_v2_coordinator, v2_reward_adapter

    coordinator = create_v2_coordinator(tokenizer)
    reward_func = v2_reward_adapter(coordinator)
    scores = reward_func(prompts, completions, answers, types=types)
"""

# Event system
from .events import (
    Event,
    EventBus,
    SAMPLE_VALIDATED,
    SAMPLE_LOADED,
    REWARD_COMPUTED,
    REWARD_INVALID,
    GENERATION_STARTED,
    GENERATION_COMPLETED,
    PHASE_COMPLETED,
    CURRICULUM_APPLIED,
    TYPE_REGISTERED,
)

# Metaclass
from .meta import (
    TypeComponentMeta,
    auto_register_pending,
    get_pending_types,
)

# Core coordinator
from .coordinator import (
    TypeCoordinator,
    TypeComponents,
    auto_register_builtin_types,
    normalize_type,
)

# Base classes (for custom implementations)
from .base.reward import (
    BaseReward,
    RewardResult,
    RewardHooks,
)

from .base.dataset_loader import (
    BaseDatasetLoader,
    DataSample,
    DatasetHooks,
    LoadedDataset,
)

from .base.rollout_generator import (
    BaseRolloutGenerator,
    GenerationConfig,
    GenerationResult,
    GeneratorHooks,
)

# Built-in implementations - Tool Call
from .rewards.tool_call import (
    ToolCallReward,
    ToolCallRewardStrict,
    ToolCallLoggingHook,
)

from .loaders.tool_call import (
    ToolCallDatasetLoader,
    ToolCallCleaningHook,
)

from .generators.tool_call import (
    ToolCallRolloutGenerator,
    ToolCallStrictHook,
)

# Intermediate base class for thinking-based generators
from .generators.thinking_based import ThinkingBasedGenerator

# Built-in implementations - MCQ
from .rewards.mcq import MCQReward
from .loaders.mcq import MCQDatasetLoader
from .generators.mcq import MCQRolloutGenerator

# Built-in implementations - General QNA
from .rewards.general_qna import GeneralQNAReward
from .loaders.general_qna import GeneralQNADatasetLoader
from .generators.general_qna import GeneralQNARolloutGenerator

# Bridge
from .bridge import (
    create_v2_coordinator,
    v2_reward_adapter,
    v2_type_normalizer,
)

__version__ = "2.1.0"

__all__ = [
    # Event system
    "Event",
    "EventBus",
    "SAMPLE_VALIDATED",
    "SAMPLE_LOADED",
    "REWARD_COMPUTED",
    "REWARD_INVALID",
    "GENERATION_STARTED",
    "GENERATION_COMPLETED",
    "PHASE_COMPLETED",
    "CURRICULUM_APPLIED",
    "TYPE_REGISTERED",

    # Metaclass
    "TypeComponentMeta",
    "auto_register_pending",
    "get_pending_types",

    # Coordinator
    "TypeCoordinator",
    "TypeComponents",
    "auto_register_builtin_types",
    "normalize_type",

    # Base classes
    "BaseReward",
    "RewardResult",
    "RewardHooks",
    "BaseDatasetLoader",
    "DataSample",
    "DatasetHooks",
    "LoadedDataset",
    "BaseRolloutGenerator",
    "GenerationConfig",
    "GenerationResult",
    "GeneratorHooks",

    # Tool call implementations
    "ToolCallReward",
    "ToolCallRewardStrict",
    "ToolCallLoggingHook",
    "ToolCallDatasetLoader",
    "ToolCallCleaningHook",
    "ToolCallRolloutGenerator",
    "ToolCallStrictHook",

    # Intermediate base
    "ThinkingBasedGenerator",

    # MCQ implementations
    "MCQReward",
    "MCQDatasetLoader",
    "MCQRolloutGenerator",

    # General QNA implementations
    "GeneralQNAReward",
    "GeneralQNADatasetLoader",
    "GeneralQNARolloutGenerator",

    # Bridge
    "create_v2_coordinator",
    "v2_reward_adapter",
    "v2_type_normalizer",
]
