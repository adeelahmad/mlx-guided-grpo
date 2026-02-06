# Type System V2 - SOLID Architecture

## Overview

The Type System V2 provides a clean, extensible architecture for handling different task types (tool calling, MCQ/exam, general Q&A) in the training pipeline.

**Architecture**: EventBus + Metaclass auto-registration + Observer hooks + Template Method pattern

```
TypeCoordinator (singleton, event bus owner)
   |
   |-- EventBus (publish/subscribe for lifecycle events)
   |
   +-- BaseReward ─────────> ToolCallReward, MCQReward, GeneralQNAReward
   +-- BaseDatasetLoader ──> ToolCallDatasetLoader, MCQDatasetLoader, GeneralQNADatasetLoader
   +-- BaseRolloutGenerator
          |
          +-- ToolCallRolloutGenerator  (no phase recovery, function scaffolding)
          |
          +-- ThinkingBasedGenerator    (shared: curriculum, completeness, phase recovery)
                 |
                 +-- MCQRolloutGenerator       (exam-specific recovery, longer generation)
                 +-- GeneralQNARolloutGenerator (default settings)
```

## Built-in Types

| Type | Canonical Name | Aliases | Description |
|------|---------------|---------|-------------|
| Tool Call | `tool_call` | `function_call`, `tool`, `function`, `hermes` | Function/API calling tasks |
| MCQ | `mcq` | `exam`, `aime`, `multiple_choice`, `letter_answer` | Multiple choice & exam tasks |
| General QNA | `general_qna` | `math`, `reasoning`, `general`, `qa`, `None` | Default for all other tasks |

## Quick Start

### Using the Bridge (Recommended)

```python
from mlx_grpo.trainer.type_system_v2 import create_v2_coordinator, v2_reward_adapter

# Create coordinator with all built-in types
coordinator = create_v2_coordinator(tokenizer)

# Get a reward function matching the old pipeline signature
reward_func = v2_reward_adapter(coordinator)

# Use it - dispatches to type-specific rewards automatically
scores = reward_func(prompts, completions, answers, types=["tool_call", "mcq", "general_qna"])
```

### Using Components Directly

```python
from mlx_grpo.trainer.type_system_v2 import TypeCoordinator, auto_register_builtin_types

coordinator = TypeCoordinator()
auto_register_builtin_types(coordinator, tokenizer)

# Get type-specific reward
reward = coordinator.get_reward("tool_call")
scores = reward.compute(prompts, completions, answers)

# Get generation config
generator = coordinator.get_generator("mcq")
config = generator.get_generation_config()
print(config.max_length)       # 1536
print(config.two_phase)        # True
print(config.enforce_thinking) # True
```

## Creating a Custom Type

Implement three classes - one for each concern:

```python
from mlx_grpo.trainer.type_system_v2 import BaseReward, BaseDatasetLoader, BaseRolloutGenerator

class CodeReward(BaseReward):
    type_name = "code"

    def get_component_weights(self):
        return {"correctness": 0.5, "style": 0.3, "efficiency": 0.2}

    def validate_completion(self, completion, type_info=None):
        if not completion.strip():
            return False, "Empty completion"
        return True, None

    def compute_single(self, prompt, completion, answer, type_info=None):
        # Your scoring logic
        ...

class CodeDatasetLoader(BaseDatasetLoader):
    type_name = "code"

    def validate_sample(self, sample):
        if "prompt" not in sample or "answer" not in sample:
            return False, "Missing fields"
        return True, None

    def preprocess_sample(self, sample):
        return sample

    def get_system_prompt(self, sample):
        return "You are a coding assistant."

class CodeRolloutGenerator(BaseRolloutGenerator):
    type_name = "code"

    def get_generation_config(self):
        return GenerationConfig(max_length=1024, temperature=0.7)

    def apply_curriculum(self, answer, ratio):
        return answer[:int(len(answer) * ratio)]

    def is_generation_complete(self, text, phase):
        return True, "complete"
```

Register with the coordinator:

```python
coordinator.register(
    "code",
    reward=CodeReward(),
    loader=CodeDatasetLoader(tokenizer),
    generator=CodeRolloutGenerator(),
)
```

## Event Bus

Cross-cutting concerns (logging, metrics) without coupling:

```python
from mlx_grpo.trainer.type_system_v2 import EventBus, REWARD_COMPUTED, REWARD_INVALID

bus = EventBus()

def on_reward(event):
    print(f"Reward computed: {event.data}")

bus.subscribe(REWARD_COMPUTED, on_reward)
bus.subscribe(REWARD_INVALID, lambda e: print(f"Invalid: {e.data['reason']}"))
```

### Event Types

| Event | Published By | Data |
|-------|-------------|------|
| `sample.validated` | DatasetLoader | `{valid, reason}` |
| `sample.loaded` | DatasetLoader | `{count}` |
| `reward.computed` | Reward | `{mean_score, count}` |
| `reward.invalid` | Reward | `{reason, completion_preview}` |
| `generation.started` | Generator | `{config}` |
| `generation.completed` | Generator | `{num_results}` |
| `type.registered` | Coordinator | `{type_name, components}` |

## Type Normalization

All type aliases are automatically normalized:

```python
from mlx_grpo.trainer.type_system_v2 import normalize_type

normalize_type("function_call")   # → "tool_call"
normalize_type("exam")            # → "mcq"
normalize_type("math")            # → "general_qna"
normalize_type(None)              # → "general_qna"
normalize_type({"type": "tool"})  # → "tool_call"
```

## Generation Pipeline

Each type has a distinct generation workflow managed by its generator:

| Type | Max Length | Temperature | Two-Phase | Phase Recovery | Curriculum |
|------|-----------|-------------|-----------|----------------|------------|
| `tool_call` | 256 | 0.7 | No | No | Function scaffolding |
| `mcq` | 1536 | 0.85 | Yes | Yes (exam-specific) | Thinking prefix |
| `general_qna` | 1024 | 0.8 | Yes | Yes (standard) | Thinking prefix |

### ThinkingBasedGenerator

MCQ and GeneralQNA share the `ThinkingBasedGenerator` intermediate class:
- **Curriculum**: Extracts `<think>...</think>` content and provides proportional prefix
- **Completeness**: Checks for `</think>` + answer content
- **Phase Recovery**: Injects `</think>\n\boxed{` for incomplete outputs
- **Token Masking**: Injected tokens tracked for loss exclusion

MCQ overrides `check_incomplete()` with exam-specific recovery (probabilistic, boxed injection).

### Type-Dispatched Generation

When `TypeCoordinator` is provided, `generate_grpo()` delegates:
- **Curriculum** → `generator.apply_curriculum(answer, ratio)` per type
- **Phase recovery decision** → `generator.needs_phase_recovery()` per type
- **Incompleteness checking** → `generator.check_incomplete(...)` per type

## Pipeline Integration

The v2 type system is wired into the training pipeline automatically:

1. `train.py` creates a `TypeCoordinator` via `create_v2_coordinator(tokenizer)`
2. Registers `type_aware_strict` / `type_aware_reward` as backward-compatible aliases
3. Passes coordinator to `train_grpo()` as `type_coordinator`
4. `train_grpo()` prepends the v2 type-dispatched reward (unless already present)
5. `generate_grpo()` uses coordinator for type-dispatched curriculum and phase recovery
6. Samples are grouped by type and scored by the appropriate reward

No configuration needed - it works out of the box.
