# Hierarchical Rewards v3

A production-grade, multi-hierarchical reward system for GRPO training with soft gating, anti-gaming mechanisms, and guaranteed gradient flow.

## Key Features

- **Soft Gating**: Sigmoid-based gates with floors ensure gradients always flow, eliminating "gradient death" from hard thresholds
- **Four-Level Hierarchy**: Foundation → Correctness → Quality → Polish with cascading gates
- **Anti-Gaming**: Information-theoretic measures (entropy, compression, NCD) resist exploitation
- **Calibrated for Constraints**: Optimized for 450-token limit, group size 2

## Installation

```python
# Add to your project
import sys
sys.path.append('/path/to/hierarchical_rewards_v3')

from hierarchical_rewards_v3 import hierarchical_reward, batch_hierarchical_reward
```

## Quick Start

### Basic Scoring

```python
from hierarchical_rewards_v3 import hierarchical_reward, quick_score

response = """<think>
To solve 6 × 7, I'll use basic multiplication.
6 × 7 = 42
</think>

<answer>42</answer>"""

# Quick score (just the number)
score = quick_score(response, expected="42", question="What is 6 × 7?")
print(f"Score: {score:.4f}")

# Full diagnostics
score, diagnostics = hierarchical_reward(response, expected="42", question="What is 6 × 7?")
print(f"Foundation: {diagnostics['levels']['foundation']['raw_score']:.3f}")
print(f"Correctness: {diagnostics['levels']['correctness']['raw_score']:.3f}")
print(f"Quality: {diagnostics['levels']['quality']['raw_score']:.3f}")
print(f"Polish: {diagnostics['levels']['polish']['raw_score']:.3f}")
```

### Batch Scoring for GRPO

```python
from hierarchical_rewards_v3 import batch_hierarchical_reward

responses = [response1, response2, response3]  # Your model outputs

result = batch_hierarchical_reward(
    responses=responses,
    expected="42",
    question="What is 6 × 7?"
)

# Use scores for GRPO training
scores = result.scores  # List[float]
print(f"Scores: {scores}")
print(f"Spread: {result.batch_stats['spread']:.4f}")
```

## Architecture

### Four-Level Hierarchy

| Level | Weight | Threshold | Floor | Purpose |
|-------|--------|-----------|-------|---------|
| Foundation | 10% | 0.4 | 0.15 | Structure (think tags, answer section) |
| Correctness | 45% | 0.1 | 0.10 | Factual accuracy (multi-method verification) |
| Quality | 30% | 0.08 | 0.08 | Reasoning depth, coherence, efficiency |
| Polish | 15% | 0.05 | 0.05 | Style, format, presentation |

### Soft Gating Formula

```
gate_value = floor + (1 - floor) × sigmoid(steepness × (score - threshold))
gated_score = score × gate_value × upstream_gate
```

This ensures:
- Gradients always flow (minimum `floor` contribution)
- Smooth transition around threshold
- Cascading: downstream levels are gated by upstream

### Anti-Gaming Mechanisms

1. **Trigram Repetition**: Detects repeated phrases
2. **Clone Detection**: Finds duplicated sentences
3. **Entropy Check**: Penalizes low-information content
4. **Compression Ratio**: Detects highly compressible (repetitive) text
5. **Token Diversity**: Requires variety in vocabulary

## Configuration

```python
from hierarchical_rewards_v3 import RewardConfig, GateConfig, hierarchical_reward

# Custom configuration with different weights
config = RewardConfig(
    foundation_gate=GateConfig(threshold=0.4, floor=0.15, steepness=10.0, weight=0.15),
    correctness_gate=GateConfig(threshold=0.1, floor=0.10, steepness=10.0, weight=0.40),
    quality_gate=GateConfig(threshold=0.08, floor=0.08, steepness=10.0, weight=0.30),
    polish_gate=GateConfig(threshold=0.05, floor=0.05, steepness=10.0, weight=0.15),
)

score, diag = hierarchical_reward(response, expected, question, config=config)
```

## Integration with GRPO Training

```python
def compute_rewards(prompts, completions, expected_answers):
    """Compute rewards for GRPO training batch."""
    all_scores = []
    
    for prompt, completion_group, expected in zip(prompts, completions, expected_answers):
        result = batch_hierarchical_reward(
            responses=completion_group,
            expected=expected,
            question=prompt,
            ensure_ranking=True  # Ensures score differentiation
        )
        all_scores.append(result.scores)
    
    return all_scores
```

## Diagnostics Structure

```python
{
    'final_score': 0.65,
    'pre_penalty_score': 0.68,
    'anti_gaming': {
        'penalty': 0.045,
        'flags': ['high_trigram_repetition:0.39'],
        'details': {...}
    },
    'gates': {
        'foundation': 0.97,
        'correctness': 0.98,
        'quality': 0.92,
        'polish': 0.96
    },
    'levels': {
        'foundation': {
            'raw_score': 0.82,
            'gated_score': 0.80,
            'components': [...]
        },
        ...
    }
}
```

## Expected Score Ranges

| Response Type | Expected Score |
|--------------|----------------|
| Perfect (correct + good reasoning) | 0.65 - 0.75 |
| Correct but minimal | 0.50 - 0.60 |
| Wrong answer | 0.20 - 0.35 |
| Gaming attempt | 0.30 - 0.45 |
| Missing structure | 0.40 - 0.55 |

## Troubleshooting

### Low Scores for Good Responses

Check the diagnostics:
```python
score, diag = hierarchical_reward(response, expected, question)
print(f"Anti-gaming penalty: {diag['anti_gaming']['penalty']}")
print(f"Flags: {diag['anti_gaming']['flags']}")
```

### No Gradient Signal

Ensure `floor` values are > 0 in your config. The default configuration guarantees minimum 5-15% contribution at each level.

### Score Compression

If all scores are similar, batch_hierarchical_reward with `ensure_ranking=True` will add small differentiation to preserve ranking signal.
