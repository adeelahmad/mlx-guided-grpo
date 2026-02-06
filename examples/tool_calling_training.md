# Tool/Function Calling Training with MLX-GRPO

This guide shows how to train models for function/tool calling using GRPO.

## Dataset Format

The tool calling dataset uses JSONL format where each line contains:

```json
{
  "prompt": "You are a helpful assistant with access to the following functions...",
  "answer": "calculate_factorial(n=5)\ncalculate_factorial(n=7)",
  "type": "tool_call",
  "ground_truth": "calculate_factorial",
  "ground_truth_text": "calculate_factorial",
  "possible_boxed_answers": ["reverse_words", "calculate_factorial", "get_range"],
  "is_multi_answer": false,
  "confidence": 1.0,
  "source": "tool_calling"
}
```

### Key Fields

- **prompt**: Contains system message, function definitions (as JSON), and user query
- **answer**: The correct function call(s) to make
- **type**: "tool_call" for function calling tasks
- **ground_truth**: The primary function that should be called
- **possible_boxed_answers**: List of all available functions
- **is_multi_answer**: Whether multiple functions can be called

## Training Configuration

### Recommended Settings

For function calling, use these settings:

```bash
python -m mlx_grpo.train \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data /path/to/tool_calling/train.jsonl \
  --train --train-type lora \
  --iters 1000 \
  --batch-size 4 \
  --group-size 8 \
  --learning-rate 1e-5 \
  --max-completion-length 256 \
  --temperature 0.7 \
  --beta 0.1 \
  --reward-functions tool_call_exact tool_call_function tool_call_overall \
  --reward-weights 0.3 0.4 0.3 \
  --save-every 100 \
  --adapter-path adapters/tool_calling
```

### Why These Settings?

1. **max-completion-length: 256**
   - Function calls are typically short
   - Reduces memory usage and speeds up training

2. **temperature: 0.7**
   - Balanced exploration/exploitation
   - Allows model to try different parameter combinations

3. **group-size: 8**
   - Larger groups for better advantage estimation
   - Function calls benefit from comparing multiple attempts

4. **reward-functions**:
   - `tool_call_exact`: Rewards perfect matches (strict)
   - `tool_call_function`: Rewards correct function name (partial credit)
   - `tool_call_overall`: Weighted combination of function + params

### Available Reward Functions

| Reward Function | Default | Description |
|----------------|---------|-------------|
| `tool_call_exact` | Yes | 1.0 for exact match, 0.0 otherwise |
| `tool_call_function` | Yes | Partial credit for correct function name |
| `tool_call_params` | No | Rewards correct parameters (requires correct function) |
| `tool_call_overall` | No | Weighted: 40% function + 40% params + 20% exact |
| `tool_call_parseable` | No | 1.0 if output contains any parseable function call |

## Training Strategies

### Strategy 1: Strict Training (Exact Match Only)

Train the model to only get rewards for perfect function calls:

```bash
--reward-functions tool_call_exact \
--reward-weights 1.0 \
--epsilon 0.1 \
--epsilon-high 0.2
```

**Pros**: Forces precision, no partial credit for wrong answers
**Cons**: Slower learning, may need more iterations

### Strategy 2: Progressive Training (Partial Credit)

Start with partial credit, gradually increase strictness:

```bash
# Phase 1: Learn function names (100 iters)
--reward-functions tool_call_function tool_call_parseable \
--reward-weights 0.8 0.2 \
--iters 100

# Phase 2: Add parameter rewards (200 iters)
--reward-functions tool_call_function tool_call_params \
--reward-weights 0.5 0.5 \
--iters 200

# Phase 3: Demand exactness (final 200 iters)
--reward-functions tool_call_exact tool_call_overall \
--reward-weights 0.7 0.3 \
--iters 200
```

**Pros**: Faster initial learning, smoother convergence
**Cons**: Requires manual phase switching or curriculum learning

### Strategy 3: Balanced (Recommended)

Use multiple rewards with balanced weights:

```bash
--reward-functions tool_call_exact tool_call_function tool_call_overall \
--reward-weights 0.3 0.4 0.3 \
--beta 0.1
```

**Pros**: Balanced learning, partial credit encourages exploration
**Cons**: May need tuning for specific datasets

## Generation Strategy

### No Thinking Phase Needed

Function calling doesn't typically require thinking tags. Disable them:

```bash
--enforce-thinking false \
--two-phase-samples-per-group -1
```

### Direct Generation

The model should directly output function calls:

**Input:**
```
You are a helpful assistant with access to the following functions...

What is the factorial of 5?
```

**Expected Output:**
```
calculate_factorial(n=5)
```

### Multi-Function Calls

For queries requiring multiple function calls:

**Input:**
```
What are the factorials of 5, 7, and 10?
```

**Expected Output:**
```
calculate_factorial(n=5)
calculate_factorial(n=7)
calculate_factorial(n=10)
```

## Advanced Configuration

### Curriculum Learning

Gradually reduce scaffolding (if using partial prompts):

```bash
--curriculum-enabled true \
--curriculum-start-ratio 1.0 \
--curriculum-end-ratio 0.0 \
--curriculum-steps 500
```

### Memory Optimization

For large models or limited memory:

```bash
--grad-checkpoint true \
--max-completion-length 128 \
--batch-size 2 \
--group-size 4
```

### LoRA Configuration

Recommended LoRA settings for function calling:

```bash
--lora-layers 16 \
--lora-rank 32 \
--lora-alpha 64 \
--lora-dropout 0.1
```

## Monitoring Training

### Key Metrics to Watch

1. **Mean Reward**: Should increase over time
   - Initial: ~0.1-0.2 (random guessing)
   - Good: ~0.6-0.7 (most calls correct)
   - Excellent: ~0.8+ (high accuracy)

2. **tool_call_function**: Tracks function name accuracy
   - Should improve faster than exact match
   - Target: 0.9+ for good performance

3. **tool_call_exact**: Tracks perfect matches
   - Harder metric, slower to improve
   - Target: 0.7+ for production use

4. **Loss**: Should decrease steadily
   - Initial: 1.5-2.5
   - Converged: 0.5-1.0

### Example Training Output

```
Iteration 100/1000 | Loss: 1.234 | Mean Reward: 0.342
  tool_call_exact: 0.125
  tool_call_function: 0.625
  tool_call_overall: 0.456

Iteration 500/1000 | Loss: 0.756 | Mean Reward: 0.687
  tool_call_exact: 0.525
  tool_call_function: 0.875
  tool_call_overall: 0.712

Iteration 1000/1000 | Loss: 0.512 | Mean Reward: 0.823
  tool_call_exact: 0.725
  tool_call_function: 0.950
  tool_call_overall: 0.834
```

## Testing the Trained Model

### Quick Test

```python
import mlx.core as mx
from mlx_lm import load

# Load trained model
model, tokenizer = load(
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    adapter_path="adapters/tool_calling"
)

# Test prompt
prompt = """You are a helpful assistant with access to the following functions. Use them if required.

[
  {
    "name": "calculate_factorial",
    "description": "Calculates the factorial of a non-negative integer.",
    "parameters": {
      "n": {
        "description": "The non-negative integer.",
        "type": "int"
      }
    }
  }
]

What is the factorial of 7?"""

# Generate
response = generate(model, tokenizer, prompt, max_tokens=100)
print(response)
# Expected: calculate_factorial(n=7)
```

### Evaluation Script

Create `eval_tool_calling.py`:

```python
import json
from pathlib import Path
from mlx_lm import load
from mlx_grpo.trainer.tool_calling_reward import extract_function_calls, compare_function_calls

def evaluate_model(model_path, adapter_path, test_data_path):
    # Load model
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    # Load test data
    test_samples = []
    with open(test_data_path) as f:
        for line in f:
            test_samples.append(json.loads(line))

    # Evaluate
    correct = 0
    total = 0

    for sample in test_samples:
        prompt = sample['prompt']
        expected = sample['answer']

        # Generate
        response = generate(model, tokenizer, prompt, max_tokens=256)

        # Compare
        pred_calls = extract_function_calls(response)
        exp_calls = extract_function_calls(expected)
        comparison = compare_function_calls(pred_calls, exp_calls)

        if comparison['exact_match'] == 1.0:
            correct += 1
        total += 1

        # Print mismatches
        if comparison['exact_match'] < 1.0:
            print(f"\n❌ Mismatch:")
            print(f"  Expected: {expected}")
            print(f"  Got: {response}")
            print(f"  Scores: {comparison}")

    accuracy = correct / total
    print(f"\n{'='*60}")
    print(f"Exact Match Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"{'='*60}")

    return accuracy

if __name__ == "__main__":
    evaluate_model(
        model_path="mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path="adapters/tool_calling",
        test_data_path="data/tool_calling/test.jsonl"
    )
```

## Troubleshooting

### Model Generates Text Instead of Function Calls

**Problem**: Model outputs explanations rather than function calls

**Solution**:
1. Increase `tool_call_exact` reward weight
2. Add negative reward for verbose outputs
3. Use lower temperature (0.5-0.7)
4. Ensure training data is clean (no explanations in answers)

### Function Names Correct but Parameters Wrong

**Problem**: High `tool_call_function` but low `tool_call_params`

**Solution**:
1. Increase `tool_call_params` reward weight
2. Use larger group size for better comparisons
3. Add more training examples with varied parameters
4. Check if parameter types are being parsed correctly

### Training is Slow or Not Converging

**Problem**: Rewards not improving after many iterations

**Solution**:
1. Try progressive training strategy (easier rewards first)
2. Increase learning rate (try 5e-5)
3. Reduce group size if variance is too high
4. Check dataset for errors or inconsistencies

## Example Training Commands

### Quick Test (5 minutes on M2 Mac)

```bash
python -m mlx_grpo.train \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --data /Users/adeelahmad/work/rewardable_dataset/grpo_tools/train.jsonl \
  --train --train-type lora \
  --iters 50 \
  --batch-size 2 \
  --group-size 4 \
  --max-completion-length 128 \
  --reward-functions tool_call_function \
  --adapter-path adapters/tool_test
```

### Full Training (2-3 hours on M2 Mac)

```bash
python -m mlx_grpo.train \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data /Users/adeelahmad/work/rewardable_dataset/grpo_tools/train.jsonl \
  --train --train-type lora \
  --iters 1000 \
  --batch-size 4 \
  --group-size 8 \
  --learning-rate 1e-5 \
  --max-completion-length 256 \
  --temperature 0.7 \
  --beta 0.1 \
  --reward-functions tool_call_exact tool_call_function tool_call_overall \
  --reward-weights 0.3 0.4 0.3 \
  --lora-layers 16 \
  --lora-rank 32 \
  --save-every 100 \
  --adapter-path adapters/tool_calling_full \
  --wandb-project tool-calling-grpo
```

### Production Training (GPU/High-end Mac)

```bash
python -m mlx_grpo.train \
  --model mlx-community/Qwen2.5-14B-Instruct-4bit \
  --data /Users/adeelahmad/work/rewardable_dataset/grpo_tools/train.jsonl \
  --train --train-type lora \
  --iters 2000 \
  --batch-size 8 \
  --group-size 16 \
  --learning-rate 5e-6 \
  --max-completion-length 384 \
  --temperature 0.6 \
  --beta 0.08 \
  --reward-functions tool_call_exact tool_call_function tool_call_params tool_call_overall \
  --reward-weights 0.25 0.3 0.2 0.25 \
  --lora-layers 24 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --grad-checkpoint \
  --save-every 50 \
  --adapter-path adapters/tool_calling_production \
  --wandb-project tool-calling-grpo
```

## Next Steps

1. **Start with quick test** to verify setup
2. **Iterate on configuration** based on results
3. **Monitor metrics** during training
4. **Evaluate on held-out test set**
5. **Fine-tune hyperparameters** for your specific use case

## References

- MLX-GRPO Documentation: `/CLAUDE.md`
- Reward Functions: `mlx_grpo/trainer/tool_calling_reward.py`
- Dataset Loader: `mlx_grpo/trainer/tool_calling_dataset.py`
