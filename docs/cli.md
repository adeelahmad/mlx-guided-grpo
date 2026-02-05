# CLI Reference

Complete command-line interface reference for MLX Guided GRPO.

## Basic Usage

```bash
mlx-grpo --model <MODEL> --data <DATA_PATH> --train [OPTIONS]
```

## Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Path to model or HuggingFace model ID |
| `--data` | Path to training data directory or JSONL file |

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` | False | Enable training mode |
| `--train-type` | `lora` | Training type: `lora`, `dora`, or `full` |
| `--iters` | 100 | Number of training iterations |
| `--batch-size` | 4 | Batch size per iteration |
| `--learning-rate` | 1e-5 | Learning rate |
| `--gradient-accumulation-steps` | 1 | Gradient accumulation steps |

## GRPO Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--group-size` | 4 | Number of completions per prompt |
| `--beta` | 0.1 | KL divergence coefficient |
| `--epsilon` | 0.1 | PPO clipping lower bound |
| `--epsilon-high` | 0.2 | PPO clipping upper bound |
| `--temperature` | 0.8 | Sampling temperature |
| `--max-completion-length` | 512 | Maximum completion tokens |
| `--grpo-loss-type` | `grpo` | Loss type: `grpo`, `dr_grpo`, `bnpo` |

## Curriculum Learning

| Argument | Default | Description |
|----------|---------|-------------|
| `--curriculum-enabled` | False | Enable curriculum scaffolding |
| `--curriculum-start-ratio` | 1.0 | Initial scaffold ratio (1.0 = full) |
| `--curriculum-end-ratio` | 0.0 | Final scaffold ratio (0.0 = none) |
| `--curriculum-warmup-iters` | 0 | Warmup iterations at start ratio |
| `--curriculum-taper-iters` | 100 | Taper iterations to end ratio |
| `--multi-curriculum-rollout` | False | Different scaffold per group member |

## Two-Phase Generation

| Argument | Default | Description |
|----------|---------|-------------|
| `--enforce-thinking` | False | Enable two-phase recovery |
| `--continuation-tokens` | 256 | Max tokens for phase 2 |
| `--think-start-token` | `<think>` | Opening think tag |
| `--think-end-token` | `</think>` | Closing think tag |

## Memory & Performance

| Argument | Default | Description |
|----------|---------|-------------|
| `--grad-checkpoint` | False | Enable gradient checkpointing |
| `--reference-model-path` | None | Separate reference model |

## Logging & Monitoring

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb` | None | WandB project name |
| `--log-rollouts` | False | Log rollouts to file |
| `--log-rollouts-to-wandb` | False | Log rollouts to WandB |
| `--steps-per-report` | 10 | Steps between reports |
| `--steps-per-save` | 100 | Steps between checkpoints |

## Adapter & Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--adapter-path` | `./adapters` | Output directory for adapters |
| `--resume` | False | Resume from checkpoint |

## Examples

### Minimal Training
```bash
mlx-grpo --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
         --data ./data --train
```

### Full Featured
```bash
mlx-grpo \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --data ./reasoning_data \
    --train \
    --train-type lora \
    --iters 2000 \
    --batch-size 2 \
    --group-size 4 \
    --curriculum-enabled \
    --curriculum-start-ratio 1.0 \
    --curriculum-end-ratio 0.0 \
    --curriculum-warmup-iters 100 \
    --curriculum-taper-iters 500 \
    --enforce-thinking \
    --grad-checkpoint \
    --wandb my-experiment \
    --adapter-path ./my-model
```
