#!/bin/bash
# GRPO training with curriculum learning example
# This script demonstrates curriculum scaffolding with thinking

set -e

# Configuration
MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DATA_DIR="$(dirname "$0")"
OUTPUT_DIR="./adapters/curriculum_example"

echo "MLX-GRPO Curriculum Learning Example"
echo "====================================="
echo "Model: $MODEL"
echo "Data: $DATA_DIR/sample_data.jsonl"
echo "Output: $OUTPUT_DIR"
echo ""

# Run training with curriculum learning
mlx-grpo \
    --model "$MODEL" \
    --data "$DATA_DIR" \
    --train \
    --train-type lora \
    --iters 50 \
    --batch-size 1 \
    --group-size 2 \
    --learning-rate 1e-5 \
    --max-completion-length 128 \
    --temperature 0.8 \
    --adapter-path "$OUTPUT_DIR" \
    --steps-per-report 10 \
    --curriculum-enabled \
    --curriculum-start-ratio 1.0 \
    --curriculum-end-ratio 0.0 \
    --curriculum-warmup-iters 10 \
    --curriculum-taper-iters 30 \
    --enforce-thinking \
    --continuation-tokens 64

echo ""
echo "Training complete! Adapters saved to: $OUTPUT_DIR"
echo ""
echo "Curriculum schedule:"
echo "  - Iterations 0-10: 100% scaffolding (warmup)"
echo "  - Iterations 10-40: Linear taper to 0%"
echo "  - Iterations 40+: 0% scaffolding (model generates independently)"
