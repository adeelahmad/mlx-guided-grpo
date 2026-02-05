#!/bin/bash
# Basic GRPO training example
# This script demonstrates a minimal training run

set -e

# Configuration
MODEL="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DATA_DIR="$(dirname "$0")"
OUTPUT_DIR="./adapters/basic_example"

echo "MLX-GRPO Basic Training Example"
echo "================================"
echo "Model: $MODEL"
echo "Data: $DATA_DIR/sample_data.jsonl"
echo "Output: $OUTPUT_DIR"
echo ""

# Run training
mlx-grpo \
    --model "$MODEL" \
    --data "$DATA_DIR" \
    --train \
    --train-type lora \
    --iters 10 \
    --batch-size 1 \
    --group-size 2 \
    --learning-rate 1e-5 \
    --max-completion-length 128 \
    --temperature 0.8 \
    --adapter-path "$OUTPUT_DIR" \
    --steps-per-report 5

echo ""
echo "Training complete! Adapters saved to: $OUTPUT_DIR"
