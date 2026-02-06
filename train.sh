#!/bin/bash

# =============================================================================
# MLX-GRPO Training Script with Auto-Restart
# =============================================================================
#
# Exit Codes:
#   0   - Normal completion
#   42  - Corruption detected (DO NOT RESTART)
#   130 - User interrupt (Ctrl+C)
#   137 - OOM killed (SIGKILL)
#   139 - Segfault
#   *   - Other errors (will restart)
#
# Usage:
#   ./train.sh              # Run with auto-restart
#   ./train.sh --no-loop    # Run once without auto-restart
#
# =============================================================================

set -e  # Exit on first error (we handle restart logic manually)

# Configuration - update these paths for your environment
MODEL="path/to/your/model"
ADAPTER="adapters/my_adapter"
DATA="path/to/your/dataset"  # Hermes-formatted data (aligned with Qwen)

# Auto-restart settings
MAX_RESTARTS=10
RESTART_DELAY=5  # seconds
KEEP_LAST_N_CHECKPOINTS=5  # Rolling adapter saves

# Exit codes that should NOT trigger a restart
EXIT_CODE_CORRUPTION=42
EXIT_CODE_INTERRUPT=130

# Parse arguments
NO_LOOP=false
for arg in "$@"; do
    case $arg in
        --no-loop)
            NO_LOOP=true
            shift
            ;;
    esac
done

# Activate conda environment (uncomment and update for your setup)
# source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
# conda activate myenv

# Training function
run_training() {
    python -m mlx_grpo.train  \
        --model "$MODEL"  \
        --train \
        --train-type dora \
        --data "$DATA" \
        --reward-weights "[1.3]" \
        --grpo-loss-type dr_grpo \
        --epsilon 0.02 \
        --epsilon-high 0.05 \
        --beta 0.08 \
        --group-size 3 \
        --max-completion-length 256 \
        --continuation-tokens 64 \
        --temperature 0.85 \
        --batch-size 1 \
        --gradient-accumulation-steps 3 \
        --learning-rate 3e-6 \
        --importance-sampling-level sequence \
        --steps-per-report 1 \
        --steps-per-eval 0 \
        --save-every 5 \
        --iters 2000 \
        --adapter-path "$ADAPTER" \
        --wandb my-experiment-name \
        --cache-dataset \
        --keep-last-n-checkpoints "$KEEP_LAST_N_CHECKPOINTS" \
        --curriculum-enabled \
        --curriculum-start-ratio 1.0 \
        --curriculum-end-ratio 0.0 \
        --curriculum-warmup-iters 0 \
        --curriculum-taper-iters 1000 --shuffle-data --shuffle-seed $RANDOM  \
        --resume --balanced-shuffle --lora-rank 32 --lora-alpha 64 --lora-dropout 0.01 --seed $RANDOM  --enforce-thinking --exam-phase-recovery-ratio 0.5

}

# Trap Ctrl+C to exit cleanly
trap 'echo ""; echo "Interrupted by user. Exiting..."; exit 130' INT

# Main loop
restart_count=0

while true; do
    echo ""
    echo "=============================================="
    echo "Starting training (attempt $((restart_count + 1))/$((MAX_RESTARTS + 1)))"
    echo "=============================================="
    echo ""

    # Run training and capture exit code
    set +e  # Temporarily allow errors
    run_training
    exit_code=$?
    set -e

    echo ""
    echo "Training exited with code: $exit_code"

    # Handle exit codes
    case $exit_code in
        0)
            echo "Training completed successfully!"
            exit 0
            ;;

        $EXIT_CODE_CORRUPTION)
            echo ""
            echo "=============================================="
            echo "FATAL: Corruption detected (exit code 42)"
            echo "=============================================="
            echo "DO NOT restart training with corrupted adapter."
            echo "Check the adapter files and consider reverting to a previous checkpoint."
            echo ""
            exit $EXIT_CODE_CORRUPTION
            ;;

        $EXIT_CODE_INTERRUPT)
            echo ""
            echo "Training interrupted by user."
            exit $EXIT_CODE_INTERRUPT
            ;;

        137)
            echo ""
            echo "Training killed by OOM (SIGKILL)."
            echo "Consider reducing batch-size or max-completion-length."
            ;;

        139)
            echo ""
            echo "Training crashed with segfault."
            ;;

        *)
            echo ""
            echo "Training failed with exit code $exit_code."
            ;;
    esac

    # Check if we should restart
    if [ "$NO_LOOP" = true ]; then
        echo "Auto-restart disabled (--no-loop). Exiting."
        exit $exit_code
    fi

    restart_count=$((restart_count + 1))

    if [ $restart_count -ge $MAX_RESTARTS ]; then
        echo ""
        echo "=============================================="
        echo "Max restarts ($MAX_RESTARTS) reached. Giving up."
        echo "=============================================="
        exit $exit_code
    fi

    echo ""
    echo "Auto-restarting in $RESTART_DELAY seconds..."
    echo "Press Ctrl+C to cancel."
    sleep $RESTART_DELAY

done
