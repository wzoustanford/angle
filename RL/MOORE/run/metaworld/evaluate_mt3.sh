#!/bin/bash

# Evaluation script for MetaWorld MT3 trained models
# Usage: sh evaluate_mt3.sh [CHECKPOINT_DIR] [N_EPISODES] [RENDER]
# Example: sh evaluate_mt3.sh logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_4e/seed_0 10
# Example with render: sh evaluate_mt3.sh logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_4e/seed_0 10 --render

# Default checkpoint directory (relative to MOORE root)
DEFAULT_CHECKPOINT="logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_4e/seed_0"

CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}
N_EPISODES=${2:-10}
RENDER_FLAG=${3:-""}  # Pass --render as third argument to enable rendering

cd ../../

echo "==========================================="
echo "Evaluating MetaWorld MT3 Model"
echo "==========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Episodes per task: $N_EPISODES"
if [ "$RENDER_FLAG" == "--render" ]; then
    echo "Rendering: ENABLED"
else
    echo "Rendering: DISABLED"
fi
echo "==========================================="
echo ""

python evaluate_metaworld_mt3.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --n_episodes $N_EPISODES \
    $RENDER_FLAG