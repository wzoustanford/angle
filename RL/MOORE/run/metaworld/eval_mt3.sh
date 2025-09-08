#!/bin/bash

# Simple evaluation script for MetaWorld MT3
# Usage: bash eval_mt3.sh

echo "Starting MetaWorld MT3 Evaluation"
echo "================================="

cd ../../

# Use the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate moore_metaworld

# Run the evaluation
python evaluate_metaworld_direct.py \
    --checkpoint_dir logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_4e/seed_0 \
    --n_episodes 5

echo "Evaluation complete!"