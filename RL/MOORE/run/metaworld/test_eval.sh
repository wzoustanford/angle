#!/bin/bash

# Quick test of evaluation
cd ../../

source ~/anaconda3/etc/profile.d/conda.sh
conda activate moore_metaworld

# Test with the pretex model that has checkpoints
python evaluate_metaworld_direct.py \
    --checkpoint_dir logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_3e_pretexTrue/seed_42 \
    --n_episodes 1