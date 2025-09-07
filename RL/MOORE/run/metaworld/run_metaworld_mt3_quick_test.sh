#!/bin/bash

# Quick MT3 test script for MetaWorld with MOORE
# Usage: sh run_metaworld_mt3_quick_test.sh [N_EXPERTS] [SEED]
# Example: sh run_metaworld_mt3_quick_test.sh 3 0

N_EXPERTS=${1:-3}  # Default to 3 experts for 3 tasks
SEED=${2:-0}       # Default seed 0

cd ../../

# Reduced parameters for faster experiments (~30-45 minutes)
python run_metaworld_sac_mt.py \
    --seed ${SEED} \
    --n_exp 1 \
    --exp_type MT3 \
    --exp_name mt3_moore_quick_test_${N_EXPERTS}e \
    --results_dir logs/metaworld_mt3 \
    --batch_size 128 \
    --n_epochs 10 \
    --n_steps 10000 \
    --horizon 150 \
    --gamma 0.99 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --lr_alpha 1e-4 \
    --log_std_min -10 \
    --log_std_max 2 \
    --actor_network MetaworldSACMixtureMHActorNetwork \
    --critic_network MetaworldSACMixtureMHCriticNetwork \
    --orthogonal \
    --n_experts ${N_EXPERTS} \
    --activation Linear \
    --agg_activation Linear Tanh \
    --actor_n_features 256 256 \
    --critic_n_features 256 256 \
    --shared_mu_sigma \
    --initial_replay_size 1500 \
    --max_replay_size 150000 \
    --warmup_transitions 3000 \
    --n_episodes_test 2 \
    --train_frequency 1 \
    --sample_task_per_episode \
    --rl_checkpoint_interval 3 \
    --use_cuda

# Original MT10 parameters for reference:
# --batch_size 128 
# --n_epochs 20 
# --n_steps 100000 
# --actor_n_features 400 400 400 
# --critic_n_features 400 400 400 
# --initial_replay_size 1500 
# --max_replay_size 1000000 
# --warmup_transitions 3000 
# --n_episodes_test 10
