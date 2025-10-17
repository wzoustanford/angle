#!/bin/bash

cd ../../../

ENV_NAME=$1
N_EXPERTS=$2

python run_minigrid_ppo_mt.py  --n_exp 10 \
                            --env_name ${ENV_NAME} --exp_name ppo_mt_moore_GRINlight_multihead_${N_EXPERTS}e \
                            --n_epochs 100 --n_steps 2000  --n_episodes_test 16 --train_frequency 2000 --lr_actor 1e-3 --lr_critic 1e-3 \
                            --critic_network MiniGridPPOMixtureMHNetworkGRIN --critic_n_features 128 --orthogonal --n_experts ${N_EXPERTS} \
                            --actor_network MiniGridPPOMixtureMHNetworkGRIN --actor_n_features 128 \
                            --batch_size 256 --gamma 0.99 
