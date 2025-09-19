#!/bin/bash

N_EXPERTS=$1
SEED=$2
NUM_GRIN_RECURRENCE=${3:-1}  # Default to 1 if not provided

cd ../../
python run_metaworld_sac_mt.py  --seed ${SEED} --n_exp 1 --exp_type MT5 --exp_name mhsac_moore_GRIN_IR_400x3lx${N_EXPERTS}e_rec${NUM_GRIN_RECURRENCE} --results_dir logs/metaworld \
                                --batch_size 128 --n_epochs 20 --n_steps 100000 --horizon 150 --gamma 0.99 --lr_actor 3e-4 --lr_critic 3e-4 --lr_alpha 1e-4 --log_std_min -10 --log_std_max 2 \
                                --actor_network MetaworldSACMixtureMHActorNetworkGRINIR --critic_network MetaworldSACMixtureMHCriticNetworkGRINIR --orthogonal --n_experts ${N_EXPERTS} --activation Linear --agg_activation Linear Tanh \
                                --actor_n_features 400 400 400 --critic_n_features 400 400 400 --shared_mu_sigma \
                                --initial_replay_size 1500 --max_replay_size 150000 --warmup_transitions 3000 \
                                --n_episodes_test 10 --train_frequency 1 --sample_task_per_episode --rl_checkpoint_interval 3 --use_cuda \
                                --num_grin_recurrence ${NUM_GRIN_RECURRENCE}
