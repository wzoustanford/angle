#!/usr/bin/env python3
"""
Simplified evaluation test
"""

import os
import sys
import pickle
import numpy as np

# Add the MOORE directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting evaluation test...")

import metaworld
from moore.algorithms.actor_critic import MTSAC

# Setup
checkpoint_dir = "logs/metaworld_mt3/MT3/mixture_orthogonal_experts/mt3_moore_quick_test_3e_pretexTrue/seed_42"
args_path = os.path.join(os.path.dirname(checkpoint_dir), 'args.pkl')

with open(args_path, 'rb') as f:
    args = pickle.load(f)

print(f"Setting up MT3 benchmark...")
benchmark = metaworld.MT3()
env_names = list(benchmark.train_classes.keys())
print(f"Tasks: {env_names}")

# Create a single environment for testing
print(f"Creating environment for {env_names[0]}...")
env = benchmark.train_classes[env_names[0]]()
env.seed(42)
task = benchmark.train_tasks[env_names[0]][0]
env.set_task(task)

# Load agent
agent_path = os.path.join(checkpoint_dir, 'agent', 'agent_6')
print(f"Loading agent...")
agent = MTSAC.load(agent_path)

# Set missing attributes
if hasattr(agent.policy._approximator.model.network, '_h'):
    agent.policy._approximator.model.network.get_activation_list = [
        f'1.model_layers.{str(i)}.act_0' for i in range(args.n_experts)
    ]
    if not hasattr(agent.policy._approximator.model.network, 'use_pretex_inhibition'):
        has_pretex = hasattr(agent.policy._approximator.model.network, 'pretex_inhibition_network')
        agent.policy._approximator.model.network.use_pretex_inhibition = has_pretex

print("Running single episode evaluation...")

# Run one episode
obs = env.reset()
done = False
total_reward = 0
steps = 0
max_steps = 150

while not done and steps < max_steps:
    # Get action from agent
    # The agent expects [task_idx, observation]
    action = agent.draw_action([0, obs])
    
    # Step environment
    obs, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1
    
    if steps % 50 == 0:
        print(f"  Step {steps}: reward_so_far={total_reward:.2f}")

print(f"Episode complete!")
print(f"  Total steps: {steps}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Success: {info.get('success', 0)}")

env.close()
print("Test complete!")