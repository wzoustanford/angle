#!/usr/bin/env python3
"""Simple test of R2D2+Agent57 for 5 episodes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.NGUConfig import Agent57Config
from model.r2d2_agent57_hybrid import R2D2Agent57Hybrid
import psutil

def get_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

print("Testing R2D2+Agent57 - 5 Episodes")
print("="*50)

config = Agent57Config()
config.env_name = 'ALE/Alien-v5'
config.num_policies = 4  # Fewer policies
config.memory_size = 2000  # Small buffer
config.episodic_memory_size = 500
config.sequence_length = 20
config.burn_in_length = 10
config.max_episode_steps = 500  # Short episodes

initial_mem = get_mem_mb()
print(f"Initial memory: {initial_mem:.1f}MB\n")

try:
    agent = R2D2Agent57Hybrid(config)
    print(f"After agent creation: {get_mem_mb():.1f}MB\n")
    
    for ep in range(5):
        stats = agent.train_episode()
        current_mem = get_mem_mb()
        print(f"Episode {ep+1}: Reward={stats['episode_reward']:.0f}, "
              f"Policy={stats['policy_id']}, "
              f"Memory={current_mem:.0f}MB (+{current_mem-initial_mem:.0f})")
    
    print(f"\nFinal memory: {get_mem_mb():.1f}MB")
    print(f"Memory per episode: {(get_mem_mb()-initial_mem)/5:.1f}MB")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("="*50)