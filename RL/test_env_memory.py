#!/usr/bin/env python3
"""Test if the environment itself is leaking memory"""

import gym
import psutil
import numpy as np

def get_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

print("Testing Environment Memory Usage")
print("="*50)

initial_mem = get_mem_mb()
print(f"Initial memory: {initial_mem:.1f}MB")

# Create environment
env = gym.make('ALE/Alien-v5')
print(f"After env creation: {get_mem_mb():.1f}MB")

# Store observations to simulate replay buffer
observations = []

# Run episodes
for ep in range(5):
    obs, _ = env.reset()
    
    for step in range(200):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Store observations (simulating replay buffer)
        observations.append(obs)
        observations.append(next_obs)
        
        obs = next_obs
        
        if terminated or truncated:
            break
    
    current_mem = get_mem_mb()
    print(f"Episode {ep+1}: Memory={current_mem:.1f}MB (+{current_mem-initial_mem:.1f}MB)")
    print(f"  Stored observations: {len(observations)}")

env.close()

print("\n" + "="*50)
final_mem = get_mem_mb()
print(f"Final memory: {final_mem:.1f}MB")
print(f"Total increase: {final_mem-initial_mem:.1f}MB")

# Calculate size of stored observations
obs_size = sum(obs.nbytes for obs in observations) / 1024 / 1024
print(f"Size of observations: {obs_size:.1f}MB")
print(f"Unexplained memory: {(final_mem-initial_mem-obs_size):.1f}MB")
print("="*50)