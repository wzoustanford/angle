#!/usr/bin/env python3
"""Quick memory test - 5 episodes only"""

import sys
import os
import gc
import torch
import psutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent

def get_memory():
    """Get memory in MB"""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    gpu_mb = 0
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
    return mem_mb, gpu_mb

print("Quick Memory Test - 5 Episodes")
print("="*50)

# Test R2D2 (most prone to memory leaks due to LSTM)
config = AgentConfig()
config.env_name = 'ALE/Alien-v5'
config.use_r2d2 = True
config.memory_size = 2000
config.sequence_length = 20
config.burn_in_length = 10
config.lstm_size = 128
config.min_replay_size = 50
config.batch_size = 16

print("\nTesting R2D2 (LSTM-based)...")
print("-"*50)

initial_mem, initial_gpu = get_memory()
print(f"Initial: {initial_mem:.1f}MB CPU, {initial_gpu:.1f}MB GPU")

try:
    agent = DQNAgent(config)
    memory_per_episode = []
    
    for ep in range(5):
        obs, _ = agent.env.reset()
        state = agent.frame_stack.reset(obs)
        agent.reset_hidden_state()
        
        episode_reward = 0
        steps = 0
        max_steps = 500  # Short episodes
        
        done = False
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            
            next_state = agent.frame_stack.append(next_obs)
            
            # Store transition
            if hasattr(agent.replay_buffer, 'push_transition'):
                agent.replay_buffer.push_transition(state, action, reward, next_state, done)
            else:
                agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update
            if agent.steps_done % 4 == 0 and len(agent.replay_buffer) > 50:
                loss = agent.update_q_network()
            
            state = next_state
            episode_reward += reward
            steps += 1
            agent.steps_done += 1
        
        # Check memory
        current_mem, current_gpu = get_memory()
        mem_increase = current_mem - initial_mem
        gpu_increase = current_gpu - initial_gpu
        memory_per_episode.append(mem_increase)
        
        print(f"Episode {ep+1}: Steps={steps:3d}, Reward={episode_reward:5.0f}, "
              f"Mem={current_mem:7.1f}MB (+{mem_increase:6.1f}), "
              f"GPU={current_gpu:6.1f}MB (+{gpu_increase:5.1f})")
    
    # Analysis
    print("\n" + "="*50)
    if len(memory_per_episode) > 1:
        avg_growth = np.mean(memory_per_episode[1:]) - memory_per_episode[0]
        print(f"Avg memory growth per episode: {avg_growth:.1f}MB")
        
        if avg_growth > 20:
            print("⚠️  WARNING: Memory leak likely present!")
        else:
            print("✓ Memory usage looks stable!")
    
    # Check if hidden states are being detached
    if hasattr(agent, 'hidden_state') and agent.hidden_state is not None:
        for h in agent.hidden_state:
            if h.requires_grad:
                print("⚠️  Hidden state still has gradients! Not properly detached!")
            else:
                print("✓ Hidden state properly detached")
            break
    
    agent.env.close()
    del agent
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

final_mem, final_gpu = get_memory()
print(f"\nFinal after cleanup: {final_mem:.1f}MB CPU, {final_gpu:.1f}MB GPU")
print("="*50)