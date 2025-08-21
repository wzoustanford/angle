#!/usr/bin/env python3
"""Quick 2-episode test to verify algorithms work"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent
import numpy as np
import gc
import torch
import psutil

def test_algorithm(name, config):
    print(f"\n{'='*40}")
    print(f"Testing: {name}")
    
    # Memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"Memory before: {mem_before:.1f}MB")
    
    try:
        agent = DQNAgent(config)
        rewards = []
        
        for ep in range(2):
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            if hasattr(agent, 'reset_hidden_state'):
                agent.reset_hidden_state()
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:  # Short episodes
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                next_state = agent.frame_stack.append(next_obs)
                
                if hasattr(agent.replay_buffer, 'push_transition'):
                    agent.replay_buffer.push_transition(state, action, reward, next_state, done)
                else:
                    agent.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                steps += 1
                agent.steps_done += 1
            
            rewards.append(episode_reward)
            print(f"  Episode {ep+1}: Reward={episode_reward:.0f}")
        
        print(f"✓ Success! Avg: {np.mean(rewards):.1f}")
        
        # Cleanup
        agent.env.close()
        del agent
        
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Memory after
    mem_after = process.memory_info().rss / 1024 / 1024
    print(f"Memory after cleanup: {mem_after:.1f}MB (freed: {mem_before-mem_after:.1f}MB)")

# Test configurations
print("Quick 2-Episode Test")
print("="*40)

# 1. Basic DQN
config = AgentConfig()
config.env_name = 'ALE/Alien-v5'
config.memory_size = 1000
config.batch_size = 8
test_algorithm("DQN", config)

# 2. R2D2
config = AgentConfig()
config.env_name = 'ALE/Alien-v5'
config.use_r2d2 = True
config.memory_size = 1000
config.sequence_length = 20
config.burn_in_length = 10
config.lstm_size = 128
test_algorithm("R2D2", config)

print("\n" + "="*40)
print("Test complete!")