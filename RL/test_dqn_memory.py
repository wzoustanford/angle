#!/usr/bin/env python3
"""Test DQN agent memory usage with fixes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import psutil
import torch
from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent

def get_mem_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

print("Testing DQN Agent Memory Usage")
print("="*50)

# R2D2 config (uses LSTM)
config = AgentConfig()
config.env_name = 'ALE/Alien-v5'
config.use_r2d2 = True
config.memory_size = 1000
config.sequence_length = 10
config.burn_in_length = 5
config.lstm_size = 64
config.batch_size = 8
config.min_replay_size = 50

initial_mem = get_mem_mb()
print(f"Initial memory: {initial_mem:.1f}MB")

try:
    agent = DQNAgent(config)
    print(f"After agent creation: {get_mem_mb():.1f}MB")
    
    # Run 3 very short episodes
    for ep in range(3):
        obs, _ = agent.env.reset()
        state = agent.frame_stack.reset(obs)
        agent.reset_hidden_state()
        
        # Just 100 steps per episode
        for step in range(100):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            
            next_state = agent.frame_stack.append(next_obs)
            
            if hasattr(agent.replay_buffer, 'push_transition'):
                agent.replay_buffer.push_transition(state, action, reward, next_state, done)
            else:
                agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if agent.steps_done % 10 == 0 and len(agent.replay_buffer) > 50:
                agent.update_q_network()
            
            state = next_state
            agent.steps_done += 1
            
            if done:
                break
        
        current_mem = get_mem_mb()
        print(f"After episode {ep+1}: {current_mem:.1f}MB (+{current_mem-initial_mem:.1f}MB)")
        
        # Check if hidden state is detached
        if hasattr(agent, 'hidden_state') and agent.hidden_state is not None:
            if isinstance(agent.hidden_state, tuple):
                for i, h in enumerate(agent.hidden_state):
                    if h.requires_grad:
                        print(f"  ⚠️  Hidden state {i} still has gradients!")
                    else:
                        print(f"  ✓ Hidden state {i} properly detached")
            
    print("\n" + "="*50)
    print("Summary:")
    final_mem = get_mem_mb()
    total_increase = final_mem - initial_mem
    per_episode = total_increase / 3
    
    print(f"Total memory increase: {total_increase:.1f}MB")
    print(f"Per episode: {per_episode:.1f}MB")
    
    if per_episode > 10:
        print("⚠️  WARNING: Possible memory leak!")
    else:
        print("✓ Memory usage appears stable!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("="*50)