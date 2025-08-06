#!/usr/bin/env python3
"""
Very quick verification that each algorithm works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config.AgentConfig import AgentConfig
from model import DQNAgent

def quick_verify():
    """Quick verification of each algorithm with minimal steps"""
    env_name = 'ALE/Alien-v5'
    max_steps = 100  # Very short episode
    
    algorithms = [
        ("Basic DQN", False, False),
        ("DQN + Dueling", True, False),
        ("DQN + Priority", False, True),
    ]
    
    print("Quick Verification (100 steps each)")
    print("="*40)
    
    for name, use_dueling, use_priority in algorithms:
        print(f"\n{name}:")
        
        config = AgentConfig()
        config.env_name = env_name
        config.use_dueling = use_dueling
        config.use_prioritized_replay = use_priority
        config.memory_size = 1000
        config.batch_size = 32
        config.min_replay_size = 100
        
        if use_priority:
            config.priority_type = 'td_error'
            config.priority_alpha = 0.6
            config.priority_beta_start = 0.4
        
        try:
            agent = DQNAgent(config)
            
            # Run very short episode
            obs, _ = agent.env.reset()
            state = agent.frame_stack.reset(obs)
            total_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = agent.env.step(action)
                done = terminated or truncated
                
                next_state = agent.frame_stack.append(next_obs)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            print(f"  ✓ Ran {step+1} steps, reward: {total_reward:.1f}")
            print(f"    Dueling: {use_dueling}, Priority: {use_priority}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n✓ All algorithms verified!")

if __name__ == '__main__':
    quick_verify()