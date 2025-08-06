#!/usr/bin/env python3
"""
Quick test script for Alien game with minimal episodes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config.AgentConfig import AgentConfig
from model import DQNAgent

def test_alien():
    """Quick test with Alien game"""
    print("Testing Alien game with Basic DQN (1 episode)")
    
    # Create basic config
    config = AgentConfig()
    config.env_name = 'ALE/Alien-v5'
    config.use_r2d2 = False
    config.use_prioritized_replay = False
    
    # Small settings for quick test
    config.memory_size = 1000
    config.batch_size = 32
    config.learning_rate = 1e-4
    config.target_update_freq = 100
    config.min_replay_size = 100
    config.save_interval = 50000
    
    try:
        # Create agent
        print("Creating DQN agent...")
        agent = DQNAgent(config)
        
        # Run 1 episode
        print("Running 1 episode...")
        obs, _ = agent.env.reset()
        state = agent.frame_stack.reset(obs)
        episode_reward = 0
        steps = 0
        max_steps = 500  # Limit steps to prevent hanging
        
        done = False
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            
            next_state = agent.frame_stack.append(next_obs)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"  Step {steps}, reward so far: {episode_reward:.1f}")
        
        print(f"\nEpisode completed!")
        print(f"Total steps: {steps}")
        print(f"Total reward: {episode_reward:.1f}")
        print("\n✓ Test successful! The environment and agent are working.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_alien()
    exit(0 if success else 1)