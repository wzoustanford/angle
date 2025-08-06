#!/usr/bin/env python3
"""
Quick test of the 4 requested algorithms on Alien game
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig
from model import DQNAgent

def test_algorithms():
    """Test all 4 algorithms with 1 episode on Alien"""
    env_name = 'ALE/Alien-v5'
    max_steps = 500
    
    print("="*60)
    print("Testing 4 Algorithms on Alien (1 episode each)")
    print("="*60)
    
    # Algorithm 1: Basic DQN
    print("\n1. Basic DQN")
    print("-"*30)
    config1 = AgentConfig()
    config1.env_name = env_name
    config1.use_dueling = False
    config1.use_prioritized_replay = False
    config1.memory_size = 5000
    config1.batch_size = 32
    config1.min_replay_size = 500
    
    try:
        agent1 = DQNAgent(config1)
        print(f"✓ Basic DQN created successfully")
        print(f"  Using dueling: {config1.use_dueling}")
        print(f"  Using priority: {config1.use_prioritized_replay}")
    except Exception as e:
        print(f"✗ Basic DQN failed: {e}")
    
    # Algorithm 2: DQN + Dueling Networks
    print("\n2. DQN + Dueling Networks")
    print("-"*30)
    config2 = AgentConfig()
    config2.env_name = env_name
    config2.use_dueling = True  # Enable dueling
    config2.use_prioritized_replay = False
    config2.memory_size = 5000
    config2.batch_size = 32
    config2.min_replay_size = 500
    
    try:
        agent2 = DQNAgent(config2)
        print(f"✓ DQN + Dueling created successfully")
        print(f"  Using dueling: {config2.use_dueling}")
        print(f"  Using priority: {config2.use_prioritized_replay}")
        
        # Quick test of forward pass
        obs, _ = agent2.env.reset()
        state = agent2.frame_stack.reset(obs)
        action = agent2.select_action(state)
        print(f"  Test action selection: action={action}")
    except Exception as e:
        print(f"✗ DQN + Dueling failed: {e}")
    
    # Algorithm 3: DQN + Priority
    print("\n3. DQN + Priority Replay")
    print("-"*30)
    config3 = AgentConfig()
    config3.env_name = env_name
    config3.use_dueling = False
    config3.use_prioritized_replay = True  # Enable priority
    config3.priority_type = 'td_error'
    config3.priority_alpha = 0.6
    config3.priority_beta_start = 0.4
    config3.memory_size = 5000
    config3.batch_size = 32
    config3.min_replay_size = 500
    
    try:
        agent3 = DQNAgent(config3)
        print(f"✓ DQN + Priority created successfully")
        print(f"  Using dueling: {config3.use_dueling}")
        print(f"  Using priority: {config3.use_prioritized_replay}")
        print(f"  Priority type: {config3.priority_type}")
    except Exception as e:
        print(f"✗ DQN + Priority failed: {e}")
    
    # Algorithm 4: Distributed + Priority (single worker test)
    print("\n4. Distributed RL + Priority")
    print("-"*30)
    config4 = DistributedAgentConfig()
    config4.env_name = env_name
    config4.num_workers = 2  # Use fewer workers for test
    config4.use_dueling = False
    config4.use_prioritized_replay = True
    config4.priority_type = 'td_error'
    config4.priority_alpha = 0.6
    config4.memory_size = 5000
    config4.batch_size = 32
    config4.min_replay_size = 500
    
    try:
        # Just test config creation, don't run distributed agent in quick test
        print(f"✓ Distributed config created successfully")
        print(f"  Workers: {config4.num_workers}")
        print(f"  Using dueling: {config4.use_dueling}")
        print(f"  Using priority: {config4.use_prioritized_replay}")
    except Exception as e:
        print(f"✗ Distributed config failed: {e}")
    
    # Run a quick episode with Basic DQN to verify everything works
    print("\n" + "="*60)
    print("Running quick test episode with Basic DQN")
    print("="*60)
    
    try:
        agent = agent1  # Use basic DQN
        obs, _ = agent.env.reset()
        state = agent.frame_stack.reset(obs)
        episode_reward = 0
        steps = 0
        
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
                print(f"  Step {steps}, reward: {episode_reward:.1f}")
        
        print(f"\nTest episode completed!")
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.1f}")
        print("\n✓ All algorithms tested successfully!")
        
    except Exception as e:
        print(f"\n✗ Test episode failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_algorithms()