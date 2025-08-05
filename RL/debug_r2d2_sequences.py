#!/usr/bin/env python3
"""
Debug R2D2 sequence generation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config.AgentConfig import AgentConfig
from model import DQNAgent


def debug_r2d2_sequences():
    """Debug how many sequences R2D2 generates"""
    print("Debugging R2D2 Sequence Generation...")
    
    config = AgentConfig()
    config.env_name = 'ALE/Breakout-v5'
    config.use_r2d2 = True
    config.use_prioritized_replay = True
    config.sequence_length = 40
    config.burn_in_length = 20
    config.memory_size = 1000
    config.min_replay_size = 50
    config.save_interval = 50000
    
    agent = DQNAgent(config)
    
    print(f"Config: seq_len={config.sequence_length}, burn_in={config.burn_in_length}")
    print(f"Buffer capacity: {agent.replay_buffer.capacity}")
    
    # Run a few episodes and check buffer
    print("\nRunning 3 episodes...")
    for ep in range(3):
        obs, _ = agent.env.reset()
        state = agent.frame_stack.reset(obs)
        agent.reset_hidden_state()
        
        episode_steps = 0
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            
            next_state = agent.frame_stack.append(next_obs)
            agent.replay_buffer.push_transition(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        buffer_size = len(agent.replay_buffer)
        current_ep_len = agent.replay_buffer.get_current_episode_length()
        
        print(f"Episode {ep+1}: {episode_steps} steps, reward: {episode_reward:.1f}")
        print(f"  Buffer size: {buffer_size} sequences")
        print(f"  Current episode buffer: {current_ep_len} transitions")
        
        # Calculate expected sequences
        if episode_steps >= config.sequence_length:
            expected_seqs = (episode_steps - config.sequence_length) // (config.sequence_length // 2) + 1
            print(f"  Expected sequences from this episode: {expected_seqs}")
    
    print(f"\nFinal buffer statistics:")
    print(f"Total sequences in buffer: {len(agent.replay_buffer)}")


if __name__ == '__main__':
    debug_r2d2_sequences()