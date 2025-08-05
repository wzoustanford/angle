#!/usr/bin/env python3
"""
Test script to compare uniform vs prioritized replay buffer performance
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory (RL) to path to import packages
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from config.AgentConfig import AgentConfig
from model import DQNAgent


def test_replay_buffer(use_prioritized=False, num_episodes=10):
    """Test either uniform or prioritized replay buffer"""
    print(f"\n{'='*50}")
    print(f"Testing {'Prioritized' if use_prioritized else 'Uniform'} Replay Buffer")
    print(f"{'='*50}")
    
    # Create configuration
    config = AgentConfig()
    config.use_prioritized_replay = use_prioritized
    config.memory_size = 1000  # Smaller for testing
    config.min_replay_size = 100
    config.target_update_freq = 50
    config.save_interval = 10000  # Disable saving for test
    
    # Create agent
    try:
        agent = DQNAgent(config)
        print(f"✓ Agent created successfully")
        print(f"  Buffer type: {type(agent.replay_buffer).__name__}")
        
        if use_prioritized:
            print(f"  Priority alpha: {agent.replay_buffer.alpha}")
            print(f"  Priority beta: {agent.replay_buffer.beta}")
            print(f"  Priority type: {agent.replay_buffer.priority_type}")
        
        # Train for a few episodes
        print(f"\nTraining for {num_episodes} episodes...")
        episode_rewards, losses = agent.train(num_episodes=num_episodes)
        
        # Print results
        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean(losses) if losses else 0
        print(f"✓ Training completed")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Buffer size: {len(agent.replay_buffer)}")
        
        return episode_rewards, losses
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return [], []


def compare_replay_methods():
    """Compare uniform vs prioritized replay performance"""
    print("Comparing Replay Buffer Methods")
    print("="*60)
    
    # Test uniform replay
    uniform_rewards, uniform_losses = test_replay_buffer(use_prioritized=False, num_episodes=10)
    
    # Test prioritized replay
    priority_rewards, priority_losses = test_replay_buffer(use_prioritized=True, num_episodes=10)
    
    # Plot comparison if we have results
    if uniform_rewards and priority_rewards:
        plt.figure(figsize=(12, 4))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(uniform_rewards, label='Uniform Replay', alpha=0.7)
        plt.plot(priority_rewards, label='Prioritized Replay', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards Comparison')
        plt.legend()
        plt.grid(True)
        
        # Plot losses
        plt.subplot(1, 2, 2)
        if uniform_losses:
            plt.plot(uniform_losses, label='Uniform Replay', alpha=0.7)
        if priority_losses:
            plt.plot(priority_losses, label='Prioritized Replay', alpha=0.7)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('./results/replay_comparison.png')
        plt.show()
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Uniform Replay   - Avg Reward: {np.mean(uniform_rewards):.2f}")
        print(f"Prioritized Replay - Avg Reward: {np.mean(priority_rewards):.2f}")
        if uniform_losses and priority_losses:
            print(f"Uniform Replay   - Avg Loss: {np.mean(uniform_losses):.4f}")
            print(f"Prioritized Replay - Avg Loss: {np.mean(priority_losses):.4f}")


def test_priority_types():
    """Test different priority calculation methods"""
    print("\nTesting Different Priority Types")
    print("="*40)
    
    priority_types = ['td_error', 'reward', 'random']
    
    for priority_type in priority_types:
        print(f"\nTesting priority type: {priority_type}")
        config = AgentConfig()
        config.use_prioritized_replay = True
        config.priority_type = priority_type
        config.memory_size = 500
        config.min_replay_size = 50
        
        try:
            agent = DQNAgent(config)
            print(f"✓ Agent with {priority_type} priorities created")
            
            # Test a few updates
            rewards, losses = agent.train(num_episodes=3)
            print(f"  Completed {len(rewards)} episodes")
            
        except Exception as e:
            print(f"✗ Error with {priority_type}: {e}")


if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('./results', exist_ok=True)
    
    # Run tests
    print("Starting Prioritized Replay Buffer Tests")
    
    # Test basic functionality
    compare_replay_methods()
    
    # Test different priority types
    test_priority_types()
    
    print("\n" + "="*60)
    print("All tests completed!")