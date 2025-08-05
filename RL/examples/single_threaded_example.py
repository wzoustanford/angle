#!/usr/bin/env python3
"""
Single-threaded Space Invaders DQN example.

This script demonstrates the traditional single-threaded DQN training approach
for comparison with the distributed version.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import sys
import os

# Add parent directory to path for imports when run from examples folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import DQNAgent
from config import AgentConfig


def plot_training_results(episode_rewards, losses, save_path='./results/single_threaded_training.png'):
    """Plot training results for single-threaded training"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6, label='Episode Rewards')
    if len(episode_rewards) >= 100:
        # Moving average
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(episode_rewards)), moving_avg, 'r', linewidth=2, label='Moving Average (100 episodes)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Single-Threaded Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses
    if losses:
        ax2.plot(losses, alpha=0.7, label='Training Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No loss data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Training Loss')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training plots saved to {save_path}")


def run_single_threaded_training(episodes=200):
    """Run traditional single-threaded DQN training"""
    print("=" * 80)
    print(f"SINGLE-THREADED DQN TRAINING - {episodes} EPISODES")
    print("=" * 80)
    
    # Create configuration
    config = AgentConfig(
        env_name='ALE/SpaceInvaders-v5',
        memory_size=10000,
        min_replay_size=1000,
        batch_size=32,
        learning_rate=1e-4,
        target_update_freq=1000,
        save_interval=episodes // 4,  # Save 4 times during training
        checkpoint_dir='./results/single_threaded_checkpoints'
    )
    
    print(f"Configuration:")
    print(f"  Environment: {config.env_name}")
    print(f"  Buffer size: {config.memory_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Create agent
    agent = DQNAgent(config)
    
    try:
        print(f"\nStarting single-threaded training...")
        start_time = time.time()
        
        # Train the agent
        episode_rewards, losses = agent.train(num_episodes=episodes)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Training completed in {total_time:.2f} seconds!")
        
        # Print final statistics
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)
        
        print("\nFinal Statistics:")
        print(f"  Total episodes: {len(episode_rewards)}")
        print(f"  Total steps: {agent.steps_done}")
        print(f"  Average reward (last 100): {avg_reward:.2f}")
        print(f"  Max reward: {max_reward:.2f}")
        print(f"  Min reward: {min_reward:.2f}")
        print(f"  Episodes/second: {episodes / total_time:.2f}")
        print(f"  Steps/second: {agent.steps_done / total_time:.1f}")
        
        # Test the trained agent
        print(f"\nTesting trained agent...")
        test_rewards = agent.test(num_episodes=5, render=False)
        avg_test_reward = np.mean(test_rewards)
        print(f"Average test reward: {avg_test_reward:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'losses': losses,
            'total_time': total_time,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'test_reward': avg_test_reward,
            'episodes_per_second': episodes / total_time,
            'steps_per_second': agent.steps_done / total_time
        }
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Single-threaded DQN Example for Space Invaders')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes to train (default: 200)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting results')
    parser.add_argument('--test-only', action='store_true',
                       help='Just test imports and basic functionality')
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Testing single-threaded DQN setup...")
        try:
            config = AgentConfig()
            agent = DQNAgent(config)
            print("✓ Single-threaded DQN agent created successfully")
            print(f"✓ Environment: {config.env_name}")
            print(f"✓ Device: {agent.device}")
            print(f"✓ Number of actions: {agent.n_actions}")
            return 0
        except Exception as e:
            print(f"✗ Setup failed: {e}")
            return 1
    
    print("Running single-threaded DQN training example...")
    results = run_single_threaded_training(episodes=args.episodes)
    
    if results and not args.no_plot:
        plot_training_results(results['episode_rewards'], results['losses'])
    
    return 0 if results else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)