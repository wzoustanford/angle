#!/usr/bin/env python3
"""
R2D2 (Recurrent Experience Replay in Distributed RL) Example

This example demonstrates R2D2 usage with LSTM networks and sequence replay.
"""

import sys
import os
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from model import DQNAgent


def create_r2d2_config(mode='fast'):
    """Create R2D2 configuration for different modes"""
    config = AgentConfig()
    
    # Enable R2D2 with prioritized replay
    config.use_r2d2 = True
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    
    # R2D2 specific settings
    config.lstm_size = 256 if mode == 'fast' else 512
    config.sequence_length = 40 if mode == 'fast' else 80
    config.burn_in_length = 20 if mode == 'fast' else 40
    
    if mode == 'fast':
        # Fast testing configuration
        config.memory_size = 2000
        config.min_replay_size = 200
        config.batch_size = 16
        config.target_update_freq = 100
        config.save_interval = 10000  # Disable saving
    elif mode == 'demo':
        # Demo configuration
        config.memory_size = 5000
        config.min_replay_size = 500
        config.batch_size = 32
        config.target_update_freq = 500
    else:  # full
        # Full R2D2 configuration
        config.memory_size = 50000
        config.min_replay_size = 1000
        config.batch_size = 32
        config.target_update_freq = 1000
    
    return config


def run_r2d2_demo(mode='fast', episodes=5):
    """Run R2D2 training demo"""
    print("R2D2 (Recurrent Experience Replay) Example")
    print("=" * 50)
    
    # Create configuration
    config = create_r2d2_config(mode)
    
    print(f"Mode: {mode}")
    print(f"LSTM size: {config.lstm_size}")
    print(f"Sequence length: {config.sequence_length}")
    print(f"Burn-in length: {config.burn_in_length}")
    print(f"Memory size: {config.memory_size}")
    print(f"Episodes: {episodes}")
    print()
    
    # Create R2D2 agent
    print("Creating R2D2 agent...")
    agent = DQNAgent(config)
    print()
    
    # Train for specified episodes
    print(f"Training R2D2 agent for {episodes} episodes...")
    episode_rewards, losses = agent.train(num_episodes=episodes)
    
    # Print results
    print(f"\nTraining completed!")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    if losses:
        print(f"Average loss: {np.mean(losses):.4f}")
    
    print(f"\nBuffer status:")
    print(f"  Total sequences stored: {len(agent.replay_buffer)}")
    if hasattr(agent.replay_buffer, 'get_current_episode_length'):
        print(f"  Current episode length: {agent.replay_buffer.get_current_episode_length()}")
    
    return episode_rewards, losses


def compare_dqn_vs_r2d2(episodes=3):
    """Quick comparison between standard DQN and R2D2"""
    print("DQN vs R2D2 Comparison")
    print("=" * 30)
    
    results = {}
    
    # Test standard DQN
    print("\n1. Testing Standard DQN...")
    dqn_config = AgentConfig()
    dqn_config.use_r2d2 = False
    dqn_config.use_prioritized_replay = False
    dqn_config.memory_size = 2000
    dqn_config.min_replay_size = 200
    dqn_config.batch_size = 16
    dqn_config.save_interval = 10000
    
    dqn_agent = DQNAgent(dqn_config)
    dqn_rewards, _ = dqn_agent.train(num_episodes=episodes)
    results['DQN'] = np.mean(dqn_rewards)
    print(f"   Average reward: {results['DQN']:.2f}")
    
    # Test R2D2
    print("\n2. Testing R2D2...")
    r2d2_config = create_r2d2_config('fast')
    r2d2_agent = DQNAgent(r2d2_config)
    r2d2_rewards, _ = r2d2_agent.train(num_episodes=episodes)
    results['R2D2'] = np.mean(r2d2_rewards)
    print(f"   Average reward: {results['R2D2']:.2f}")
    
    # Print comparison
    print(f"\nComparison Results ({episodes} episodes each):")
    print(f"  Standard DQN: {results['DQN']:.2f}")
    print(f"  R2D2:         {results['R2D2']:.2f}")
    
    if results['R2D2'] > results['DQN']:
        print("  → R2D2 performed better!")
    elif results['DQN'] > results['R2D2']:
        print("  → Standard DQN performed better!")
    else:
        print("  → Performance was similar!")
    
    print("\nNote: Results may vary due to random initialization and short training.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='R2D2 Example')
    parser.add_argument('--mode', choices=['fast', 'demo', 'full', 'compare'], 
                       default='fast', help='Example mode')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='Number of episodes to train')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'compare':
            compare_dqn_vs_r2d2(episodes=args.episodes)
        else:
            run_r2d2_demo(mode=args.mode, episodes=args.episodes)
            
        print(f"\n{'='*50}")
        print("Example completed successfully!")
        print("Try different modes:")
        print("  --mode fast     : Quick test (small LSTM, short sequences)")
        print("  --mode demo     : Demo settings (medium size)")
        print("  --mode full     : Full R2D2 configuration")
        print("  --mode compare  : Compare DQN vs R2D2")
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()