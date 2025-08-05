#!/usr/bin/env python3
"""
Example usage of prioritized vs uniform replay buffer.

This script demonstrates how to configure and use prioritized experience replay
vs standard uniform replay for DQN training.
"""

import sys
import os

# Add parent directory to path for imports when run from examples folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.AgentConfig import AgentConfig


def example_uniform_replay():
    """Example configuration for uniform replay"""
    print("Example 1: Uniform Replay Buffer")
    print("-" * 40)
    
    config = AgentConfig()
    config.use_prioritized_replay = False
    config.memory_size = 50000
    config.batch_size = 32
    
    print(f"Configuration:")
    print(f"  use_prioritized_replay: {config.use_prioritized_replay}")
    print(f"  memory_size: {config.memory_size}")
    print(f"  batch_size: {config.batch_size}")
    print(f"\n  This will use standard uniform random sampling")
    
    return config


def example_prioritized_replay():
    """Example configuration for prioritized replay"""
    print("Example 2: Prioritized Replay Buffer (TD-Error based)")
    print("-" * 50)
    
    config = AgentConfig()
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.priority_alpha = 0.6  # Moderate prioritization
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    config.memory_size = 50000
    config.batch_size = 32
    
    print(f"Configuration:")
    print(f"  use_prioritized_replay: {config.use_prioritized_replay}")
    print(f"  priority_type: {config.priority_type}")
    print(f"  priority_alpha: {config.priority_alpha} (0=uniform, 1=full priority)")
    print(f"  priority_beta_start: {config.priority_beta_start}")
    print(f"  priority_beta_end: {config.priority_beta_end}")
    print(f"  memory_size: {config.memory_size}")
    print(f"  batch_size: {config.batch_size}")
    print(f"\n  This will prioritize experiences with high TD-errors")
    
    return config


def example_reward_prioritized():
    """Example configuration for reward-based prioritization"""
    print("Example 3: Prioritized Replay Buffer (Reward based)")
    print("-" * 50)
    
    config = AgentConfig()
    config.use_prioritized_replay = True
    config.priority_type = 'reward'
    config.priority_alpha = 0.8  # Higher prioritization for rewards
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    config.memory_size = 50000
    config.batch_size = 32
    
    print(f"Configuration:")
    print(f"  use_prioritized_replay: {config.use_prioritized_replay}")
    print(f"  priority_type: {config.priority_type}")
    print(f"  priority_alpha: {config.priority_alpha}")
    print(f"  This will prioritize experiences with high absolute rewards")
    
    return config


def example_random_prioritized():
    """Example configuration for random prioritization (for comparison)"""
    print("Example 4: Random Prioritized Replay Buffer")
    print("-" * 45)
    
    config = AgentConfig()
    config.use_prioritized_replay = True
    config.priority_type = 'random'
    config.priority_alpha = 0.6
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    
    print(f"Configuration:")
    print(f"  use_prioritized_replay: {config.use_prioritized_replay}")
    print(f"  priority_type: {config.priority_type}")
    print(f"  This assigns random priorities (baseline for comparison)")
    
    return config


def training_example():
    """Show how to use the configurations for training"""
    print("\nTraining Example:")
    print("=" * 60)
    
    print("""
# To train with uniform replay:
from model import DQNAgent
from config import AgentConfig

config = AgentConfig()
config.use_prioritized_replay = False
agent = DQNAgent(config)
episode_rewards, losses = agent.train(num_episodes=100)

# To train with prioritized replay:
config = AgentConfig()
config.use_prioritized_replay = True
config.priority_type = 'td_error'  # or 'reward' or 'random'
config.priority_alpha = 0.6        # prioritization strength
config.priority_beta_start = 0.4   # importance sampling start
config.priority_beta_end = 1.0     # importance sampling end
agent = DQNAgent(config)
episode_rewards, losses = agent.train(num_episodes=100)
""")


def test_priority_configuration():
    """Test that prioritized replay configurations work"""
    print("Testing Prioritized Replay Configurations")
    print("=" * 50)
    
    try:
        from model import DQNAgent
        
        # Test uniform replay
        print("✓ Testing uniform replay configuration...")
        config = AgentConfig()
        config.use_prioritized_replay = False
        config.memory_size = 1000  # Small for testing
        agent = DQNAgent(config)
        print(f"  Buffer type: {type(agent.replay_buffer).__name__}")
        
        # Test prioritized replay
        print("✓ Testing prioritized replay configuration...")
        config = AgentConfig()
        config.use_prioritized_replay = True
        config.priority_type = 'td_error'
        config.memory_size = 1000  # Small for testing
        agent = DQNAgent(config)
        print(f"  Buffer type: {type(agent.replay_buffer).__name__}")
        print(f"  Priority type: {agent.replay_buffer.priority_type}")
        print(f"  Alpha: {agent.replay_buffer.alpha}")
        print(f"  Beta: {agent.replay_buffer.beta}")
        
        print("\n✅ All configurations work correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Show all configuration examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Priority Replay Buffer Examples')
    parser.add_argument('--mode', choices=['examples', 'test'], default='examples',
                       help='Mode to run: examples or test')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        success = test_priority_configuration()
        return 0 if success else 1
    
    print("Prioritized Replay Buffer Configuration Examples")
    print("=" * 60)
    
    # Show different configurations
    example_uniform_replay()
    print("\n")
    
    example_prioritized_replay()
    print("\n")
    
    example_reward_prioritized()
    print("\n")
    
    example_random_prioritized()
    print("\n")
    
    training_example()
    
    print("\nKey Parameters Explained:")
    print("-" * 30)
    print("• use_prioritized_replay: Enable/disable prioritized sampling")
    print("• priority_type: 'td_error', 'reward', or 'random'")
    print("• priority_alpha: Prioritization exponent (0=uniform, 1=full priority)")
    print("• priority_beta_start/end: Importance sampling correction (anneals from start to end)")
    print("• priority_epsilon: Small constant to prevent zero priorities")
    
    print("\nRecommended Settings:")
    print("-" * 20)
    print("• For most RL tasks: priority_type='td_error', alpha=0.6, beta_start=0.4")
    print("• For sparse rewards: priority_type='reward', alpha=0.8")
    print("• For comparison: priority_type='random' (should perform similar to uniform)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())