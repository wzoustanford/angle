#!/usr/bin/env python3
"""
Test script to verify distributed algorithms work correctly
Tests with minimal episodes to ensure no crashes
"""

import sys
import os
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config.DistributedAgentConfig import DistributedAgentConfig
from model import DistributedDQNAgent


def create_test_config(env_name: str) -> DistributedAgentConfig:
    """Create a minimal test configuration"""
    config = DistributedAgentConfig()
    config.env_name = env_name
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.priority_alpha = 0.6
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    
    # Minimal settings for testing
    config.num_workers = 2       # Just 2 workers for testing
    config.memory_size = 1000    # Small buffer
    config.batch_size = 16       # Small batch
    config.min_replay_size = 100 # Start quickly
    
    config.learning_rate = 1e-4
    config.target_update_freq = 100
    config.save_interval = 50000
    
    return config


def test_distributed_agent(env_name: str, test_episodes: int = 2):
    """Test distributed agent with a few episodes"""
    print(f"\n{'='*60}")
    print(f"Testing Distributed DQN on {env_name}")
    print(f"Test episodes: {test_episodes}")
    print(f"{'='*60}")
    
    try:
        # Create configuration
        config = create_test_config(env_name)
        print(f"✓ Configuration created")
        print(f"  - Workers: {config.num_workers}")
        print(f"  - Memory size: {config.memory_size}")
        print(f"  - Batch size: {config.batch_size}")
        
        # Initialize agent
        print("\nInitializing DistributedDQNAgent...")
        agent = DistributedDQNAgent(config, num_workers=config.num_workers)
        print("✓ Agent initialized successfully")
        
        # Run training
        print(f"\nRunning {test_episodes} test episodes...")
        start_time = time.time()
        
        results = agent.train_distributed(total_episodes=test_episodes)
        
        elapsed = time.time() - start_time
        print(f"✓ Training completed in {elapsed:.1f}s")
        
        # Check results
        if results and 'env_stats' in results:
            env_stats = results['env_stats']
            
            # Extract episode information
            total_episodes = env_stats.get('total_episodes', 0)
            overall_avg_reward = env_stats.get('overall_avg_reward', 0)
            
            print(f"\nResults:")
            print(f"  - Total episodes: {total_episodes}")
            print(f"  - Average reward: {overall_avg_reward:.2f}")
            
            if 'worker_stats' in env_stats:
                print(f"  - Worker stats:")
                for i, worker_stat in enumerate(env_stats['worker_stats']):
                    worker_episodes = worker_stat.get('total_episodes', 0)
                    worker_avg = worker_stat.get('avg_reward', 0)
                    print(f"    Worker {i}: {worker_episodes} episodes, avg reward: {worker_avg:.2f}")
        
        # Check training stats
        if results and 'training_stats' in results:
            training_stats = results['training_stats']
            losses = training_stats.get('losses', [])
            if losses:
                print(f"  - Training losses recorded: {len(losses)}")
                print(f"  - Avg loss: {np.mean(losses):.4f}")
        
        print(f"\n✅ Test PASSED for {env_name}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED for {env_name}")
        print(f"Error: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return False


def main():
    """Run tests for both games"""
    print("="*60)
    print("DISTRIBUTED ALGORITHM TEST SUITE")
    print("="*60)
    
    games = [
        ('ALE/Alien-v5', 'Alien'),
        ('ALE/IceHockey-v5', 'IceHockey')
    ]
    
    results = {}
    
    for env_name, game_name in games:
        success = test_distributed_agent(env_name, test_episodes=2)
        results[game_name] = success
        
        # Add delay between tests
        if env_name != games[-1][0]:
            print("\nWaiting 5 seconds before next test...")
            time.sleep(5)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for game_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{game_name}: {status}")
        all_passed = all_passed and success
    
    if all_passed:
        print("\n🎉 All tests passed! Distributed algorithms are working.")
        print("You can now run the full 50-episode experiment.")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running the full experiment.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)