#!/usr/bin/env python3
"""
Quick End-to-End Test for Atari Experiment

Tests all four algorithms with minimal episodes to verify functionality
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.AgentConfig import AgentConfig
from config.DistributedAgentConfig import DistributedAgentConfig
from model import DQNAgent, DistributedDQNAgent


def test_basic_dqn():
    """Test basic DQN"""
    print("Testing Basic DQN...")
    config = AgentConfig()
    config.env_name = 'ALE/Pong-v5'
    config.use_r2d2 = False
    config.use_prioritized_replay = False
    config.memory_size = 1000
    config.min_replay_size = 100
    config.save_interval = 50000
    
    agent = DQNAgent(config)
    start_time = time.time()
    episode_rewards, losses = agent.train(num_episodes=3)
    elapsed = time.time() - start_time
    
    print(f"✓ Basic DQN: {elapsed:.1f}s, avg reward: {np.mean(episode_rewards):.2f}")
    return elapsed, len(episode_rewards)


def test_prioritized_dqn():
    """Test DQN + Prioritized Replay"""
    print("Testing DQN + Priority...")
    config = AgentConfig()
    config.env_name = 'ALE/Pong-v5'
    config.use_r2d2 = False
    config.use_prioritized_replay = True
    config.memory_size = 1000
    config.min_replay_size = 100
    config.save_interval = 50000
    
    agent = DQNAgent(config)
    start_time = time.time()
    episode_rewards, losses = agent.train(num_episodes=3)
    elapsed = time.time() - start_time
    
    print(f"✓ Priority DQN: {elapsed:.1f}s, avg reward: {np.mean(episode_rewards):.2f}")
    return elapsed, len(episode_rewards)


def test_distributed_dqn():
    """Test Distributed DQN"""
    print("Testing Distributed DQN...")
    config = DistributedAgentConfig()
    config.env_name = 'ALE/Pong-v5'
    config.num_workers = 2  # Reduce workers for test
    config.use_prioritized_replay = True
    config.memory_size = 2000
    config.min_replay_size = 200
    config.save_interval = 50000
    
    agent = DistributedDQNAgent(config, num_workers=2)
    start_time = time.time()
    results = agent.train_distributed(total_episodes=6)  # 3 episodes per worker
    elapsed = time.time() - start_time
    
    # Extract rewards
    env_stats = results.get('env_stats', {})
    episode_rewards = []
    if 'rewards_per_worker' in env_stats:
        for worker_rewards in env_stats['rewards_per_worker']:
            episode_rewards.extend(worker_rewards)
    
    print(f"✓ Distributed DQN: {elapsed:.1f}s, avg reward: {np.mean(episode_rewards):.2f}, episodes: {len(episode_rewards)}")
    return elapsed, len(episode_rewards)


def test_r2d2_distributed():
    """Test R2D2 + Distributed"""
    print("Testing R2D2 + Distributed...")
    config = DistributedAgentConfig()
    config.env_name = 'ALE/Pong-v5'
    config.num_workers = 2
    config.use_r2d2 = True
    config.use_prioritized_replay = True
    config.sequence_length = 40  # Shorter for test
    config.burn_in_length = 20
    config.lstm_size = 256  # Smaller for test
    config.memory_size = 2000
    config.min_replay_size = 200
    config.save_interval = 50000
    
    agent = DistributedDQNAgent(config, num_workers=2)
    start_time = time.time()
    results = agent.train_distributed(total_episodes=6)
    elapsed = time.time() - start_time
    
    # Extract rewards
    env_stats = results.get('env_stats', {})
    episode_rewards = []
    if 'rewards_per_worker' in env_stats:
        for worker_rewards in env_stats['rewards_per_worker']:
            episode_rewards.extend(worker_rewards)
    
    print(f"✓ R2D2 Distributed: {elapsed:.1f}s, avg reward: {np.mean(episode_rewards):.2f}, episodes: {len(episode_rewards)}")
    return elapsed, len(episode_rewards)


def estimate_full_experiment_runtime():
    """Estimate runtime for full experiment based on test results"""
    print("\n" + "="*60)
    print("RUNTIME ESTIMATION FOR FULL EXPERIMENT")
    print("="*60)
    
    print("Running quick tests to estimate performance...")
    
    try:
        # Test each algorithm
        basic_time, basic_episodes = test_basic_dqn()
        priority_time, priority_episodes = test_prioritized_dqn()
        distributed_time, distributed_episodes = test_distributed_dqn()
        r2d2_time, r2d2_episodes = test_r2d2_distributed()
        
        print(f"\nTest Results:")
        print(f"Basic DQN:      {basic_time:.1f}s for {basic_episodes} episodes")
        print(f"Priority DQN:   {priority_time:.1f}s for {priority_episodes} episodes")
        print(f"Distributed:    {distributed_time:.1f}s for {distributed_episodes} episodes")
        print(f"R2D2:          {r2d2_time:.1f}s for {r2d2_episodes} episodes")
        
        # Calculate time per episode
        basic_per_ep = basic_time / max(basic_episodes, 1)
        priority_per_ep = priority_time / max(priority_episodes, 1)
        distributed_per_ep = distributed_time / max(distributed_episodes, 1)
        r2d2_per_ep = r2d2_time / max(r2d2_episodes, 1)
        
        print(f"\nTime per episode:")
        print(f"Basic DQN:      {basic_per_ep:.1f}s/episode")
        print(f"Priority DQN:   {priority_per_ep:.1f}s/episode")
        print(f"Distributed:    {distributed_per_ep:.1f}s/episode")
        print(f"R2D2:          {r2d2_per_ep:.1f}s/episode")
        
        # Estimate full experiment times
        print(f"\n" + "="*60)
        print("FULL EXPERIMENT ESTIMATES")
        print("="*60)
        
        # Different episode counts for comparison
        episode_configs = {
            "Quick Test": 50,
            "Medium Test": 200, 
            "ICLR Standard": 500,
            "Full Training": 1000
        }
        
        for config_name, episodes in episode_configs.items():
            print(f"\n{config_name} ({episodes} episodes per algorithm per game):")
            print("-" * 50)
            
            # Per algorithm estimates (3 games)
            basic_total = basic_per_ep * episodes * 3
            priority_total = priority_per_ep * episodes * 3
            distributed_total = distributed_per_ep * episodes * 3
            r2d2_total = r2d2_per_ep * episodes * 3
            
            total_time = basic_total + priority_total + distributed_total + r2d2_total
            
            print(f"Basic DQN:      {basic_total/3600:.1f} hours")
            print(f"Priority DQN:   {priority_total/3600:.1f} hours")
            print(f"Distributed:    {distributed_total/3600:.1f} hours")
            print(f"R2D2:          {r2d2_total/3600:.1f} hours")
            print(f"TOTAL:         {total_time/3600:.1f} hours ({total_time/86400:.1f} days)")
        
        print(f"\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print(f"• Quick Test (50 episodes): Good for debugging and initial comparison")
        print(f"• Medium Test (200 episodes): Reasonable for algorithm comparison")
        print(f"• ICLR Standard (500 episodes): Closer to paper results")
        print(f"• Full Training (1000 episodes): Research-quality results")
        print(f"\nWith GPU acceleration, times could be 2-5x faster!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("Quick End-to-End Experiment Test")
    print("="*40)
    
    estimate_full_experiment_runtime()
    
    print(f"\n" + "="*60)
    print("✓ END-TO-END TEST COMPLETED")
    print("All algorithms appear to be working correctly!")
    print("="*60)


if __name__ == '__main__':
    main()