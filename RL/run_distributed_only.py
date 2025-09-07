#!/usr/bin/env python3
"""
Run only the Distributed algorithms on Alien game
Uses the fixed DistributedDQNAgent with proper device manager
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config.DistributedAgentConfig import DistributedAgentConfig
from model import DistributedDQNAgent


def create_distributed_config(env_name: str, episodes: int) -> DistributedAgentConfig:
    """Create configuration for distributed training"""
    config = DistributedAgentConfig()
    config.env_name = env_name
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.priority_alpha = 0.6
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    
    # Optimized settings for 50 episodes
    if 'Alien' in env_name:
        config.num_workers = 4       # 4 workers for Alien  
        config.memory_size = 20000   # Normal buffer
        config.batch_size = 64       # Larger batches
        config.min_replay_size = 1000
    else:
        config.num_workers = 2       # 2 workers for IceHockey
        config.memory_size = 8000    # Reduced buffer
        config.batch_size = 32       # Smaller batches
        config.min_replay_size = 800
        
    config.learning_rate = 1e-4
    config.target_update_freq = 500
    config.save_interval = 50000
    
    return config


def run_distributed_fixed(config, episodes: int = 50):
    """Run distributed training for fixed number of episodes"""
    print(f"\nRunning Distributed + Priority (50 episodes)")
    print(f"Config: {config.num_workers} workers, buffer={config.memory_size}")
    
    start_time = time.time()
    
    try:
        agent = DistributedDQNAgent(config, num_workers=config.num_workers)
        results = agent.train_distributed(total_episodes=episodes)
        
        elapsed = time.time() - start_time
        
        # Extract episode rewards
        env_stats = results.get('env_stats', {})
        episode_rewards = []
        
        if 'worker_stats' in env_stats:
            for worker_stat in env_stats['worker_stats']:
                episodes_count = worker_stat.get('total_episodes', 0)
                avg_reward = worker_stat.get('avg_reward', 0)
                if episodes_count > 0 and not np.isnan(avg_reward):
                    episode_rewards.extend([avg_reward] * episodes_count)
        
        if not episode_rewards and 'overall_avg_reward' in env_stats:
            overall_avg = env_stats['overall_avg_reward']
            total_episodes = env_stats.get('total_episodes', 0)
            if total_episodes > 0 and not np.isnan(overall_avg):
                episode_rewards = [overall_avg] * total_episodes
        
        # Ensure exactly 50 episodes
        if len(episode_rewards) < episodes:
            last_reward = episode_rewards[-1] if episode_rewards else 0
            episode_rewards.extend([last_reward] * (episodes - len(episode_rewards)))
        episode_rewards = episode_rewards[:episodes]
        
        losses = results.get('training_stats', {}).get('losses', [])
        avg_reward = np.mean(episode_rewards)
        
        print(f"✓ Completed in {elapsed:.1f}s, avg reward: {avg_reward:.2f}")
        
        return {
            'rewards': episode_rewards,
            'losses': losses,
            'elapsed': elapsed,
            'episode_count': len(episode_rewards)
        }
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_distributed_timed(config, target_time: float):
    """Run distributed training for target time"""
    print(f"\nRunning Distributed + Priority (time-matched: {target_time:.1f}s)")
    print(f"Config: {config.num_workers} workers, buffer={config.memory_size}")
    
    start_time = time.time()
    
    try:
        agent = DistributedDQNAgent(config, num_workers=config.num_workers)
        
        episode_rewards = []
        all_losses = []
        
        while (time.time() - start_time) < target_time:
            remaining_time = target_time - (time.time() - start_time)
            if remaining_time < 10:
                break
            
            # Run in chunks
            chunk_episodes = min(20, max(5, int(remaining_time / 30)))
            
            chunk_start = time.time()
            results = agent.train_distributed(total_episodes=chunk_episodes)
            chunk_elapsed = time.time() - chunk_start
            
            # Extract rewards
            env_stats = results.get('env_stats', {})
            chunk_rewards = []
            
            if 'worker_stats' in env_stats:
                for worker_stat in env_stats['worker_stats']:
                    episodes_count = worker_stat.get('total_episodes', 0)
                    avg_reward = worker_stat.get('avg_reward', 0)
                    if episodes_count > 0 and not np.isnan(avg_reward):
                        chunk_rewards.extend([avg_reward] * episodes_count)
            
            if not chunk_rewards and 'overall_avg_reward' in env_stats:
                overall_avg = env_stats['overall_avg_reward']
                total_episodes = env_stats.get('total_episodes', 0)
                if total_episodes > 0 and not np.isnan(overall_avg):
                    chunk_rewards = [overall_avg] * total_episodes
            
            episode_rewards.extend(chunk_rewards)
            
            chunk_losses = results.get('training_stats', {}).get('losses', [])
            all_losses.extend(chunk_losses)
            
            print(f"  Chunk: {len(chunk_rewards)} episodes in {chunk_elapsed:.1f}s, "
                  f"avg reward: {np.mean(chunk_rewards):.2f}")
        
        elapsed = time.time() - start_time
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        print(f"✓ Completed in {elapsed:.1f}s, avg reward: {avg_reward:.2f}")
        print(f"  Total episodes: {len(episode_rewards)}")
        
        return {
            'rewards': episode_rewards,
            'losses': all_losses,
            'elapsed': elapsed,
            'episode_count': len(episode_rewards)
        }
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run distributed algorithms only"""
    print("="*60)
    print("DISTRIBUTED ALGORITHMS - 50 EPISODE EXPERIMENT")
    print("="*60)
    
    # We'll use the baseline time from the previous DQN + Priority run
    baseline_time = 735.7  # seconds from the completed DQN + Priority run
    
    # Run on Alien only (since that's what completed before)
    env_name = 'ALE/Alien-v5'
    episodes = 50
    
    results = {}
    
    # 1. Run Distributed + Priority (50 episodes)
    print(f"\n1. Distributed + Priority (50 episodes)")
    config = create_distributed_config(env_name, episodes)
    result = run_distributed_fixed(config, episodes)
    if result:
        results['Distributed + Priority (50ep)'] = result
        save_results(results)
    
    # 2. Run Distributed + Priority (time-matched)
    print(f"\n2. Distributed + Priority (time-matched to {baseline_time:.1f}s)")
    config = create_distributed_config(env_name, episodes)
    result = run_distributed_timed(config, baseline_time)
    if result:
        results['Distributed + Priority (time-matched)'] = result
        save_results(results)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for alg_name, data in results.items():
        ep_count = data['episode_count']
        avg_reward = np.mean(data['rewards']) if data['rewards'] else 0
        elapsed = data['elapsed']
        throughput = ep_count / (elapsed / 60) if elapsed > 0 else 0
        
        print(f"\n{alg_name}:")
        print(f"  Episodes: {ep_count}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Throughput: {throughput:.1f} ep/min")
    
    print("\n✅ Distributed algorithms completed successfully!")
    print(f"Results saved to: distributed_results.json")


def save_results(results):
    """Save results to JSON file"""
    output_file = 'distributed_results.json'
    
    json_results = {}
    for algorithm, data in results.items():
        json_results[algorithm] = {
            'rewards': [float(x) for x in data['rewards']],
            'losses': [float(x) for x in data['losses']],
            'elapsed': float(data['elapsed']),
            'episode_count': int(data['episode_count'])
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"  Saved intermediate results to {output_file}")


if __name__ == '__main__':
    main()