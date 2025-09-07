#!/usr/bin/env python3
"""
Ultra-lightweight time-matched distributed experiment
Uses minimal resources to avoid OOM/hanging issues
"""

import sys
import os
import time
import json
import gc
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.DistributedAgentConfig import DistributedAgentConfig
from model import DistributedDQNAgent


def create_ultra_light_config() -> DistributedAgentConfig:
    """Create ultra-lightweight configuration"""
    config = DistributedAgentConfig()
    config.env_name = 'ALE/Alien-v5'
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.priority_alpha = 0.6
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    
    # Ultra-lightweight settings
    config.num_workers = 2  # Only 2 workers
    config.memory_size = 2000  # Very small buffer
    config.batch_size = 16  # Tiny batches
    config.min_replay_size = 200  # Start training quickly
    
    config.learning_rate = 1e-4
    config.target_update_freq = 1000  # Less frequent updates
    config.save_interval = 100000  # Don't save checkpoints
    
    return config


def run_ultra_lightweight_timematched(target_time: float = 747.8):
    """Run ultra-lightweight time-matched experiment"""
    
    print("="*80)
    print("ULTRA-LIGHTWEIGHT TIME-MATCHED DISTRIBUTED EXPERIMENT")
    print("="*80)
    print(f"Target time: {target_time:.1f}s")
    print("Settings: 2 workers, 2000 buffer size, 16 batch size")
    
    config = create_ultra_light_config()
    start_time = time.time()
    
    # Results tracking
    all_rewards = []
    all_losses = []
    total_episodes = 0
    chunks_completed = 0
    
    try:
        # Initialize agent
        print("\nInitializing agent...")
        agent = DistributedDQNAgent(config, num_workers=2)
        print("Agent initialized successfully")
        
        # Run in very small chunks to avoid memory buildup
        chunk_size = 3  # Only 3 episodes per chunk
        
        while (time.time() - start_time) < target_time:
            remaining = target_time - (time.time() - start_time)
            if remaining < 3:
                break
            
            # Clear any accumulated memory every 10 chunks
            if chunks_completed > 0 and chunks_completed % 10 == 0:
                gc.collect()
                print(f"  Performed garbage collection at chunk {chunks_completed}")
            
            # Run small chunk
            chunk_start = time.time()
            results = agent.train_distributed(total_episodes=chunk_size)
            chunk_time = time.time() - chunk_start
            
            # Extract results
            env_stats = results.get('env_stats', {})
            
            # Get rewards
            chunk_rewards = []
            if 'overall_avg_reward' in env_stats:
                avg = env_stats['overall_avg_reward']
                eps = env_stats.get('total_episodes', chunk_size)
                chunk_rewards = [avg] * eps
            
            all_rewards.extend(chunk_rewards)
            total_episodes += len(chunk_rewards)
            chunks_completed += 1
            
            # Progress update every 20 chunks
            if chunks_completed % 20 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {chunks_completed} chunks, {total_episodes} episodes, "
                      f"{elapsed:.1f}s elapsed, avg reward: {np.mean(all_rewards):.2f}")
        
        # Final results
        total_time = time.time() - start_time
        
        results = {
            'algorithm': 'Distributed + Priority (time-matched, ultra-light)',
            'total_episodes': total_episodes,
            'avg_reward': np.mean(all_rewards) if all_rewards else 0,
            'std_reward': np.std(all_rewards) if all_rewards else 0,
            'total_time': total_time,
            'target_time': target_time,
            'episodes_per_minute': (total_episodes / total_time) * 60 if total_time > 0 else 0,
            'chunks_completed': chunks_completed,
            'config': {
                'num_workers': 2,
                'buffer_size': 2000,
                'batch_size': 16
            }
        }
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED!")
        print("="*80)
        print(f"Time: {total_time:.1f}s (target: {target_time:.1f}s)")
        print(f"Episodes: {total_episodes}")
        print(f"Avg reward: {results['avg_reward']:.2f}")
        print(f"Throughput: {results['episodes_per_minute']:.2f} ep/min")
        print(f"Episode advantage over single-threaded: {total_episodes - 50}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'ultra_light_timematched_results_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial results
        if all_rewards:
            partial = {
                'status': 'partial',
                'episodes_completed': total_episodes,
                'avg_reward': np.mean(all_rewards),
                'time_elapsed': time.time() - start_time,
                'error': str(e)
            }
            
            with open('partial_results.json', 'w') as f:
                json.dump(partial, f, indent=2)
            print(f"Partial results saved")
        
        return None


def main():
    """Run the ultra-lightweight experiment"""
    
    # First do a 30-second test
    print("RUNNING 30-SECOND TEST")
    print("-"*80)
    
    test_result = run_ultra_lightweight_timematched(30.0)
    
    if test_result and test_result['total_episodes'] > 0:
        print(f"\n✅ Test successful! Generated {test_result['total_episodes']} episodes in 30s")
        print(f"Projected for 747.8s: {test_result['episodes_per_minute'] * 12.46:.0f} episodes")
        
        # Run full experiment
        print("\n" + "="*80)
        print("RUNNING FULL 747.8-SECOND EXPERIMENT")
        print("="*80)
        
        full_result = run_ultra_lightweight_timematched(747.8)
        
        if full_result:
            print("\n✅ SUCCESS! Time-matched experiment completed")
            print("\nFINAL COMPARISON:")
            print(f"DQN + Priority: 50 episodes in 747.8s")
            print(f"Distributed (ultra-light): {full_result['total_episodes']} episodes in {full_result['total_time']:.1f}s")
            print(f"Episode advantage: {full_result['total_episodes'] - 50} more episodes")
    else:
        print("\n❌ Test failed")


if __name__ == '__main__':
    main()