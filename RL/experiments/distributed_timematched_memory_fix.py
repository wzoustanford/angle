#!/usr/bin/env python3
"""
Memory-Optimized Distributed Time-Matched Experiment
Fixes OOM issues by implementing memory management strategies
"""

import sys
import os
import time
import json
import gc
import numpy as np
from datetime import datetime
import traceback
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.DistributedAgentConfig import DistributedAgentConfig
from model import DistributedDQNAgent


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_memory_optimized_config() -> DistributedAgentConfig:
    """Create memory-optimized configuration for distributed training"""
    config = DistributedAgentConfig()
    config.env_name = 'ALE/Alien-v5'
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.priority_alpha = 0.6
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    
    # Memory-optimized settings
    config.num_workers = 3  # Reduced from 4 to 3 workers
    config.memory_size = 10000  # Reduced from 20000 to 10000
    config.batch_size = 32  # Reduced from 64 to 32
    config.min_replay_size = 500  # Reduced from 1000
    
    config.learning_rate = 1e-4
    config.target_update_freq = 500
    config.save_interval = 50000
    
    return config


def run_distributed_timed_with_memory_management(target_time: float, output_dir: str):
    """Run time-matched distributed training with memory management"""
    
    log_file = os.path.join(output_dir, 'timematched_memory_optimized.log')
    
    def log(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
    
    log("="*80)
    log("MEMORY-OPTIMIZED DISTRIBUTED TIME-MATCHED EXPERIMENT")
    log("="*80)
    log(f"Target time: {target_time:.1f}s")
    log(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    config = create_memory_optimized_config()
    log(f"Config: workers={config.num_workers}, buffer={config.memory_size}, batch={config.batch_size}")
    
    start_time = time.time()
    
    try:
        # Initialize agent
        log("\nInitializing DistributedDQNAgent...")
        agent = DistributedDQNAgent(config, num_workers=config.num_workers)
        log(f"Agent initialized. Memory: {get_memory_usage():.1f} MB")
        
        all_episode_rewards = []
        all_losses = []
        total_episodes_run = 0
        chunk_count = 0
        memory_threshold_mb = 12000  # Set memory threshold at 12GB
        
        # Run in chunks with memory management
        while (time.time() - start_time) < target_time:
            remaining_time = target_time - (time.time() - start_time)
            if remaining_time < 5:
                break
            
            # Check memory before each chunk
            current_memory = get_memory_usage()
            log(f"\nChunk {chunk_count + 1} - Memory: {current_memory:.1f} MB")
            
            if current_memory > memory_threshold_mb:
                log(f"WARNING: Memory usage ({current_memory:.1f} MB) exceeds threshold ({memory_threshold_mb} MB)")
                log("Attempting memory cleanup...")
                
                # Clear replay buffer if it's too large
                if hasattr(agent, 'replay_buffer'):
                    buffer_size = len(agent.replay_buffer)
                    if buffer_size > 5000:
                        # Keep only recent experiences
                        log(f"Reducing replay buffer from {buffer_size} to 5000")
                        agent.replay_buffer.buffer = agent.replay_buffer.buffer[-5000:]
                
                # Force garbage collection
                gc.collect()
                new_memory = get_memory_usage()
                log(f"Memory after cleanup: {new_memory:.1f} MB (freed {current_memory - new_memory:.1f} MB)")
            
            # Adaptive chunk size based on remaining time and memory
            if current_memory > 10000:  # If using more than 10GB
                chunk_episodes = 5  # Small chunks
            elif current_memory > 8000:  # If using more than 8GB
                chunk_episodes = 10  # Medium chunks
            else:
                chunk_episodes = min(20, max(5, int(remaining_time / 30)))  # Normal chunks
            
            log(f"Running {chunk_episodes} episodes (remaining time: {remaining_time:.1f}s)")
            
            chunk_start = time.time()
            
            # Run training chunk
            results = agent.train_distributed(total_episodes=chunk_episodes)
            
            chunk_time = time.time() - chunk_start
            
            # Extract results
            env_stats = results.get('env_stats', {})
            training_stats = results.get('training_stats', {})
            
            # Extract episode rewards
            chunk_rewards = []
            if 'worker_stats' in env_stats:
                for worker_stat in env_stats['worker_stats']:
                    worker_rewards = worker_stat.get('episode_rewards', [])
                    if worker_rewards:
                        chunk_rewards.extend(worker_rewards)
                    else:
                        # Fallback to average
                        eps = worker_stat.get('total_episodes', 0)
                        avg = worker_stat.get('avg_reward', 0)
                        if eps > 0:
                            chunk_rewards.extend([avg] * eps)
            
            # Fallback if no detailed rewards
            if not chunk_rewards and 'overall_avg_reward' in env_stats:
                avg = env_stats['overall_avg_reward']
                eps = env_stats.get('total_episodes', chunk_episodes)
                chunk_rewards = [avg] * eps
            
            all_episode_rewards.extend(chunk_rewards)
            
            # Extract losses
            chunk_losses = training_stats.get('losses', [])
            all_losses.extend(chunk_losses)
            
            total_episodes_run += len(chunk_rewards)
            chunk_count += 1
            
            log(f"  Chunk completed: {len(chunk_rewards)} episodes in {chunk_time:.1f}s")
            log(f"  Chunk avg reward: {np.mean(chunk_rewards):.2f}")
            log(f"  Total episodes so far: {total_episodes_run}")
            log(f"  Memory usage: {get_memory_usage():.1f} MB")
            
            # Periodic garbage collection every 5 chunks
            if chunk_count % 5 == 0:
                log("Performing periodic garbage collection...")
                gc.collect()
        
        total_time = time.time() - start_time
        
        # Final results
        results = {
            'algorithm': 'Distributed + Priority (time-matched)',
            'episode_rewards': all_episode_rewards,
            'losses': all_losses,
            'total_time': total_time,
            'target_time': target_time,
            'avg_reward': np.mean(all_episode_rewards) if all_episode_rewards else 0,
            'std_reward': np.std(all_episode_rewards) if all_episode_rewards else 0,
            'max_reward': np.max(all_episode_rewards) if all_episode_rewards else 0,
            'min_reward': np.min(all_episode_rewards) if all_episode_rewards else 0,
            'total_episodes': len(all_episode_rewards),
            'episodes_per_minute': (len(all_episode_rewards) / total_time) * 60 if total_time > 0 else 0,
            'num_workers': config.num_workers,
            'total_chunks': chunk_count,
            'final_memory_mb': get_memory_usage()
        }
        
        log("\n" + "="*80)
        log("EXPERIMENT COMPLETED SUCCESSFULLY!")
        log("="*80)
        log(f"Time: {total_time:.1f}s (target was {target_time:.1f}s)")
        log(f"Episodes: {results['total_episodes']}")
        log(f"Avg reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        log(f"Throughput: {results['episodes_per_minute']:.2f} ep/min")
        log(f"Final memory: {results['final_memory_mb']:.1f} MB")
        
        # Save results
        results_file = os.path.join(output_dir, 'timematched_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"\nResults saved to: {results_file}")
        
        return results
        
    except Exception as e:
        log(f"\nERROR: {e}")
        traceback.print_exc()
        
        # Save partial results if available
        if all_episode_rewards:
            partial_results = {
                'algorithm': 'Distributed + Priority (time-matched) - PARTIAL',
                'episode_rewards': all_episode_rewards,
                'total_episodes': len(all_episode_rewards),
                'error': str(e),
                'completed_time': time.time() - start_time
            }
            
            error_file = os.path.join(output_dir, 'timematched_partial_results.json')
            with open(error_file, 'w') as f:
                json.dump(partial_results, f, indent=2)
            log(f"Partial results saved to: {error_file}")
        
        return None


def main():
    """Run memory-optimized time-matched experiment"""
    
    # Create output directory
    output_dir = f'experiments/results/timematched_memory_fix_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Target time from DQN + Priority baseline
    target_time = 747.8  # seconds
    
    print(f"Starting memory-optimized time-matched experiment")
    print(f"Output directory: {output_dir}")
    print(f"Target time: {target_time:.1f}s")
    
    # First run a short test
    print("\n" + "="*80)
    print("RUNNING SHORT TEST (60 seconds)")
    print("="*80)
    
    test_results = run_distributed_timed_with_memory_management(60.0, output_dir)
    
    if test_results and test_results['total_episodes'] > 0:
        print("\n✅ Short test successful!")
        print(f"Generated {test_results['total_episodes']} episodes in 60s")
        print(f"Memory usage remained at {test_results['final_memory_mb']:.1f} MB")
        
        # Now run the full experiment
        print("\n" + "="*80)
        print("RUNNING FULL TIME-MATCHED EXPERIMENT")
        print("="*80)
        
        full_results = run_distributed_timed_with_memory_management(target_time, output_dir)
        
        if full_results:
            print("\n✅ Full experiment completed successfully!")
            
            # Compare with single-threaded baseline
            print("\n" + "="*80)
            print("COMPARISON WITH DQN + PRIORITY BASELINE")
            print("="*80)
            print(f"DQN + Priority: 50 episodes in {target_time:.1f}s")
            print(f"Distributed (time-matched): {full_results['total_episodes']} episodes in {full_results['total_time']:.1f}s")
            print(f"Episode advantage: {full_results['total_episodes'] - 50} more episodes")
            print(f"Throughput advantage: {full_results['episodes_per_minute'] / 4.01:.1f}x")
    else:
        print("\n❌ Short test failed. Please check memory settings.")


if __name__ == '__main__':
    main()