#!/usr/bin/env python3
"""
Working time-matched distributed experiment
Creates a single agent instance and runs once for the target duration
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.DistributedAgentConfig import DistributedAgentConfig
from model import DistributedDQNAgent


def create_optimized_config() -> DistributedAgentConfig:
    """Create optimized configuration for long-running distributed training"""
    config = DistributedAgentConfig()
    config.env_name = 'ALE/Alien-v5'
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.priority_alpha = 0.6
    config.priority_beta_start = 0.4
    config.priority_beta_end = 1.0
    
    # Optimized for long run
    config.num_workers = 4  # Use 4 workers like original
    config.memory_size = 20000  # Standard buffer size
    config.batch_size = 64  # Standard batch size
    config.min_replay_size = 1000
    
    config.learning_rate = 1e-4
    config.target_update_freq = 500
    config.save_interval = 100000  # Don't save during experiment
    
    return config


def calculate_episodes_for_time(target_time: float, throughput_per_minute: float = 56.25) -> int:
    """Calculate how many episodes to run for target time"""
    # Based on observed throughput of 56.25 episodes/minute from 50-episode run
    target_minutes = target_time / 60
    estimated_episodes = int(throughput_per_minute * target_minutes)
    return estimated_episodes


def run_timematched_single_call(target_time: float = 747.8):
    """Run time-matched experiment with a single train_distributed call"""
    
    print("="*80)
    print("TIME-MATCHED DISTRIBUTED EXPERIMENT (Single Call Method)")
    print("="*80)
    print(f"Target time: {target_time:.1f}s")
    
    # Calculate episodes to run
    estimated_episodes = calculate_episodes_for_time(target_time)
    print(f"Estimated episodes for {target_time:.1f}s: {estimated_episodes}")
    print("(Based on 56.25 episodes/minute throughput)")
    
    config = create_optimized_config()
    print(f"\nConfig: {config.num_workers} workers, buffer={config.memory_size}, batch={config.batch_size}")
    
    start_time = time.time()
    
    try:
        # Create agent
        print("\nInitializing agent...")
        agent = DistributedDQNAgent(config, num_workers=config.num_workers)
        print("Agent initialized")
        
        # Run training for estimated episodes
        # Set max_time_seconds to ensure we don't exceed target time
        print(f"\nRunning distributed training for {estimated_episodes} episodes...")
        print(f"Max time limit: {target_time:.1f}s")
        
        results = agent.train_distributed(
            total_episodes=estimated_episodes,
            max_time_seconds=target_time
        )
        
        total_time = time.time() - start_time
        
        # Extract results
        env_stats = results.get('env_stats', {})
        training_stats = results.get('training_stats', {})
        
        # Get actual episodes completed
        actual_episodes = env_stats.get('total_episodes', 0)
        avg_reward = env_stats.get('overall_avg_reward', 0)
        
        # Get per-worker stats
        worker_stats = env_stats.get('worker_stats', [])
        worker_episodes = [w.get('total_episodes', 0) for w in worker_stats]
        worker_rewards = [w.get('avg_reward', 0) for w in worker_stats]
        
        # Calculate throughput
        episodes_per_minute = (actual_episodes / total_time) * 60 if total_time > 0 else 0
        
        # Final results
        final_results = {
            'algorithm': 'Distributed + Priority (time-matched)',
            'target_time': target_time,
            'actual_time': total_time,
            'estimated_episodes': estimated_episodes,
            'actual_episodes': actual_episodes,
            'avg_reward': avg_reward,
            'episodes_per_minute': episodes_per_minute,
            'num_workers': config.num_workers,
            'worker_episodes': worker_episodes,
            'worker_avg_rewards': worker_rewards,
            'total_steps': env_stats.get('total_steps', 0),
            'training_losses': len(training_stats.get('losses', []))
        }
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Target time: {target_time:.1f}s")
        print(f"Actual time: {total_time:.1f}s")
        print(f"Estimated episodes: {estimated_episodes}")
        print(f"Actual episodes: {actual_episodes}")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Throughput: {episodes_per_minute:.2f} episodes/minute")
        
        print(f"\nWorker breakdown:")
        for i, (eps, rew) in enumerate(zip(worker_episodes, worker_rewards)):
            print(f"  Worker {i}: {eps} episodes, avg reward {rew:.2f}")
        
        print(f"\n📊 COMPARISON WITH DQN + PRIORITY:")
        print(f"  DQN + Priority: 50 episodes in {target_time:.1f}s")
        print(f"  Distributed: {actual_episodes} episodes in {total_time:.1f}s")
        print(f"  Episode advantage: {actual_episodes - 50} more episodes")
        print(f"  Throughput advantage: {episodes_per_minute / 4.01:.1f}x")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'experiments/results/timematched_success_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, 'timematched_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Create summary report
        report_file = os.path.join(output_dir, 'timematched_report.md')
        with open(report_file, 'w') as f:
            f.write("# Time-Matched Distributed Training Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n")
            f.write(f"- **Target Time:** {target_time:.1f}s\n")
            f.write(f"- **Actual Time:** {total_time:.1f}s\n")
            f.write(f"- **Episodes Completed:** {actual_episodes}\n")
            f.write(f"- **Average Reward:** {avg_reward:.2f}\n")
            f.write(f"- **Throughput:** {episodes_per_minute:.2f} episodes/minute\n\n")
            f.write("## Comparison with DQN + Priority\n")
            f.write(f"- **DQN + Priority:** 50 episodes in {target_time:.1f}s\n")
            f.write(f"- **Distributed (time-matched):** {actual_episodes} episodes in {total_time:.1f}s\n")
            f.write(f"- **Episode Advantage:** {actual_episodes - 50} more episodes\n")
            f.write(f"- **Speedup Factor:** {episodes_per_minute / 4.01:.1f}x\n")
        
        print(f"Report saved to: {report_file}")
        
        return final_results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run the working time-matched experiment"""
    
    # Target time from DQN + Priority
    target_time = 747.8  # seconds
    
    print("RUNNING TIME-MATCHED DISTRIBUTED EXPERIMENT")
    print("Using single train_distributed call to avoid hanging")
    print()
    
    result = run_timematched_single_call(target_time)
    
    if result:
        print("\n" + "="*80)
        print("✅ EXPERIMENT SUCCESSFUL!")
        print("="*80)
        print("\nKey Results:")
        print(f"- Completed {result['actual_episodes']} episodes in {result['actual_time']:.1f}s")
        print(f"- Average reward: {result['avg_reward']:.2f}")
        print(f"- {result['actual_episodes'] - 50} more episodes than single-threaded")
    else:
        print("\n❌ Experiment failed")


if __name__ == '__main__':
    main()