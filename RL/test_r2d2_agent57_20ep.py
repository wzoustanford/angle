#!/usr/bin/env python3
"""
Test R2D2+Agent57 Hybrid for 20 episodes
Single algorithm to avoid memory issues
"""

import sys
import os
import time
import json
import gc
import torch
import psutil
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.NGUConfig import Agent57Config
from model.r2d2_agent57_hybrid import R2D2Agent57Hybrid


def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    mem_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
    return mem_mb, mem_gb


def main():
    print("="*60)
    print("R2D2+Agent57 20-Episode Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Configuration
    episodes = 20
    game = 'ALE/Alien-v5'
    
    # Memory check
    initial_mem, available_gb = get_memory_info()
    print(f"Initial memory: {initial_mem:.1f}MB")
    print(f"Available memory: {available_gb:.1f}GB")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./experiments/results/r2d2_agent57_20ep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure R2D2+Agent57
    config = Agent57Config()
    config.env_name = game
    config.num_policies = 8  # Full 8 policies
    config.memory_size = 5000  # Small buffer to save memory
    config.episodic_memory_size = 2000
    config.sequence_length = 40  # Shorter sequences
    config.burn_in_length = 20
    config.max_episode_steps = 2000  # Limit episode length
    
    print(f"\nConfiguration:")
    print(f"  Game: {game}")
    print(f"  Episodes: {episodes}")
    print(f"  Policies: {config.num_policies}")
    print(f"  Memory size: {config.memory_size}")
    print(f"  Max steps: {config.max_episode_steps}")
    print("-"*60)
    
    try:
        # Create agent
        print("\nInitializing R2D2+Agent57...")
        agent = R2D2Agent57Hybrid(config)
        
        # Training metrics
        episode_rewards = []
        intrinsic_rewards = []
        policy_usage = {}
        memory_usage = []
        
        # Run episodes
        print("\nStarting training...")
        start_time = time.time()
        
        for ep in range(episodes):
            ep_start = time.time()
            
            # Train one episode
            stats = agent.train_episode()
            
            # Record metrics
            episode_rewards.append(stats['episode_reward'])
            intrinsic_rewards.append(stats['episode_intrinsic_reward'])
            policy_id = stats['policy_id']
            policy_usage[policy_id] = policy_usage.get(policy_id, 0) + 1
            
            # Check memory
            current_mem, available_gb = get_memory_info()
            mem_increase = current_mem - initial_mem
            memory_usage.append(current_mem)
            
            # Log progress
            ep_time = time.time() - ep_start
            print(f"Episode {ep+1:2d}/{episodes}: "
                  f"Reward={stats['episode_reward']:6.1f}, "
                  f"Intrinsic={stats['episode_intrinsic_reward']:6.1f}, "
                  f"Policy={policy_id}, "
                  f"Time={ep_time:.1f}s, "
                  f"Mem={current_mem:.0f}MB (+{mem_increase:.0f})")
            
            # Check if running low on memory
            if available_gb < 2.0:
                print(f"⚠️  Warning: Low memory ({available_gb:.1f}GB available)")
                
            # Periodic cleanup
            if (ep + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Training complete
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Calculate statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_intrinsic = np.mean(intrinsic_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)
        
        # Memory statistics
        final_mem = memory_usage[-1]
        total_mem_increase = final_mem - initial_mem
        avg_mem_per_ep = total_mem_increase / episodes
        
        print(f"\nPerformance Statistics:")
        print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Max reward: {max_reward:.1f}")
        print(f"  Min reward: {min_reward:.1f}")
        print(f"  Average intrinsic: {avg_intrinsic:.1f}")
        
        print(f"\nPolicy Usage:")
        for pid in sorted(policy_usage.keys()):
            print(f"  Policy {pid}: {policy_usage[pid]} episodes")
        
        print(f"\nMemory Usage:")
        print(f"  Total increase: {total_mem_increase:.1f}MB")
        print(f"  Per episode: {avg_mem_per_ep:.1f}MB")
        
        print(f"\nRuntime:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Per episode: {total_time/episodes:.1f}s")
        
        # Save results
        results = {
            'config': {
                'episodes': episodes,
                'game': game,
                'num_policies': config.num_policies,
                'memory_size': config.memory_size
            },
            'rewards': {
                'episode_rewards': episode_rewards,
                'average': avg_reward,
                'std': std_reward,
                'max': max_reward,
                'min': min_reward
            },
            'intrinsic_rewards': intrinsic_rewards,
            'policy_usage': policy_usage,
            'memory_usage': memory_usage,
            'runtime': {
                'total_seconds': total_time,
                'per_episode': total_time/episodes
            }
        }
        
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Create summary
        summary_file = os.path.join(output_dir, 'summary.md')
        with open(summary_file, 'w') as f:
            f.write("# R2D2+Agent57 20-Episode Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Game:** Alien\n")
            f.write(f"**Episodes:** {episodes}\n\n")
            
            f.write("## Performance\n\n")
            f.write(f"- **Average Reward:** {avg_reward:.2f} ± {std_reward:.2f}\n")
            f.write(f"- **Best Episode:** {max_reward:.1f}\n")
            f.write(f"- **Worst Episode:** {min_reward:.1f}\n")
            f.write(f"- **Avg Intrinsic:** {avg_intrinsic:.1f}\n\n")
            
            f.write("## Episode Progression\n\n")
            f.write("| Episode | Reward | Policy |\n")
            f.write("|---------|--------|--------|\n")
            for i in range(episodes):
                f.write(f"| {i+1} | {episode_rewards[i]:.1f} | {stats['policy_id']} |\n")
        
        print(f"Summary saved to: {summary_file}")
        
        # Cleanup
        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())