#!/usr/bin/env python3
"""
Example script demonstrating distributed reinforcement learning for Space Invaders.

This script shows how to use the distributed DQN implementation to train an agent
using multiple parallel environments for faster data collection.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import sys
import os
from typing import List

# Add parent directory to path for imports when run from examples folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import DistributedDQNAgent
from config import DistributedAgentConfig


def plot_distributed_training_results(stats_history: List[dict], save_path: str = './results/distributed_training.png'):
    """Plot distributed training results"""
    
    if not stats_history:
        print("No training statistics to plot")
        return
    
    # Extract data from statistics
    episodes = [stats['env_stats']['total_episodes'] for stats in stats_history if 'env_stats' in stats]
    avg_rewards = [stats['env_stats'].get('overall_recent_avg', 0) for stats in stats_history if 'env_stats' in stats]
    buffer_sizes = [stats['buffer_stats']['size'] for stats in stats_history if 'buffer_stats' in stats]
    fill_ratios = [stats['buffer_stats']['fill_ratio'] for stats in stats_history if 'buffer_stats' in stats]
    
    if not episodes:
        print("No episode data to plot")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Episodes vs Average Reward
    ax1.plot(episodes, avg_rewards, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Total Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Progress: Reward vs Episodes')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Buffer Size Over Time
    ax2.plot(episodes, buffer_sizes, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Total Episodes')
    ax2.set_ylabel('Buffer Size')
    ax2.set_title('Replay Buffer Growth')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Buffer Fill Ratio
    ax3.plot(episodes, [f * 100 for f in fill_ratios], 'r-', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Total Episodes')
    ax3.set_ylabel('Buffer Fill Ratio (%)')
    ax3.set_title('Buffer Fill Ratio Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Worker Statistics (if available)
    if stats_history and 'env_stats' in stats_history[-1] and 'worker_stats' in stats_history[-1]['env_stats']:
        worker_stats = stats_history[-1]['env_stats']['worker_stats']
        worker_ids = [w['worker_id'] for w in worker_stats]
        worker_episodes = [w['total_episodes'] for w in worker_stats]
        worker_rewards = [w.get('recent_avg_reward', 0) for w in worker_stats]
        
        ax4.bar(worker_ids, worker_episodes, alpha=0.7, label='Episodes')
        ax4_twin = ax4.twinx()
        ax4_twin.bar([w + 0.4 for w in worker_ids], worker_rewards, alpha=0.7, color='orange', label='Avg Reward')
        ax4.set_xlabel('Worker ID')
        ax4.set_ylabel('Episodes', color='blue')
        ax4_twin.set_ylabel('Average Reward', color='orange')
        ax4.set_title('Per-Worker Statistics')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'Worker statistics\nnot available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Per-Worker Statistics')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training plots saved to {save_path}")


def compare_single_vs_distributed(episodes: int = 100):
    """Compare single-threaded vs distributed training performance"""
    print("=" * 80)
    print("COMPARING SINGLE-THREADED VS DISTRIBUTED TRAINING")
    print("=" * 80)
    
    # Single-threaded configuration
    single_config = DistributedAgentConfig(
        num_workers=1,
        env_name='ALE/SpaceInvaders-v5',
        memory_size=10000,
        min_replay_size=1000,
        batch_size=32
    )
    
    # Distributed configuration
    distributed_config = DistributedAgentConfig(
        num_workers=4,
        env_name='ALE/SpaceInvaders-v5',
        memory_size=20000,
        min_replay_size=2000,
        batch_size=64
    )
    
    results = {}
    
    for name, config in [("Single-threaded", single_config), ("Distributed", distributed_config)]:
        print(f"\nTesting {name} approach...")
        print(f"Workers: {config.num_workers}")
        print(f"Batch size: {config.batch_size}")
        print(f"Buffer size: {config.memory_size}")
        
        agent = DistributedDQNAgent(config, num_workers=config.num_workers)
        
        try:
            start_time = time.time()
            
            # Run training
            final_stats = agent.train_batch_collection(
                total_episodes=episodes,
                episodes_per_batch=20
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Collect results
            results[name] = {
                'training_time': training_time,
                'final_reward': final_stats['env_stats'].get('overall_recent_avg', 0),
                'total_steps': final_stats['env_stats']['total_steps'],
                'buffer_size': final_stats['buffer_stats']['size'],
                'episodes_per_second': episodes / training_time,
                'steps_per_second': final_stats['env_stats']['total_steps'] / training_time
            }
            
            print(f"✓ {name} completed in {training_time:.2f} seconds")
            print(f"  Final reward: {results[name]['final_reward']:.2f}")
            print(f"  Steps/second: {results[name]['steps_per_second']:.1f}")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = None
        
        finally:
            try:
                agent.env_manager.stop_collection()
                agent.env_manager.executor.shutdown(wait=False)
            except:
                pass
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if results["Single-threaded"] and results["Distributed"]:
        single = results["Single-threaded"]
        distributed = results["Distributed"]
        
        print(f"Training Time:")
        print(f"  Single-threaded: {single['training_time']:.2f}s")
        print(f"  Distributed:     {distributed['training_time']:.2f}s")
        print(f"  Speedup:         {single['training_time'] / distributed['training_time']:.2f}x")
        
        print(f"\nData Collection Rate:")
        print(f"  Single-threaded: {single['steps_per_second']:.1f} steps/s")
        print(f"  Distributed:     {distributed['steps_per_second']:.1f} steps/s")
        print(f"  Improvement:     {distributed['steps_per_second'] / single['steps_per_second']:.2f}x")
        
        print(f"\nFinal Performance:")
        print(f"  Single-threaded reward: {single['final_reward']:.2f}")
        print(f"  Distributed reward:     {distributed['final_reward']:.2f}")
    
    return results


def run_distributed_training_example(episodes: int = 200, num_workers: int = 4):
    """Run a complete distributed training example"""
    print("=" * 80)
    print(f"DISTRIBUTED TRAINING EXAMPLE - {episodes} EPISODES WITH {num_workers} WORKERS")
    print("=" * 80)
    
    # Create configuration
    config = DistributedAgentConfig(
        env_name='ALE/SpaceInvaders-v5',
        num_workers=num_workers,
        memory_size=50000,
        min_replay_size=5000,
        batch_size=64,
        learning_rate=1e-4,
        target_update_freq=2000,
        save_interval=episodes // 4,  # Save 4 times during training
        checkpoint_dir='./results/distributed_checkpoints'
    )
    
    print(f"Configuration:")
    print(f"  Environment: {config.env_name}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Buffer size: {config.memory_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Create agent
    agent = DistributedDQNAgent(config, num_workers=num_workers)
    
    # Track statistics during training
    stats_history = []
    
    try:
        print(f"\nStarting distributed training...")
        start_time = time.time()
        
        # Option 1: Batch-based training (more controlled)
        final_stats = agent.train_batch_collection(
            total_episodes=episodes,
            episodes_per_batch=episodes // 10  # 10 batches
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Training completed in {total_time:.2f} seconds!")
        
        # Print final statistics
        print("\nFinal Statistics:")
        print(f"  Total episodes: {final_stats['final_episodes']}")
        print(f"  Total steps: {final_stats['env_stats']['total_steps']}")
        print(f"  Average reward: {final_stats['env_stats'].get('overall_recent_avg', 0):.2f}")
        print(f"  Buffer size: {final_stats['buffer_stats']['size']}")
        print(f"  Episodes/second: {episodes / total_time:.2f}")
        print(f"  Steps/second: {final_stats['env_stats']['total_steps'] / total_time:.1f}")
        
        # Test the trained agent
        print(f"\nTesting trained agent...")
        test_results = agent.test_distributed(num_episodes=10)
        print(f"Test results:")
        print(f"  Average reward: {test_results['avg_reward']:.2f} ± {test_results['std_reward']:.2f}")
        print(f"  Max reward: {test_results['max_reward']:.2f}")
        print(f"  Min reward: {test_results['min_reward']:.2f}")
        
        # Save final statistics for plotting
        stats_history.append(final_stats)
        
        return final_stats, stats_history
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None, stats_history
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, stats_history
        
    finally:
        try:
            agent.env_manager.stop_collection()
            agent.env_manager.executor.shutdown(wait=False)
        except:
            pass


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Distributed RL Example for Space Invaders')
    parser.add_argument('--mode', choices=['train', 'compare', 'test'], default='train',
                       help='Mode to run: train, compare, or test')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of episodes to train (default: 200)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting results')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Running distributed training example...")
        final_stats, stats_history = run_distributed_training_example(
            episodes=args.episodes, 
            num_workers=args.workers
        )
        
        if final_stats and not args.no_plot:
            plot_distributed_training_results(stats_history)
    
    elif args.mode == 'compare':
        print("Running single-threaded vs distributed comparison...")
        compare_single_vs_distributed(episodes=args.episodes)
    
    elif args.mode == 'test':
        print("Running system test...")
        try:
            # Try to import the test function
            sys.path.append(parent_dir)  # Make sure parent is in path
            from tests.run_all_tests import run_quick_smoke_test
            success = run_quick_smoke_test()
            if success:
                print("✓ System test passed!")
            else:
                print("✗ System test failed!")
                return 1
        except ImportError as e:
            print(f"Could not import test functions: {e}")
            print("Running basic functionality test instead...")
            
            # Basic functionality test
            try:
                config = DistributedAgentConfig(num_workers=1, memory_size=100)
                agent = DistributedDQNAgent(config, num_workers=1)
                print("✓ Distributed agent created successfully")
                agent.env_manager.stop_collection()
                agent.env_manager.executor.shutdown(wait=False)
                print("✓ Basic functionality test passed!")
            except Exception as test_e:
                print(f"✗ Basic functionality test failed: {test_e}")
                return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()