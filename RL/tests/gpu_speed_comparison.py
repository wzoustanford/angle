#!/usr/bin/env python3
"""
Speed comparison: GPU vs CPU training for Space Invaders DQN
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent

def run_speed_test(device_type: str, num_episodes: int = 3):
    """Run speed test for specified device"""
    print(f"\n=== Testing {device_type.upper()} Performance ===")
    
    # Configure for speed test - very small and fast
    config = AgentConfig()
    config.device = device_type
    config.epsilon_start = 0.1  # Less exploration for consistent timing
    config.epsilon_end = 0.1
    config.batch_size = 16  # Smaller batch
    config.memory_size = 2000  # Smaller buffer for faster testing
    config.min_replay_size = 200  # Lower threshold
    config.policy_update_interval = 2  # More frequent updates
    config.target_update_freq = 200
    config.save_interval = 999999  # Disable saving
    
    # Create agent
    agent = DQNAgent(config)
    print(f"Using device: {agent.device}")
    print(f"Network device: {next(agent.q_network.parameters()).device}")
    
    # Warm up - let buffer fill a bit
    print("Warming up...")
    warmup_start = time.time()
    agent.train(2)  # 2 episodes warmup
    warmup_time = time.time() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s")
    
    # Actual speed test
    print(f"Running {num_episodes} episodes for timing...")
    start_time = time.time()
    episode_rewards, losses = agent.train(num_episodes)
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_episode = total_time / num_episodes
    
    # Get some stats
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    avg_loss = np.mean(losses) if losses else 0
    
    print(f"\n{device_type.upper()} Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per episode: {time_per_episode:.2f}s")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Training steps: {agent.steps_done}")
    
    return {
        'device': device_type,
        'total_time': total_time,
        'time_per_episode': time_per_episode,
        'avg_reward': avg_reward,
        'avg_loss': avg_loss,
        'steps': agent.steps_done
    }

def compare_devices():
    """Compare GPU vs CPU performance"""
    print("Space Invaders DQN: GPU vs CPU Speed Comparison")
    print("=" * 60)
    
    num_episodes = 5  # Fewer episodes for faster testing
    
    results = {}
    
    # Test GPU if available
    if torch.cuda.is_available():
        try:
            results['gpu'] = run_speed_test('cuda', num_episodes)
        except Exception as e:
            print(f"GPU test failed: {e}")
            results['gpu'] = None
    else:
        print("CUDA not available, skipping GPU test")
        results['gpu'] = None
    
    # Test CPU
    try:
        results['cpu'] = run_speed_test('cpu', num_episodes)
    except Exception as e:
        print(f"CPU test failed: {e}")
        results['cpu'] = None
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if results['gpu'] and results['cpu']:
        gpu_time = results['gpu']['time_per_episode']
        cpu_time = results['cpu']['time_per_episode']
        speedup = cpu_time / gpu_time
        
        print(f"GPU time per episode: {gpu_time:.2f}s")
        print(f"CPU time per episode: {cpu_time:.2f}s")
        print(f"GPU speedup: {speedup:.2f}x faster than CPU")
        
        if speedup > 1.5:
            print("🚀 Significant GPU acceleration achieved!")
        elif speedup > 1.1:
            print("✅ Moderate GPU acceleration")
        else:
            print("⚠️  Limited or no GPU acceleration")
            
        # Memory usage info
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak GPU memory usage: {memory_gb:.2f}GB")
    
    elif results['gpu']:
        print(f"GPU time per episode: {results['gpu']['time_per_episode']:.2f}s")
        print("CPU test failed")
    elif results['cpu']:
        print(f"CPU time per episode: {results['cpu']['time_per_episode']:.2f}s")
        print("GPU not available or failed")
    else:
        print("Both tests failed!")
    
    return results

if __name__ == "__main__":
    compare_devices()